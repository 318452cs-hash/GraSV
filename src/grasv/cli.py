from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import default_encoder_path, default_scorer_path
from .pipeline import run_grasv_inference


LOGGER = logging.getLogger("grasv.cli")
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def _path_or_none(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path) if path.exists() else None


def _configure_logging(log_level: str, log_file: str | None = None) -> None:
    level = getattr(logging, str(log_level or "INFO").upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grasv",
        description="GraSV structural variant inference from long-read BAMs or precomputed signatures.",
    )
    subparsers = parser.add_subparsers(dest="command")
    infer_parser = subparsers.add_parser("infer", help="Run GraSV inference.")
    add_infer_args(infer_parser)
    return parser


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--platform", required=True, choices=["ont", "ccs", "clr"])
    parser.add_argument("--bam-path", "--bam_path", dest="bam_path", default=None, help="Indexed BAM path.")
    parser.add_argument(
        "--signatures-pkl",
        "--signatures_pkl",
        dest="signatures_pkl",
        default=None,
        help="Optional precomputed signatures.pkl. Pass either this or --bam-path.",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--output-vcf", "--output_vcf", dest="output_vcf", default=None)
    parser.add_argument("--chrom", default=None, help="Optional chromosome/contig restriction, for example 22 or chr22.")
    parser.add_argument("--start", type=int, default=None, help="Optional 0-based region start used with --chrom.")
    parser.add_argument("--end", type=int, default=None, help="Optional 0-based region end used with --chrom.")
    parser.add_argument("--model-path", "--model_path", dest="model_path", default=str(default_encoder_path()))
    parser.add_argument(
        "--cluster-scorer-path",
        "--cluster_scorer_path",
        dest="cluster_scorer_path",
        default=_path_or_none(default_scorer_path()),
    )
    parser.add_argument("--disable-default-scorer", "--disable_default_scorer", action="store_true")
    parser.add_argument("--domain", default="auto", choices=["auto", "real", "sim"])
    parser.add_argument("--global-coverage", "--global_coverage", dest="global_coverage", type=float, default=None)
    parser.add_argument("--include-tra", "--include_tra", dest="include_tra", action="store_true")
    parser.add_argument("--save-signatures-path", "--save_signatures_path", dest="save_signatures_path", default=None)
    parser.add_argument("--save-call-features-path", "--save_call_features_path", dest="save_call_features_path", default=None)
    parser.add_argument(
        "--cluster-scorer-threshold",
        "--cluster_scorer_threshold",
        dest="cluster_scorer_threshold",
        type=float,
        default=None,
    )

    parser.add_argument("--input-dim", "--input_dim", dest="input_dim", type=int, default=27)
    parser.add_argument("--embed-dim", "--embed_dim", dest="embed_dim", type=int, default=128)
    parser.add_argument("--hidden-dims", "--hidden_dims", dest="hidden_dims", type=int, nargs="*", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1024)
    parser.add_argument("--coverage-bin-size", "--coverage_bin_size", dest="coverage_bin_size", type=int, default=1000)
    parser.add_argument("--min-svlen", "--min_svlen", dest="min_svlen", type=int, default=20)
    parser.add_argument("--split-alleles", "--split_alleles", dest="split_alleles", action="store_true")
    parser.add_argument(
        "--length-ratio-threshold",
        "--length_ratio_threshold",
        dest="length_ratio_threshold",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Parallel worker count. Used for BAM signature extraction and chromosome-level clustering.",
    )
    parser.add_argument("--region-size", "--region_size", dest="region_size", type=int, default=5_000_000)
    parser.add_argument("--min-sv-size", "--min_sv_size", dest="min_sv_size", type=int, default=30)
    parser.add_argument("--max-sv-size", "--max_sv_size", dest="max_sv_size", type=int, default=100000)
    parser.add_argument("--min-mapq", "--min_mapq", dest="min_mapq", type=int, default=20)
    parser.add_argument("--min-read-len", "--min_read_len", dest="min_read_len", type=int, default=500)
    parser.add_argument("--min-siglength", "--min_siglength", dest="min_siglength", type=int, default=30)
    parser.add_argument("--merge-del-threshold", "--merge_del_threshold", dest="merge_del_threshold", type=int, default=0)
    parser.add_argument("--merge-ins-threshold", "--merge_ins_threshold", dest="merge_ins_threshold", type=int, default=100)
    parser.add_argument("--max-split-parts", "--max_split_parts", dest="max_split_parts", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-embeddings", "--save_embeddings", dest="save_embeddings", action="store_true")
    parser.add_argument("--verbose-progress", "--verbose_progress", action="store_true")
    parser.add_argument("--progress-json-path", "--progress_json_path", dest="progress_json_path", default=None)
    parser.add_argument(
        "--log-level",
        "--log_level",
        dest="log_level",
        type=str.upper,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console/file logging verbosity.",
    )
    parser.add_argument(
        "--log-file",
        "--log_file",
        dest="log_file",
        default=None,
        help="Optional path for a full run log.",
    )


def run_infer(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level, args.log_file)
    if not args.bam_path and not args.signatures_pkl:
        raise SystemExit("Pass either --bam-path or --signatures-pkl.")
    if not args.model_path:
        raise SystemExit("No encoder checkpoint available. Pass --model-path explicitly.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_vcf = Path(args.output_vcf) if args.output_vcf else output_dir / "calls.vcf"
    args.output_vcf = str(output_vcf)
    args.output_dir = str(output_dir)
    args.data_path = args.signatures_pkl
    args.chrom = getattr(args, "chrom", None)
    args.start = getattr(args, "start", None)
    args.end = getattr(args, "end", None)
    args.coverage_cache_path = getattr(args, "coverage_cache_path", None)
    args.save_call_features_path = args.save_call_features_path
    args.save_signatures_path = args.save_signatures_path

    LOGGER.info(
        "starting GraSV inference platform=%s input=%s output_dir=%s processes=%s",
        args.platform,
        args.signatures_pkl or args.bam_path,
        args.output_dir,
        args.processes,
    )
    result = run_grasv_inference(args)
    LOGGER.info(
        "finished GraSV inference n_signatures=%s n_clusters=%s n_calls=%s vcf=%s metadata=%s",
        result.n_signatures,
        result.n_clusters,
        result.n_calls,
        output_vcf,
        result.metadata_path,
    )
    print(f"vcf={output_vcf}")
    print(f"metadata={result.metadata_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command in {None, "infer"}:
        return run_infer(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

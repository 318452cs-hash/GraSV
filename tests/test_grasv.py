from __future__ import annotations

from grasv import select_grasv_preset
from grasv.cli import build_arg_parser


def test_import_and_preset() -> None:
    preset = select_grasv_preset("ont", 5.0)
    assert preset.platform == "ont"
    assert preset.graph_variant == "merge_b"


def test_cli_parses_bam_infer_args() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "infer",
            "--platform",
            "ccs",
            "--bam-path",
            "sample.bam",
            "--chrom",
            "22",
            "--output-dir",
            "outputs/sample",
        ]
    )
    assert args.command == "infer"
    assert args.platform == "ccs"
    assert args.bam_path == "sample.bam"
    assert args.chrom == "22"

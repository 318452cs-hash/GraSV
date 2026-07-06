from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import pysam
except ImportError:
    pysam = None

from .signature import guess_svtype_from_signature, normalize_svtype
from .utils import get_device


DEFAULT_INPUT_DIM = 27
DEFAULT_EMBED_DIM = 128
DEFAULT_ENCODER_HIDDEN_DIMS = [128, 256, 256, 128]


class GraSVEncoder(nn.Module):
    def __init__(self, input_dim=DEFAULT_INPUT_DIM, embed_dim=DEFAULT_EMBED_DIM, hidden_dims=None, dropout=0.1):
        super().__init__()
        hidden_dims = hidden_dims or DEFAULT_ENCODER_HIDDEN_DIMS

        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if i == 0 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embed_dim))

        self.encoder = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.encoder(x)


@dataclass
class SimpleCall:
    contig: str
    start: int
    end: int
    svtype: str
    svlen: int
    support: int
    prob: float
    filter_status: str = "PASS"
    partner_contig: str | None = None
    call_id: str | None = None
    scorer_prob: float | None = None
    score_features: Dict[str, float] | None = None


@dataclass(frozen=True)
class ClusterMetrics:
    start_std: float
    end_std: float
    length_cv: float
    mean_cosine: float
    median_mapq: float


def default_inference_config() -> Dict[str, Any]:
    return {
        "clustering_method": "hybrid",
        "k_neighbors": 12,
        "k_neighbors_del": None,
        "k_neighbors_ins": None,
        "k_neighbors_dup": None,
        "k_neighbors_inv": None,
        "k_neighbors_tra": None,
        "similarity_threshold": 0.82,
        "similarity_threshold_del": None,
        "similarity_threshold_ins": None,
        "similarity_threshold_dup": None,
        "similarity_threshold_inv": None,
        "similarity_threshold_tra": None,
        "n_clusters": "auto",
        "max_clusters": 20,
        "min_cluster_size": 2,
        "max_position_gap": 1000,
        "max_position_gap_del": None,
        "max_position_gap_ins": None,
        "max_position_gap_dup": None,
        "max_position_gap_inv": None,
        "max_position_gap_tra": None,
        "max_group_size": 4000,
        "split_alleles": False,
        "length_ratio_threshold": 1.5,
        "disable_compactness_filter": False,
        "min_cluster_cosine": 0.75,
        "max_cluster_start_std": 220.0,
        "max_cluster_end_std": 320.0,
        "max_cluster_length_cv": 0.55,
        "min_cluster_median_mapq": 20.0,
        "compactness_start_scale": 1.0,
        "compactness_end_scale": 1.0,
        "compactness_length_cv_scale": 1.0,
        "compactness_cosine_relax": 0.0,
        "scorer_prefilter_mode": "strict",
        "candidate_min_support": 1,
        "cluster_scorer_path": None,
        "cluster_scorer_threshold": None,
        "coverage_calibration_path": None,
        "enable_rule_postfilter": False,
        "rule_postfilter_activation_coverage_x": None,
        "coverage_cache_path": None,
        "global_coverage": None,
        "min_svlen": 20,
        "min_support": 3,
        "min_support_del": 20,
        "min_support_ins": 10,
        "min_support_dup": 6,
        "min_support_inv": 6,
        "min_support_tra": 6,
    }


def coverage_bucket(global_coverage: float | None) -> str:
    if global_coverage is None or not math.isfinite(float(global_coverage)) or float(global_coverage) <= 0.0:
        return "unknown"
    if float(global_coverage) < 12.0:
        return "low"
    if float(global_coverage) < 30.0:
        return "mid"
    return "high"


def estimate_global_coverage(
    bam_path: str | None,
    *,
    sample_reads: int = 5000,
) -> float | None:
    if not bam_path or pysam is None:
        return None

    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam_file:
            total_reference_bases = float(sum(int(length) for length in bam_file.lengths if int(length) > 0))
            if total_reference_bases <= 0.0:
                return None

            try:
                stats = bam_file.get_index_statistics()
                mapped_reads = float(sum(int(stat.mapped) for stat in stats))
            except Exception:
                mapped_reads = 0.0
            if mapped_reads <= 0.0:
                return None

            lengths: List[int] = []
            try:
                iterator = bam_file.fetch(until_eof=True)
            except Exception:
                iterator = iter(())
            for read in iterator:
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                read_len = int(read.query_length or 0)
                if read_len <= 0 and read.reference_end is not None:
                    read_len = max(0, int(read.reference_end) - int(read.reference_start))
                if read_len <= 0:
                    continue
                lengths.append(read_len)
                if len(lengths) >= int(sample_reads):
                    break

            if not lengths:
                return None
            mean_read_length = float(np.mean(np.asarray(lengths, dtype=np.float64)))
            if mean_read_length <= 0.0:
                return None
            return float(mapped_reads * mean_read_length / total_reference_bases)
    except Exception:
        return None
def _infer_encoder_shape_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> tuple[int | None, int | None, List[int] | None]:
    linear_layers: List[tuple[int, torch.Size]] = []
    for key, value in state_dict.items():
        if not key.startswith("encoder.") or not key.endswith(".weight"):
            continue
        if getattr(value, "ndim", 0) != 2:
            continue
        parts = key.split(".")
        if len(parts) != 3 or not parts[1].isdigit():
            continue
        linear_layers.append((int(parts[1]), value.shape))
    linear_layers.sort(key=lambda item: item[0])
    if not linear_layers:
        return None, None, None
    inferred_input_dim = int(linear_layers[0][1][1])
    inferred_hidden_dims = [int(shape[0]) for _idx, shape in linear_layers[:-1]]
    inferred_embed_dim = int(linear_layers[-1][1][0])
    return inferred_input_dim, inferred_embed_dim, inferred_hidden_dims


def load_model(args: argparse.Namespace, device: torch.device) -> GraSVEncoder:
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    if "encoder_state_dict" in checkpoint:
        encoder_state = checkpoint["encoder_state_dict"]
    elif "model_state_dict" in checkpoint:
        encoder_state = {
            key[len("encoder."):]: value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("encoder.")
        }
    else:
        raise ValueError("Checkpoint does not contain an encoder state dict.")

    inferred_input_dim, inferred_embed_dim, inferred_hidden_dims = _infer_encoder_shape_from_state_dict(encoder_state)
    input_dim = int(model_config.get("input_dim", inferred_input_dim or args.input_dim))
    embed_dim = int(model_config.get("embed_dim", inferred_embed_dim or args.embed_dim))
    hidden_dims = args.hidden_dims or model_config.get("hidden_dims") or inferred_hidden_dims or [64, 128]
    dropout = model_config.get("dropout", 0.1) if args.dropout is None else args.dropout

    encoder = GraSVEncoder(input_dim=input_dim, embed_dim=embed_dim, hidden_dims=hidden_dims, dropout=dropout)
    encoder.load_state_dict(encoder_state)

    encoder.to(device)
    encoder.eval()
    encoder.input_dim = input_dim
    return encoder


def extract_embeddings(features: np.ndarray, encoder: GraSVEncoder, device: torch.device, batch_size: int) -> np.ndarray:
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(features), batch_size), desc="embed", leave=False):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            embeddings = encoder(batch)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            chunks.append(embeddings.cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, encoder.encoder[-1].out_features), dtype=np.float32)


def split_allele_indices(cluster_sigs: Sequence[Any], length_ratio_threshold: float) -> List[List[int]]:
    if len(cluster_sigs) < 4:
        return [list(range(len(cluster_sigs)))]

    lengths = np.array([max(1, abs(int(getattr(sig, "svlen", 0) or 0))) for sig in cluster_sigs], dtype=np.float32)
    length_ratio = float(lengths.max() / max(1.0, lengths.min()))
    if length_ratio < float(length_ratio_threshold):
        return [list(range(len(cluster_sigs)))]

    order = np.argsort(lengths)
    sorted_lengths = lengths[order]
    log_lengths = np.log1p(sorted_lengths)
    gaps = np.diff(log_lengths)
    if gaps.size == 0:
        return [list(range(len(cluster_sigs)))]

    split_at = int(np.argmax(gaps) + 1)
    left = order[:split_at].tolist()
    right = order[split_at:].tolist()
    if not left or not right:
        return [list(range(len(cluster_sigs)))]

    boundary_ratio = float(sorted_lengths[split_at] / max(1.0, sorted_lengths[split_at - 1]))
    min_gap_ratio = max(1.2, math.sqrt(float(length_ratio_threshold)))
    if boundary_ratio < min_gap_ratio:
        return [list(range(len(cluster_sigs)))]

    groups = [left, right]
    return [group for group in groups if group]


def _median_int(values: Iterable[int]) -> int:
    return int(np.median(np.asarray(list(values), dtype=np.int64)))


def _call_probability(positions: Sequence[int], lengths: Sequence[int], support: int) -> float:
    pos_std = float(np.std(positions)) if len(positions) > 1 else 0.0
    len_std = float(np.std(lengths)) if len(lengths) > 1 else 0.0
    scale = max(1.0, float(np.median(np.abs(lengths))) if lengths else 1.0)
    score = 0.35 + min(0.45, support * 0.05) + max(0.0, 0.2 - pos_std / 500.0 - len_std / (5.0 * scale))
    return float(max(0.01, min(0.99, score)))


def _cluster_source_summary(cluster_sigs: Sequence[Any]) -> Dict[str, float]:
    if not cluster_sigs:
        return {"split_fraction": 0.0, "cigar_fraction": 0.0, "source_diversity": 0.0}

    normalized_sources: List[str] = []
    split_count = 0
    cigar_count = 0
    for sig in cluster_sigs:
        source = str(getattr(sig, "source", "") or "").upper()
        normalized_sources.append(source)
        if "SA" in source or "SPLIT" in source:
            split_count += 1
        if "CIGAR" in source:
            cigar_count += 1

    total = float(len(cluster_sigs))
    return {
        "split_fraction": split_count / total,
        "cigar_fraction": cigar_count / total,
        "source_diversity": float(len(set(normalized_sources))) / total,
    }


def _build_call_score_features(
    *,
    cluster_sigs: Sequence[Any],
    svtype: str,
    start: int,
    end: int,
    svlen: int,
    support: int,
    prob: float,
    metrics: ClusterMetrics,
    platform: str | None,
    global_coverage: float | None,
    strict_support_threshold: int,
    strict_support_met: bool,
    compactness_failures: Sequence[str],
) -> Dict[str, float]:
    event_length = max(1, abs(int(svlen)) or abs(int(end) - int(start)) or 1)
    span = max(1, int(end) - int(start))
    cluster_size = max(1, len(cluster_sigs))
    source_summary = _cluster_source_summary(cluster_sigs)
    cov = None if global_coverage is None else float(global_coverage)
    cov_bucket = coverage_bucket(cov)
    platform_key = (str(platform).strip().lower() if platform else "").lower()
    compactness_failures = tuple(str(reason) for reason in compactness_failures)
    compactness_fail_set = set(compactness_failures)
    return {
        "log_support": math.log1p(max(1, support)),
        "log_cluster_size": math.log1p(cluster_size),
        "support_ratio": float(support) / float(cluster_size),
        "log_svlen": math.log1p(event_length),
        "log_span": math.log1p(span),
        "start_std": float(metrics.start_std),
        "end_std": float(metrics.end_std),
        "start_std_norm": float(metrics.start_std) / float(event_length),
        "end_std_norm": float(metrics.end_std) / float(event_length),
        "length_cv": float(metrics.length_cv),
        "mean_cosine": float(metrics.mean_cosine),
        "median_mapq": float(metrics.median_mapq),
        "prob_base": float(prob),
        "split_fraction": float(source_summary["split_fraction"]),
        "cigar_fraction": float(source_summary["cigar_fraction"]),
        "source_diversity": float(source_summary["source_diversity"]),
        "global_coverage": 0.0 if cov is None or not math.isfinite(cov) else float(cov),
        "log_global_coverage": 0.0 if cov is None or not math.isfinite(cov) or cov <= 0.0 else math.log1p(cov),
        "support_over_global_coverage": 0.0 if cov is None or not math.isfinite(cov) or cov <= 0.0 else float(support) / float(cov),
        "coverage_low": 1.0 if cov_bucket == "low" else 0.0,
        "coverage_mid": 1.0 if cov_bucket == "mid" else 0.0,
        "coverage_high": 1.0 if cov_bucket == "high" else 0.0,
        "platform_ont": 1.0 if platform_key == "ont" else 0.0,
        "platform_ccs": 1.0 if platform_key == "ccs" else 0.0,
        "platform_clr": 1.0 if platform_key == "clr" else 0.0,
        "strict_support_threshold": float(strict_support_threshold),
        "support_margin_to_strict": float(support - strict_support_threshold),
        "below_strict_support": 0.0 if strict_support_met else 1.0,
        "failed_compactness": 1.0 if compactness_failures else 0.0,
        "compactness_fail_count": float(len(compactness_failures)),
        "failed_mapq": 1.0 if "mapq" in compactness_fail_set else 0.0,
        "failed_start_std": 1.0 if "start_std" in compactness_fail_set else 0.0,
        "failed_end_std": 1.0 if "end_std" in compactness_fail_set else 0.0,
        "failed_length_cv": 1.0 if "length_cv" in compactness_fail_set else 0.0,
        "failed_cosine": 1.0 if "cosine" in compactness_fail_set else 0.0,
        "is_del": 1.0 if svtype == "DEL" else 0.0,
        "is_ins": 1.0 if svtype == "INS" else 0.0,
        "is_dup": 1.0 if svtype == "DUP" else 0.0,
        "is_inv": 1.0 if svtype == "INV" else 0.0,
        "is_tra": 1.0 if svtype == "TRA" else 0.0,
    }


def _support_override_for_type(args: argparse.Namespace, svtype: str) -> int:
    override = getattr(args, f"min_support_{svtype.lower()}", None)
    if override is None:
        return int(getattr(args, "min_support", 1))
    return int(override)


def _candidate_support_threshold_for_type(args: argparse.Namespace, svtype: str) -> int:
    strict_threshold = _support_override_for_type(args, svtype)
    mode = str(getattr(args, "scorer_prefilter_mode", "strict") or "strict").lower()
    if mode == "strict":
        return strict_threshold
    candidate_floor = max(1, int(getattr(args, "candidate_min_support", 1)))
    if mode == "balanced":
        return max(candidate_floor, min(strict_threshold, max(2, strict_threshold // 2)))
    return min(strict_threshold, candidate_floor)


def _estimate_cluster_support(cluster_sigs: Sequence[Any]) -> int:
    qnames = {
        str(getattr(sig, "qname", "")).strip()
        for sig in cluster_sigs
        if str(getattr(sig, "qname", "")).strip()
    }
    if qnames:
        return len(qnames)

    explicit_support = 0
    for sig in cluster_sigs:
        try:
            explicit_support += max(1, int(getattr(sig, "support", 1) or 1))
        except Exception:
            explicit_support += 1
    return max(1, explicit_support)


def _call_length(call: SimpleCall) -> int:
    if call.svtype == "TRA":
        return 0
    return max(1, abs(int(call.svlen)))


def _signature_event_length(sig: Any) -> int:
    start = int(getattr(sig, "tstart", 0))
    end = int(getattr(sig, "tend", start + 1))
    svlen = abs(int(getattr(sig, "svlen", 0) or 0))
    return max(1, svlen, abs(end - start))


def _compute_cluster_metrics(cluster_sigs: Sequence[Any], cluster_embeddings: np.ndarray) -> ClusterMetrics:
    starts = np.asarray([int(getattr(sig, "tstart", 0)) for sig in cluster_sigs], dtype=np.float32)
    ends = np.asarray([int(getattr(sig, "tend", getattr(sig, "tstart", 0) + 1)) for sig in cluster_sigs], dtype=np.float32)
    lengths = np.asarray([_signature_event_length(sig) for sig in cluster_sigs], dtype=np.float32)
    mapqs = np.asarray([float(getattr(sig, "mapq", 0.0) or 0.0) for sig in cluster_sigs], dtype=np.float32)

    start_std = float(np.std(starts)) if len(starts) > 1 else 0.0
    end_std = float(np.std(ends)) if len(ends) > 1 else 0.0
    length_scale = max(1.0, float(np.median(lengths))) if len(lengths) > 0 else 1.0
    length_cv = float(np.std(lengths) / length_scale) if len(lengths) > 1 else 0.0
    median_mapq = float(np.median(mapqs)) if len(mapqs) > 0 else 0.0

    if cluster_embeddings.size == 0:
        mean_cosine = 0.0
    else:
        normalized = np.asarray(cluster_embeddings, dtype=np.float32)
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        normalized = normalized / np.clip(norms, 1e-12, None)
        centroid = normalized.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= 1e-12:
            mean_cosine = 0.0
        else:
            centroid = centroid / centroid_norm
            mean_cosine = float(np.mean(np.clip(normalized @ centroid, -1.0, 1.0)))

    return ClusterMetrics(
        start_std=start_std,
        end_std=end_std,
        length_cv=length_cv,
        mean_cosine=mean_cosine,
        median_mapq=median_mapq,
    )


def _compactness_limits(args: argparse.Namespace, svtype: str, support: int) -> Dict[str, float]:
    support_bonus = min(5.0, math.log2(max(2, support)))
    start_scale = max(0.1, float(args.compactness_start_scale))
    end_scale = max(0.1, float(args.compactness_end_scale))
    length_cv_scale = max(0.1, float(args.compactness_length_cv_scale))
    cosine_relax = max(0.0, float(args.compactness_cosine_relax))
    base = {
        # DEL/INS need slightly looser positional compactness than the first pass,
        # otherwise we lose high-support true calls with modest breakpoint jitter.
        "INS": {"start_std": 50.0, "end_std": float("inf"), "length_cv": 0.35, "cosine": 0.84, "mapq": 25.0},
        "DEL": {"start_std": 90.0, "end_std": 140.0, "length_cv": 0.28, "cosine": 0.87, "mapq": 30.0},
        "DUP": {"start_std": 100.0, "end_std": 150.0, "length_cv": 0.32, "cosine": 0.85, "mapq": 25.0},
        "INV": {"start_std": 140.0, "end_std": 200.0, "length_cv": 0.40, "cosine": 0.82, "mapq": 20.0},
        "TRA": {"start_std": 200.0, "end_std": float("inf"), "length_cv": float("inf"), "cosine": 0.80, "mapq": 20.0},
        "UNK": {"start_std": 90.0, "end_std": 140.0, "length_cv": 0.35, "cosine": 0.84, "mapq": 25.0},
    }.get(svtype, {"start_std": 90.0, "end_std": 140.0, "length_cv": 0.35, "cosine": 0.84, "mapq": 25.0})

    max_end_std = float("inf")
    if math.isfinite(base["end_std"]):
        max_end_std = min(float(args.max_cluster_end_std), base["end_std"] * end_scale + 15.0 * support_bonus)

    requested_min_mapq = float(args.min_cluster_median_mapq)
    min_mapq = 0.0 if requested_min_mapq <= 0.0 else max(requested_min_mapq, base["mapq"])
    adjusted_cosine = max(0.0, base["cosine"] - cosine_relax)
    adjusted_length_cv = float("inf") if not math.isfinite(base["length_cv"]) else base["length_cv"] * length_cv_scale

    return {
        "max_start_std": min(float(args.max_cluster_start_std), base["start_std"] * start_scale + 10.0 * support_bonus),
        "max_end_std": max_end_std,
        "max_length_cv": min(float(args.max_cluster_length_cv), adjusted_length_cv),
        "min_cosine": max(float(args.min_cluster_cosine), adjusted_cosine - 0.01 * support_bonus),
        "min_mapq": min_mapq,
    }


def _passes_compactness_filter(
    cluster_sigs: Sequence[Any],
    cluster_embeddings: np.ndarray,
    svtype: str,
    support: int,
    args: argparse.Namespace,
) -> Tuple[bool, ClusterMetrics, Tuple[str, ...]]:
    metrics = _compute_cluster_metrics(cluster_sigs, cluster_embeddings)
    if args.disable_compactness_filter:
        return True, metrics, ()

    limits = _compactness_limits(args, svtype, support)
    failures: List[str] = []
    if metrics.mean_cosine < limits["min_cosine"]:
        failures.append("cosine")
    if limits["min_mapq"] > 0.0 and metrics.median_mapq < limits["min_mapq"]:
        failures.append("mapq")
    if metrics.start_std > limits["max_start_std"]:
        failures.append("start_std")
    if math.isfinite(limits["max_end_std"]) and metrics.end_std > limits["max_end_std"]:
        failures.append("end_std")
    if math.isfinite(limits["max_length_cv"]) and metrics.length_cv > limits["max_length_cv"]:
        failures.append("length_cv")
    return not failures, metrics, tuple(failures)


def _passes_scorer_prefilter_guard(metrics: ClusterMetrics, support: int, args: argparse.Namespace) -> Tuple[bool, Tuple[str, ...]]:
    mode = str(getattr(args, "scorer_prefilter_mode", "strict") or "strict").lower()
    if mode == "strict":
        return True, ()

    reasons: List[str] = []
    if support < max(1, int(getattr(args, "candidate_min_support", 1))):
        reasons.append("candidate_support")
    if metrics.mean_cosine < max(0.10, float(args.min_cluster_cosine) - 0.35):
        reasons.append("guard_cosine")
    if metrics.start_std > float(args.max_cluster_start_std) * 4.0:
        reasons.append("guard_start_std")
    if metrics.end_std > float(args.max_cluster_end_std) * 4.0:
        reasons.append("guard_end_std")
    if metrics.length_cv > float(args.max_cluster_length_cv) * 3.0:
        reasons.append("guard_length_cv")
    return not reasons, tuple(reasons)


def _calls_should_merge(existing: SimpleCall, candidate: SimpleCall) -> bool:
    if (
        existing.contig != candidate.contig
        or existing.svtype != candidate.svtype
        or (existing.partner_contig or "") != (candidate.partner_contig or "")
    ):
        return False

    start_dist = abs(candidate.start - existing.start)
    if existing.svtype == "TRA":
        return start_dist <= 1000

    len_a = _call_length(existing)
    len_b = _call_length(candidate)
    length_ratio = min(len_a, len_b) / max(len_a, len_b)
    if existing.svtype != "TRA" and length_ratio < 0.5:
        return False

    if existing.svtype == "INS":
        return start_dist <= max(150, min(len_a, len_b))

    end_dist = abs(candidate.end - existing.end)
    distance_limit = max(200, min(len_a, len_b))
    overlap = max(0, min(existing.end, candidate.end) - max(existing.start, candidate.start))
    shorter_span = max(1, min(existing.end - existing.start, candidate.end - candidate.start))
    overlap_ratio = overlap / shorter_span
    return (start_dist <= distance_limit and end_dist <= distance_limit) or overlap_ratio >= 0.6


def _call_passes_rule_postfilter(call: SimpleCall, args: argparse.Namespace) -> bool:
    svtype = str(call.svtype or "UNK").upper()
    support = int(call.support or 0)
    if support < _support_override_for_type(args, svtype):
        return False

    features = dict(call.score_features or {})
    metrics = ClusterMetrics(
        start_std=float(features.get("start_std", 0.0) or 0.0),
        end_std=float(features.get("end_std", 0.0) or 0.0),
        length_cv=float(features.get("length_cv", 0.0) or 0.0),
        mean_cosine=float(features.get("mean_cosine", 0.0) or 0.0),
        median_mapq=float(features.get("median_mapq", 0.0) or 0.0),
    )
    limits = _compactness_limits(args, svtype, support)
    if metrics.mean_cosine < limits["min_cosine"]:
        return False
    if limits["min_mapq"] > 0.0 and metrics.median_mapq < limits["min_mapq"]:
        return False
    if metrics.start_std > limits["max_start_std"]:
        return False
    if math.isfinite(limits["max_end_std"]) and metrics.end_std > limits["max_end_std"]:
        return False
    if math.isfinite(limits["max_length_cv"]) and metrics.length_cv > limits["max_length_cv"]:
        return False
    return True


def apply_rule_postfilter(calls: Sequence[SimpleCall], args: argparse.Namespace) -> Tuple[List[SimpleCall], Dict[str, int]]:
    kept: List[SimpleCall] = []
    stats: Counter[str] = Counter()
    for call in calls:
        if _call_passes_rule_postfilter(call, args):
            kept.append(call)
            continue
        stats["rule_postfilter"] += 1
        stats[f"rule_postfilter_{str(call.svtype or 'UNK').lower()}"] += 1
    return kept, dict(stats)


def generate_calls(
    clusters: Sequence[Sequence[int]],
    signatures: Sequence[Any],
    embeddings: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[List[SimpleCall], Dict[str, int]]:
    calls: List[SimpleCall] = []
    filter_stats: Counter[str] = Counter()

    for cluster in clusters:
        cluster_sigs = [signatures[idx] for idx in cluster]
        cluster_embeddings = embeddings[np.asarray(cluster, dtype=np.int64)]
        svtype = normalize_svtype(getattr(cluster_sigs[0], "svtype", None) or guess_svtype_from_signature(cluster_sigs[0])) or "INS"

        allele_groups = [list(range(len(cluster_sigs)))]
        if args.split_alleles and svtype in {"INS", "DEL", "DUP", "INV"}:
            allele_groups = split_allele_indices(cluster_sigs, args.length_ratio_threshold)

        for allele_indices in allele_groups:
            allele = [cluster_sigs[idx] for idx in allele_indices]
            allele_embeddings = cluster_embeddings[np.asarray(allele_indices, dtype=np.int64)]
            support = _estimate_cluster_support(allele)
            strict_support_threshold = _support_override_for_type(args, svtype)
            candidate_support_threshold = _candidate_support_threshold_for_type(args, svtype)
            if support < candidate_support_threshold:
                filter_stats["support"] += 1
                filter_stats[f"support_{svtype.lower()}"] += 1
                continue

            strict_compactness_passed, metrics, reasons = _passes_compactness_filter(
                cluster_sigs=allele,
                cluster_embeddings=allele_embeddings,
                svtype=svtype,
                support=support,
                args=args,
            )
            strict_support_met = support >= strict_support_threshold

            prefilter_mode = str(getattr(args, "scorer_prefilter_mode", "strict") or "strict").lower()
            if prefilter_mode == "strict":
                if not strict_support_met:
                    filter_stats["support"] += 1
                    filter_stats[f"support_{svtype.lower()}"] += 1
                    continue
                if not strict_compactness_passed:
                    filter_stats["compactness"] += 1
                    filter_stats[f"compactness_{svtype.lower()}"] += 1
                    for reason in reasons:
                        filter_stats[f"compactness_{reason}"] += 1
                    continue
            else:
                guard_passed, guard_reasons = _passes_scorer_prefilter_guard(metrics, support, args)
                if not guard_passed:
                    filter_stats["prefilter_guard"] += 1
                    filter_stats[f"prefilter_guard_{svtype.lower()}"] += 1
                    for reason in guard_reasons:
                        filter_stats[f"prefilter_guard_{reason}"] += 1
                    continue
                if not strict_support_met:
                    filter_stats["rescued_below_strict_support"] += 1
                    filter_stats[f"rescued_below_strict_support_{svtype.lower()}"] += 1
                if not strict_compactness_passed:
                    filter_stats["rescued_failed_compactness"] += 1
                    filter_stats[f"rescued_failed_compactness_{svtype.lower()}"] += 1
                    for reason in reasons:
                        filter_stats[f"rescued_compactness_{reason}"] += 1

            contig = str(getattr(allele[0], "contig", "chrNA"))
            starts = [int(getattr(sig, "tstart", 0)) for sig in allele]
            ends = [int(getattr(sig, "tend", getattr(sig, "tstart", 0) + 1)) for sig in allele]
            lengths = [max(1, abs(int(getattr(sig, "svlen", 0) or 0))) for sig in allele]

            start = _median_int(starts)
            end = _median_int(ends)
            svlen = _median_int(lengths)

            partner_contig = None
            if svtype == "INS":
                end = start + 1
            elif svtype == "TRA":
                end = start + 1
                partner_counts = Counter(str(partner) for sig in allele for partner in (getattr(sig, "sa_contigs", None) or []))
                partner_contig = partner_counts.most_common(1)[0][0] if partner_counts else None
                svlen = 0
            else:
                if end <= start:
                    end = start + max(1, svlen)
                svlen = max(args.min_svlen if svtype != "TRA" else 0, svlen)

            if svtype != "TRA" and svlen < args.min_svlen:
                continue

            prob = max(_call_probability(starts, lengths, support), min(0.99, metrics.mean_cosine))
            signed_svlen = svlen if svtype != "DEL" else -svlen
            score_features = _build_call_score_features(
                cluster_sigs=allele,
                svtype=svtype,
                start=start,
                end=end,
                svlen=signed_svlen,
                support=support,
                prob=prob,
                metrics=metrics,
                platform=getattr(args, "platform", None),
                global_coverage=getattr(args, "global_coverage", None),
                strict_support_threshold=strict_support_threshold,
                strict_support_met=strict_support_met,
                compactness_failures=reasons,
            )
            calls.append(
                SimpleCall(
                    contig=contig,
                    start=start,
                    end=end,
                    svtype=svtype,
                    svlen=signed_svlen,
                    support=support,
                    prob=prob,
                    partner_contig=partner_contig,
                    score_features=score_features,
                )
            )

    calls.sort(key=lambda call: (call.contig, call.start, call.svtype, call.partner_contig or ""))
    return deduplicate_calls(calls), dict(filter_stats)


def deduplicate_calls(calls: Sequence[SimpleCall]) -> List[SimpleCall]:
    deduped: List[SimpleCall] = []
    active_by_key: Dict[Tuple[str, str, str], deque[int]] = defaultdict(deque)
    active_window = 1500

    for call in calls:
        key = (call.contig, call.svtype, call.partner_contig or "")
        active = active_by_key[key]

        while active and call.start - deduped[active[0]].start > active_window:
            active.popleft()

        merged = False
        for idx in reversed(active):
            existing = deduped[idx]
            if not _calls_should_merge(existing, call):
                continue

            merged = True
            if (call.support, call.prob) > (existing.support, existing.prob):
                deduped[idx] = call
            break

        if not merged:
            deduped.append(call)
            active.append(len(deduped) - 1)

    deduped.sort(key=lambda call: (call.contig, call.start, call.svtype, call.partner_contig or ""))
    return deduped

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def package_root() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return package_root().parents[1]


def model_dir() -> Path:
    env_path = os.environ.get("GRASV_MODEL_DIR")
    if env_path:
        return Path(os.path.expanduser(env_path)).resolve()
    for candidate in (project_root() / "models", Path.cwd() / "models"):
        if candidate.exists():
            return candidate.resolve()
    return project_root() / "models"


def default_encoder_path() -> Path:
    return model_dir() / "grasv_encoder.pt"


def default_scorer_path() -> Path:
    return model_dir() / "grasv_scorer.pt"


@dataclass(frozen=True)
class GraphParams:
    k_neighbors: int
    similarity_threshold: float
    max_position_gap: int
    max_group_size: int = 4000
    min_cluster_size: int = 2


@dataclass(frozen=True)
class PostfilterParams:
    cluster_scorer_threshold: float
    enable_rule_postfilter: bool
    min_support_del: int
    min_support_ins: int
    min_support_dup: int
    min_support_inv: int
    compactness_start_scale: float
    compactness_end_scale: float
    compactness_length_cv_scale: float
    compactness_cosine_relax: float
    min_cluster_median_mapq: float = 0.0


@dataclass(frozen=True)
class SupportPoint:
    coverage: float
    graph_variant: str
    graph: GraphParams
    postfilter: PostfilterParams


@dataclass(frozen=True)
class GraSVPipelinePreset:
    name: str
    platform: str
    requested_coverage: float
    graph_anchor_coverage: float
    graph_variant: str
    graph: GraphParams
    postfilter: PostfilterParams


SUPPORT_POINTS: Dict[str, Sequence[SupportPoint]] = {
    "ccs": (
        SupportPoint(
            coverage=5.0,
            graph_variant="merge_a",
            graph=GraphParams(k_neighbors=15, similarity_threshold=0.76, max_position_gap=1200),
            postfilter=PostfilterParams(0.51, False, 1, 1, 1, 1, 1.50, 1.50, 1.30, 0.06),
        ),
        SupportPoint(
            coverage=8.0,
            graph_variant="base_graph",
            graph=GraphParams(k_neighbors=12, similarity_threshold=0.80, max_position_gap=900),
            postfilter=PostfilterParams(0.51, True, 1, 1, 1, 1, 1.65, 1.65, 1.40, 0.08),
        ),
        SupportPoint(
            coverage=10.0,
            graph_variant="base_graph",
            graph=GraphParams(k_neighbors=12, similarity_threshold=0.80, max_position_gap=900),
            postfilter=PostfilterParams(0.51, True, 2, 2, 2, 2, 1.50, 1.50, 1.30, 0.06),
        ),
        SupportPoint(
            coverage=12.0,
            graph_variant="merge_a",
            graph=GraphParams(k_neighbors=15, similarity_threshold=0.80, max_position_gap=1100),
            postfilter=PostfilterParams(0.00, True, 2, 2, 3, 3, 1.40, 1.40, 1.28, 0.05),
        ),
        SupportPoint(
            coverage=15.0,
            graph_variant="merge_a",
            graph=GraphParams(k_neighbors=15, similarity_threshold=0.80, max_position_gap=1100),
            postfilter=PostfilterParams(0.00, True, 3, 2, 3, 3, 1.10, 1.10, 1.08, 0.01),
        ),
        SupportPoint(
            coverage=18.0,
            graph_variant="merge_a",
            graph=GraphParams(k_neighbors=15, similarity_threshold=0.80, max_position_gap=1100),
            postfilter=PostfilterParams(0.00, True, 4, 3, 3, 3, 1.10, 1.10, 1.07, 0.01),
        ),
        SupportPoint(
            coverage=20.0,
            graph_variant="base_graph",
            graph=GraphParams(k_neighbors=12, similarity_threshold=0.84, max_position_gap=800),
            postfilter=PostfilterParams(0.00, True, 5, 3, 3, 3, 1.15, 1.15, 1.10, 0.03),
        ),
        SupportPoint(
            coverage=23.0,
            graph_variant="merge_a",
            graph=GraphParams(k_neighbors=15, similarity_threshold=0.80, max_position_gap=1100),
            postfilter=PostfilterParams(0.00, True, 6, 4, 3, 3, 1.15, 1.15, 1.10, 0.03),
        ),
        SupportPoint(
            coverage=25.0,
            graph_variant="base_graph",
            graph=GraphParams(k_neighbors=12, similarity_threshold=0.84, max_position_gap=800),
            postfilter=PostfilterParams(0.00, True, 7, 4, 4, 4, 1.00, 1.00, 1.00, 0.00),
        ),
        SupportPoint(
            coverage=28.0,
            graph_variant="base_graph",
            graph=GraphParams(k_neighbors=12, similarity_threshold=0.84, max_position_gap=800),
            postfilter=PostfilterParams(0.00, True, 7, 4, 4, 4, 1.15, 1.15, 1.10, 0.03),
        ),
    ),
    "ont": (
        SupportPoint(
            coverage=5.0,
            graph_variant="merge_b",
            graph=GraphParams(k_neighbors=10, similarity_threshold=0.84, max_position_gap=900),
            postfilter=PostfilterParams(0.945, True, 2, 2, 2, 2, 1.35, 1.35, 1.25, 0.04),
        ),
        SupportPoint(
            coverage=8.0,
            graph_variant="merge_b",
            graph=GraphParams(k_neighbors=10, similarity_threshold=0.84, max_position_gap=900),
            postfilter=PostfilterParams(0.995, True, 3, 2, 3, 3, 1.65, 1.65, 1.45, 0.09),
        ),
        SupportPoint(
            coverage=10.0,
            graph_variant="merge_b",
            graph=GraphParams(k_neighbors=10, similarity_threshold=0.84, max_position_gap=900),
            postfilter=PostfilterParams(0.945, True, 3, 2, 3, 3, 1.65, 1.65, 1.45, 0.09),
        ),
        SupportPoint(
            coverage=12.0,
            graph_variant="merge_b",
            graph=GraphParams(k_neighbors=10, similarity_threshold=0.85, max_position_gap=850),
            postfilter=PostfilterParams(0.895, True, 3, 3, 4, 4, 1.50, 1.50, 1.35, 0.07),
        ),
        SupportPoint(
            coverage=15.0,
            graph_variant="merge_b",
            graph=GraphParams(k_neighbors=10, similarity_threshold=0.84, max_position_gap=850),
            postfilter=PostfilterParams(0.945, True, 3, 3, 4, 4, 1.50, 1.50, 1.35, 0.07),
        ),
        SupportPoint(
            coverage=20.0,
            graph_variant="merge_b",
            graph=GraphParams(k_neighbors=10, similarity_threshold=0.84, max_position_gap=850),
            postfilter=PostfilterParams(0.85, True, 5, 4, 6, 6, 1.50, 1.50, 1.35, 0.07),
        ),
        SupportPoint(
            coverage=30.0,
            graph_variant="merge_b",
            graph=GraphParams(k_neighbors=10, similarity_threshold=0.85, max_position_gap=800),
            postfilter=PostfilterParams(0.85, True, 7, 5, 7, 7, 1.35, 1.35, 1.20, 0.06),
        ),
        SupportPoint(
            coverage=48.0,
            graph_variant="tight_gap",
            graph=GraphParams(k_neighbors=10, similarity_threshold=0.86, max_position_gap=650),
            postfilter=PostfilterParams(0.80, True, 11, 7, 8, 8, 1.35, 1.35, 1.20, 0.06),
        ),
    ),
    "clr": (
        SupportPoint(
            coverage=5.0,
            graph_variant="merge_b",
            graph=GraphParams(k_neighbors=18, similarity_threshold=0.74, max_position_gap=1800),
            postfilter=PostfilterParams(0.81, True, 2, 2, 2, 2, 1.50, 1.50, 1.30, 0.08),
        ),
        SupportPoint(
            coverage=8.0,
            graph_variant="base_graph",
            graph=GraphParams(k_neighbors=12, similarity_threshold=0.82, max_position_gap=1000),
            postfilter=PostfilterParams(0.81, True, 3, 3, 2, 2, 1.75, 1.75, 1.40, 0.10),
        ),
        SupportPoint(
            coverage=12.0,
            graph_variant="tight_graph",
            graph=GraphParams(k_neighbors=10, similarity_threshold=0.86, max_position_gap=800),
            postfilter=PostfilterParams(0.85, False, 2, 2, 1, 1, 1.35, 1.35, 1.20, 0.05),
        ),
        SupportPoint(
            coverage=20.0,
            graph_variant="base_graph",
            graph=GraphParams(k_neighbors=12, similarity_threshold=0.82, max_position_gap=1000),
            postfilter=PostfilterParams(0.00, True, 4, 3, 3, 3, 1.20, 1.20, 1.10, 0.03),
        ),
        SupportPoint(
            coverage=30.0,
            graph_variant="base_graph",
            graph=GraphParams(k_neighbors=12, similarity_threshold=0.82, max_position_gap=1000),
            postfilter=PostfilterParams(0.00, True, 5, 4, 4, 4, 1.20, 1.20, 1.10, 0.05),
        ),
    ),
}


def get_default_unified_scorer_path() -> str:
    return str(default_scorer_path())


def resolve_default_unified_scorer_path() -> str | None:
    path = Path(get_default_unified_scorer_path())
    if path.exists():
        return str(path)
    return None


def _sorted_points(points: Iterable[SupportPoint]) -> List[SupportPoint]:
    return sorted(points, key=lambda item: float(item.coverage))


def _nearest_point(points: Sequence[SupportPoint], coverage: float) -> SupportPoint:
    return min(points, key=lambda item: (abs(float(item.coverage) - float(coverage)), float(item.coverage)))


def _bracket_points(points: Sequence[SupportPoint], coverage: float) -> tuple[SupportPoint, SupportPoint]:
    if coverage <= float(points[0].coverage):
        return points[0], points[0]
    if coverage >= float(points[-1].coverage):
        return points[-1], points[-1]
    for left, right in zip(points[:-1], points[1:]):
        x0 = float(left.coverage)
        x1 = float(right.coverage)
        if x0 <= coverage <= x1:
            return left, right
    return points[-1], points[-1]


def _interp_float(points: Sequence[SupportPoint], coverage: float, getter) -> float:
    left, right = _bracket_points(points, coverage)
    if left is right:
        return float(getter(left))
    x0 = float(left.coverage)
    x1 = float(right.coverage)
    y0 = float(getter(left))
    y1 = float(getter(right))
    alpha = 0.0 if x1 <= x0 else (coverage - x0) / (x1 - x0)
    return float(y0 + alpha * (y1 - y0))


def _interp_int(points: Sequence[SupportPoint], coverage: float, getter, *, minimum: int = 0) -> int:
    return max(minimum, int(round(_interp_float(points, coverage, getter))))


def _interp_bool(points: Sequence[SupportPoint], coverage: float, getter) -> bool:
    return bool(getter(_nearest_point(points, coverage)))


def select_grasv_preset(platform: str, coverage: float | None) -> GraSVPipelinePreset:
    platform_key = str(platform or "").lower()
    if platform_key not in SUPPORT_POINTS:
        raise ValueError(f"Unsupported platform for GraSV preset selection: {platform}")

    points = _sorted_points(SUPPORT_POINTS[platform_key])
    if coverage is None or not math.isfinite(float(coverage)) or float(coverage) <= 0.0:
        coverage = float(points[len(points) // 2].coverage)
    else:
        coverage = float(coverage)

    graph_point = _nearest_point(points, coverage)

    graph = GraphParams(
        k_neighbors=int(graph_point.graph.k_neighbors),
        similarity_threshold=float(graph_point.graph.similarity_threshold),
        max_position_gap=int(graph_point.graph.max_position_gap),
        max_group_size=int(graph_point.graph.max_group_size),
        min_cluster_size=int(graph_point.graph.min_cluster_size),
    )
    postfilter = PostfilterParams(
        cluster_scorer_threshold=_interp_float(points, coverage, lambda item: item.postfilter.cluster_scorer_threshold),
        enable_rule_postfilter=_interp_bool(points, coverage, lambda item: item.postfilter.enable_rule_postfilter),
        min_support_del=_interp_int(points, coverage, lambda item: item.postfilter.min_support_del, minimum=1),
        min_support_ins=_interp_int(points, coverage, lambda item: item.postfilter.min_support_ins, minimum=1),
        min_support_dup=_interp_int(points, coverage, lambda item: item.postfilter.min_support_dup, minimum=1),
        min_support_inv=_interp_int(points, coverage, lambda item: item.postfilter.min_support_inv, minimum=1),
        compactness_start_scale=_interp_float(points, coverage, lambda item: item.postfilter.compactness_start_scale),
        compactness_end_scale=_interp_float(points, coverage, lambda item: item.postfilter.compactness_end_scale),
        compactness_length_cv_scale=_interp_float(points, coverage, lambda item: item.postfilter.compactness_length_cv_scale),
        compactness_cosine_relax=_interp_float(points, coverage, lambda item: item.postfilter.compactness_cosine_relax),
        min_cluster_median_mapq=_interp_float(points, coverage, lambda item: item.postfilter.min_cluster_median_mapq),
    )
    return GraSVPipelinePreset(
        name=f"{platform_key}_cov{coverage:.1f}_graph{graph_point.coverage:.1f}_{graph_point.graph_variant}",
        platform=platform_key,
        requested_coverage=coverage,
        graph_anchor_coverage=float(graph_point.coverage),
        graph_variant=graph_point.graph_variant,
        graph=graph,
        postfilter=postfilter,
    )


GraSVPreset = GraSVPipelinePreset

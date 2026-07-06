from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .signature_features import safe_interval, signature_len_est
from .utils import build_affinity_matrix

from .config import GraphParams
from .signature import guess_svtype_from_signature, normalize_svtype


TYPE_BUCKETS = ("DEL", "INS", "DUP", "INV", "UNK")


def signature_bucket(sig: Any, *, include_tra: bool = False) -> str | None:
    svtype = normalize_svtype(getattr(sig, "svtype", None) or guess_svtype_from_signature(sig))
    if svtype == "TRA":
        return "TRA" if include_tra else None
    if svtype in {"DEL", "INS", "DUP", "INV"}:
        return svtype
    return "UNK"


def group_signature_indices(
    signatures: Sequence[Any],
    *,
    include_tra: bool = False,
) -> Dict[Tuple[str, str], List[int]]:
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, sig in enumerate(signatures):
        bucket = signature_bucket(sig, include_tra=include_tra)
        if bucket is None:
            continue
        contig = str(getattr(sig, "contig", "NA"))
        groups[(contig, bucket)].append(idx)
    return groups


def partition_group_by_position(
    indices: Sequence[int],
    signatures: Sequence[Any],
    max_position_gap: int,
    max_group_size: int,
) -> List[List[int]]:
    if not indices:
        return []

    sorted_indices = sorted(indices, key=lambda idx: int(getattr(signatures[idx], "tstart", 0)))
    partitions: List[List[int]] = []
    current: List[int] = [sorted_indices[0]]
    previous_pos = int(getattr(signatures[sorted_indices[0]], "tstart", 0))

    for idx in sorted_indices[1:]:
        pos = int(getattr(signatures[idx], "tstart", 0))
        if pos - previous_pos > max_position_gap or len(current) >= max_group_size:
            partitions.append(current)
            current = [idx]
        else:
            current.append(idx)
        previous_pos = pos

    if current:
        partitions.append(current)
    return partitions


def _center(sig: Any) -> int:
    s0, e0 = safe_interval(int(getattr(sig, "tstart", 0)), int(getattr(sig, "tend", 0)))
    return (s0 + e0) // 2


def _source_overlap(sig_a: Any, sig_b: Any) -> float:
    tokens_a = set(str(getattr(sig_a, "source", "") or "").split("|"))
    tokens_b = set(str(getattr(sig_b, "source", "") or "").split("|"))
    tokens_a.discard("")
    tokens_b.discard("")
    if not tokens_a or not tokens_b:
        return 1.0
    return 1.10 if tokens_a & tokens_b else 1.0


def _length_similarity(sig_a: Any, sig_b: Any, svtype: str) -> float:
    if svtype == "INS":
        len_a = max(1, abs(int(getattr(sig_a, "svlen", 0) or 0)))
        len_b = max(1, abs(int(getattr(sig_b, "svlen", 0) or 0)))
    else:
        len_a = max(1, signature_len_est(sig_a))
        len_b = max(1, signature_len_est(sig_b))
    ratio = min(len_a, len_b) / max(len_a, len_b)
    return float(max(0.20, math.sqrt(ratio)))


def _position_similarity(sig_a: Any, sig_b: Any, max_position_gap: int) -> float:
    gap = abs(_center(sig_a) - _center(sig_b))
    if gap > max_position_gap:
        return 0.0
    return float(max(0.10, 1.0 - gap / max(1.0, float(max_position_gap))))


def build_group_affinity(
    local_indices: Sequence[int],
    signatures: Sequence[Any],
    embeddings: np.ndarray,
    *,
    svtype: str,
    params: GraphParams,
) -> csr_matrix:
    local_embeddings = embeddings[np.asarray(local_indices, dtype=np.int64)]
    affinity = build_affinity_matrix(local_embeddings, k_neighbors=params.k_neighbors).tocsr(copy=True)
    if affinity.nnz == 0:
        return affinity

    rows, cols = affinity.nonzero()
    weights = affinity.data
    for idx, (row_idx, col_idx) in enumerate(zip(rows.tolist(), cols.tolist())):
        sig_a = signatures[local_indices[row_idx]]
        sig_b = signatures[local_indices[col_idx]]
        pos_sim = _position_similarity(sig_a, sig_b, params.max_position_gap)
        if pos_sim <= 0.0:
            weights[idx] = 0.0
            continue
        len_sim = _length_similarity(sig_a, sig_b, svtype)
        source_sim = _source_overlap(sig_a, sig_b)
        weights[idx] = float(weights[idx] * pos_sim * len_sim * source_sim)

    affinity.eliminate_zeros()
    return affinity.maximum(affinity.T).tocsr()


def connectivity_clusters_from_affinity(
    affinity: csr_matrix,
    *,
    threshold: float,
    min_cluster_size: int,
) -> List[List[int]]:
    if affinity.shape[0] == 0:
        return []
    if affinity.nnz == 0:
        return [[idx] for idx in range(affinity.shape[0]) if min_cluster_size <= 1]

    graph = affinity.tocsr(copy=True)
    keep_mask = graph.data >= float(threshold)
    graph.data = graph.data * keep_mask
    graph.eliminate_zeros()

    n_components, labels = connected_components(graph, directed=False)
    clusters: List[List[int]] = []
    for component_id in range(n_components):
        members = np.where(labels == component_id)[0].tolist()
        if len(members) >= int(min_cluster_size):
            clusters.append(members)
    return clusters

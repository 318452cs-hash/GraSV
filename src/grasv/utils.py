#!/usr/bin/env python3
"""GraSV inference utilities."""

from __future__ import annotations

import random
from typing import Any, List, Sequence

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def build_affinity_matrix(embeddings: np.ndarray, k_neighbors: int = 20) -> csr_matrix:
    """Build a sparse symmetric affinity matrix from kNN cosine similarity."""

    embeddings = np.asarray(embeddings, dtype=np.float32)
    n_samples = embeddings.shape[0]
    if n_samples == 0:
        return csr_matrix((0, 0), dtype=np.float32)
    if n_samples == 1:
        return csr_matrix((1, 1), dtype=np.float32)

    effective_k = max(1, min(int(k_neighbors), n_samples - 1))
    knn = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine")
    knn.fit(embeddings)
    distances, indices = knn.kneighbors(embeddings, return_distance=True)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for row_idx in range(n_samples):
        for distance, col_idx in zip(distances[row_idx, 1:], indices[row_idx, 1:]):
            similarity = max(0.0, 1.0 - float(distance))
            if similarity <= 0.0:
                continue
            rows.append(row_idx)
            cols.append(int(col_idx))
            data.append(similarity)

    affinity = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples), dtype=np.float32)
    return affinity.maximum(affinity.T).tocsr()


def save_vcf(calls: Sequence[Any], output_path: str) -> None:
    """Write site-level structural-variant calls to a VCF file."""

    alt_map = {
        "DEL": "<DEL>",
        "INS": "<INS>",
        "DUP": "<DUP>",
        "INV": "<INV>",
        "TRA": "<TRA>",
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##source=GraSV\n")
        handle.write("##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position\">\n")
        handle.write("##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"SV type\">\n")
        handle.write("##INFO=<ID=SVLEN,Number=1,Type=Integer,Description=\"SV length\">\n")
        handle.write("##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description=\"Number of supporting signatures\">\n")
        handle.write("##INFO=<ID=CHR2,Number=1,Type=String,Description=\"Partner chromosome for TRA\">\n")
        handle.write("##INFO=<ID=CSCORE,Number=1,Type=Float,Description=\"Cluster scorer probability\">\n")
        handle.write("##ALT=<ID=DEL,Description=\"Deletion\">\n")
        handle.write("##ALT=<ID=INS,Description=\"Insertion\">\n")
        handle.write("##ALT=<ID=DUP,Description=\"Duplication\">\n")
        handle.write("##ALT=<ID=INV,Description=\"Inversion\">\n")
        handle.write("##ALT=<ID=TRA,Description=\"Translocation\">\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        for idx, call in enumerate(calls, start=1):
            svtype = str(getattr(call, "svtype", "INS")).upper()
            chrom = str(getattr(call, "contig", "chrNA"))
            pos = max(1, int(getattr(call, "start", 1)))
            end = max(pos, int(getattr(call, "end", pos)))
            support = int(getattr(call, "support", 1))
            svlen = int(getattr(call, "svlen", 0))
            qual = int(round(float(getattr(call, "prob", 0.5)) * 100))
            filt = str(getattr(call, "filter_status", "PASS"))
            info_fields = [
                f"END={end}",
                f"SVTYPE={svtype}",
                f"SVLEN={svlen}",
                f"SUPPORT={support}",
            ]

            partner = getattr(call, "partner_contig", None)
            if partner:
                info_fields.append(f"CHR2={partner}")
            scorer_prob = getattr(call, "scorer_prob", None)
            if scorer_prob is not None:
                info_fields.append(f"CSCORE={float(scorer_prob):.4f}")
            call_id = getattr(call, "call_id", None) or f"call_{idx}"

            handle.write(
                f"{chrom}\t{pos}\t{call_id}\tN\t{alt_map.get(svtype, '<SV>')}\t{qual}\t{filt}\t"
                f"{';'.join(info_fields)}\n"
            )


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and Torch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the preferred Torch device."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = [
    "build_affinity_matrix",
    "get_device",
    "save_vcf",
    "set_seed",
]

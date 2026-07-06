#!/usr/bin/env python3
"""
Signature IO, BAM extraction, and feature matrix construction.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pysam

from .signature_extraction import (
    collect_signatures_multi_chrom,
    collect_signatures_parallel,
    collect_signatures_region,
    save_signatures_to_pickle,
)
from .signature_features import BASE_NODE_FEAT_DIM, NODE_FEAT_DIM, ensure_fields, normalize_platform, signature_to_node_feat
from .signature import Signature, guess_svtype_from_signature, normalize_svtype


NORMALIZED_SIGNATURE_FIELDS_VERSION = 1
LOGGER = logging.getLogger(__name__)
SIGNATURE_REQUIRED_FIELDS = (
    "sid",
    "contig",
    "tstart",
    "tend",
    "svlen",
    "source",
    "strand",
    "mapq",
    "confidence",
    "num_splits",
    "support",
)


def _coerce_signature(obj: Any) -> Any:
    """Convert dictionaries to `Signature` objects when possible."""

    if isinstance(obj, Signature):
        return obj
    if isinstance(obj, dict):
        return Signature.from_dict(obj)
    if hasattr(obj, "_asdict"):
        return Signature.from_dict(obj._asdict())
    if hasattr(obj, "__dict__"):
        legacy_type = getattr(obj, "type", None)
        legacy_svtype = getattr(obj, "svtype", None)
        if legacy_svtype is None and legacy_type is not None:
            legacy_upper = str(legacy_type).upper()
            if "DEL" in legacy_upper or "UNCOVERED" in legacy_upper or "GAP" in legacy_upper:
                legacy_svtype = "DEL"
            elif "INS" in legacy_upper or "SOFT" in legacy_upper or "CLIP" in legacy_upper:
                legacy_svtype = "INS"
            elif "DUP" in legacy_upper:
                legacy_svtype = "DUP"
            elif "INV" in legacy_upper:
                legacy_svtype = "INV"
            elif "TRA" in legacy_upper or "BND" in legacy_upper:
                legacy_svtype = "TRA"
        payload = {
            "sid": getattr(obj, "sid", 0),
            "contig": getattr(obj, "contig", getattr(obj, "chrom", "")),
            "tstart": getattr(obj, "tstart", getattr(obj, "start", getattr(obj, "pos", 0))),
            "tend": getattr(obj, "tend", getattr(obj, "end", getattr(obj, "tstart", 0))),
            "svtype": legacy_svtype or "UNKNOWN",
            "svlen": getattr(obj, "svlen", 0),
            "qname": getattr(obj, "qname", getattr(obj, "read_name", "")),
            "mapq": getattr(obj, "mapq", 0),
            "strand": getattr(obj, "strand", "+"),
            "source": getattr(obj, "source", legacy_type or ""),
            "insert_seq": getattr(obj, "insert_seq", getattr(obj, "ins_seq", None)),
            "num_splits": getattr(obj, "num_splits", 1),
            "sa_contigs": getattr(obj, "sa_contigs", []),
            "confidence": getattr(obj, "confidence", getattr(obj, "prob", 0.5)),
            "support": getattr(obj, "support", 1),
        }
        return Signature.from_dict(payload)
    return obj


def _extract_signatures_payload(payload: Any) -> Sequence[Any]:
    if isinstance(payload, dict):
        return payload.get("signatures", payload.get("sigs", []))
    return payload


def _payload_params(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    params = payload.get("params", {})
    return params if isinstance(params, dict) else {}


def _sample_signatures_are_native(signatures: Sequence[Any], sample_size: int = 64) -> bool:
    sample = list(signatures[:sample_size]) if isinstance(signatures, list) else list(signatures)[:sample_size]
    return bool(sample) and all(isinstance(sig, Signature) for sig in sample)


def _sample_signatures_have_required_fields(signatures: Sequence[Any], sample_size: int = 64) -> bool:
    sample = list(signatures[:sample_size]) if isinstance(signatures, list) else list(signatures)[:sample_size]
    if not sample:
        return True
    for sig in sample:
        for field in SIGNATURE_REQUIRED_FIELDS:
            if getattr(sig, field, None) is None:
                return False
    return True


def _payload_is_normalized(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if int(payload.get("normalized_fields_version", 0) or 0) >= NORMALIZED_SIGNATURE_FIELDS_VERSION:
        return True
    params = _payload_params(payload)
    return int(params.get("normalized_fields_version", 0) or 0) >= NORMALIZED_SIGNATURE_FIELDS_VERSION


def load_signatures(pickle_path: str) -> List[Any]:
    """Load signatures from a pickle payload."""

    load_started = time.time()
    with open(pickle_path, "rb") as handle:
        payload = pickle.load(handle)

    signatures_payload = _extract_signatures_payload(payload)
    signatures = signatures_payload if isinstance(signatures_payload, list) else list(signatures_payload)
    pickle_load_seconds = time.time() - load_started
    LOGGER.info(
        "signature_load pickle_load_done path=%s seconds=%.3f n_signatures=%s",
        pickle_path,
        pickle_load_seconds,
        len(signatures),
    )

    # Fast path: current bundle pickles persist native Signature objects with all
    # required fields already materialized. Skipping the per-record coercion and
    # ensure_fields scan avoids minutes of Python attribute churn on 5M+ records.
    if _sample_signatures_are_native(signatures) and (
        _payload_is_normalized(payload) or _sample_signatures_have_required_fields(signatures)
    ):
        LOGGER.info(
            "signature_load fast_path_native_signatures path=%s seconds=%.3f",
            pickle_path,
            time.time() - load_started,
        )
        return signatures

    signatures = [_coerce_signature(sig) for sig in signatures]
    ensure_fields(signatures)
    LOGGER.info(
        "signature_load normalize_done path=%s seconds=%.3f",
        pickle_path,
        time.time() - load_started,
    )
    return signatures


def save_signatures(signatures: Sequence[Any], output_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Persist signatures to a pickle file with lightweight metadata."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_signatures_to_pickle(list(signatures), output_path, params=metadata or {})


def _parse_requested_contigs(chrom: Optional[str]) -> Optional[set[str]]:
    if chrom is None:
        return None
    contigs = {item.strip() for item in str(chrom).split(",") if item.strip()}
    return contigs or None


def _filter_signatures(
    signatures: Sequence[Any],
    chrom: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> List[Any]:
    requested_contigs = _parse_requested_contigs(chrom)
    region_start = int(start) if start is not None else None
    region_end = int(end) if end is not None else None

    filtered: List[Any] = []
    for sig in signatures:
        contig = str(getattr(sig, "contig", ""))
        if requested_contigs is not None and contig not in requested_contigs:
            continue

        sig_start = int(getattr(sig, "tstart", 0))
        sig_end = int(getattr(sig, "tend", sig_start + 1))
        if region_start is not None and sig_end <= region_start:
            continue
        if region_end is not None and sig_start >= region_end:
            continue
        filtered.append(sig)

    return filtered


def _filter_signatures_by_properties(
    signatures: Sequence[Any],
    min_sv_size: Optional[int] = None,
    max_sv_size: Optional[int] = None,
    min_mapq: Optional[int] = None,
) -> List[Any]:
    filtered: List[Any] = []
    min_size = int(min_sv_size) if min_sv_size is not None else None
    max_size = int(max_sv_size) if max_sv_size is not None else None
    min_quality = int(min_mapq) if min_mapq is not None else None

    for sig in signatures:
        if min_quality is not None and int(getattr(sig, "mapq", 0) or 0) < min_quality:
            continue

        svtype = str(getattr(sig, "svtype", "") or "").upper()
        if svtype == "TRA":
            filtered.append(sig)
            continue

        sig_start = int(getattr(sig, "tstart", 0))
        sig_end = int(getattr(sig, "tend", sig_start + 1))
        sig_len = abs(int(getattr(sig, "svlen", 0) or 0))
        event_size = max(1, sig_len, abs(sig_end - sig_start))

        if min_size is not None and event_size < min_size:
            continue
        if max_size is not None and max_size != -1 and event_size > max_size:
            continue
        filtered.append(sig)

    return filtered


def extract_signatures_from_bam(
    bam_path: str,
    chrom: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    processes: int = 1,
    min_sv_size: int = 30,
    max_sv_size: int = 100000,
    min_mapq: int = 20,
    min_read_len: int = 500,
    min_siglength: int = 30,
    merge_del_threshold: int = 0,
    merge_ins_threshold: int = 100,
    max_split_parts: int = 7,
    region_size: int = 5_000_000,
) -> List[Any]:
    """Extract SV signatures from BAM using the bundled alignment-signal collector."""

    if not os.path.exists(bam_path):
        raise FileNotFoundError(f"BAM file not found: {bam_path}")

    LOGGER.info(
        "extract_signatures_start bam=%s chrom=%s start=%s end=%s processes=%s",
        bam_path,
        chrom,
        start,
        end,
        processes,
    )
    started = time.time()
    if chrom and start is not None and end is not None:
        signatures = collect_signatures_region(
            bam_path=bam_path,
            contig=chrom,
            start=int(start),
            end=int(end),
            min_sv_size=min_sv_size,
            max_sv_size=max_sv_size,
            min_mapq=min_mapq,
            min_read_len=min_read_len,
            min_siglength=min_siglength,
            merge_del_threshold=merge_del_threshold,
            merge_ins_threshold=merge_ins_threshold,
            max_split_parts=max_split_parts,
        )
    else:
        if chrom is None:
            with pysam.AlignmentFile(bam_path, "rb") as bam_file:
                chrom = ",".join(bam_file.references)
        if processes > 1:
            signatures = collect_signatures_parallel(
                bam_path=bam_path,
                chrom_input=chrom,
                num_processes=processes,
                min_sv_size=min_sv_size,
                max_sv_size=max_sv_size,
                min_mapq=min_mapq,
                min_read_len=min_read_len,
                min_siglength=min_siglength,
                merge_del_threshold=merge_del_threshold,
                merge_ins_threshold=merge_ins_threshold,
                max_split_parts=max_split_parts,
                region_size=region_size,
            )
        else:
            signatures = collect_signatures_multi_chrom(
                bam_path=bam_path,
                chrom_input=chrom,
                min_sv_size=min_sv_size,
                max_sv_size=max_sv_size,
                min_mapq=min_mapq,
                min_read_len=min_read_len,
                min_siglength=min_siglength,
                merge_del_threshold=merge_del_threshold,
                merge_ins_threshold=merge_ins_threshold,
                max_split_parts=max_split_parts,
            )

    signatures = _filter_signatures(signatures, chrom=chrom, start=start, end=end)
    ensure_fields(signatures)
    LOGGER.info(
        "extract_signatures_done n_signatures=%s seconds=%.3f",
        len(signatures),
        time.time() - started,
    )
    return list(signatures)


def load_or_extract_signatures(
    data_path: Optional[str] = None,
    bam_path: Optional[str] = None,
    save_path: Optional[str] = None,
    **extract_kwargs: Any,
) -> List[Any]:
    """Load signatures from pickle or extract them from BAM."""

    resolved_data_path = os.path.expanduser(data_path) if data_path else None
    if resolved_data_path and os.path.exists(resolved_data_path):
        LOGGER.info("loading signatures from pickle path=%s", resolved_data_path)
        signatures = load_signatures(data_path)
    elif bam_path:
        LOGGER.info("extracting signatures from BAM path=%s", bam_path)
        signatures = extract_signatures_from_bam(bam_path=bam_path, **extract_kwargs)
        if save_path:
            metadata = {"bam_path": bam_path, **extract_kwargs}
            save_signatures(signatures, save_path, metadata=metadata)
            LOGGER.info("saved extracted signatures path=%s n_signatures=%s", save_path, len(signatures))
    else:
        raise ValueError("Either `data_path` or `bam_path` must be provided.")

    n_before_filter = len(signatures)
    signatures = _filter_signatures(
        signatures,
        chrom=extract_kwargs.get("chrom"),
        start=extract_kwargs.get("start"),
        end=extract_kwargs.get("end"),
    )
    signatures = _filter_signatures_by_properties(
        signatures,
        min_sv_size=extract_kwargs.get("min_sv_size"),
        max_sv_size=extract_kwargs.get("max_sv_size"),
        min_mapq=extract_kwargs.get("min_mapq"),
    )
    ensure_fields(signatures)
    LOGGER.info("signatures_ready n_before_filter=%s n_after_filter=%s", n_before_filter, len(signatures))
    return signatures


def load_source_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Load a JSON or JSONL manifest describing multiple signature sources."""

    resolved_path = os.path.expanduser(manifest_path)
    with open(resolved_path, "r", encoding="utf-8") as handle:
        raw_text = handle.read().strip()

    if not raw_text:
        raise ValueError(f"Manifest is empty: {resolved_path}")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        if "sources" in payload or "records" in payload:
            records = payload.get("sources", payload.get("records", []))
        else:
            records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        records = []
        with open(resolved_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                records.append(json.loads(stripped))

    normalized: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Manifest record #{idx + 1} must be an object, got {type(record).__name__}")

        item = dict(record)
        for key in ("data_path", "bam_path", "feature_bam_path", "save_signatures_path"):
            if item.get(key):
                item[key] = os.path.expanduser(str(item[key]))

        if not item.get("data_path") and not item.get("bam_path"):
            raise ValueError(f"Manifest record #{idx + 1} must include `data_path` or `bam_path`.")
        normalized.append(item)

    if not normalized:
        raise ValueError(f"No usable records found in manifest: {resolved_path}")
    return normalized


def load_feature_matrix_from_manifest(
    manifest_path: str,
    default_extract_kwargs: Optional[Dict[str, Any]] = None,
    default_bin_size: int = 1000,
    feature_dim: int = NODE_FEAT_DIM,
    return_metadata: bool = False,
) -> tuple[np.ndarray, List[Dict[str, Any]]] | tuple[np.ndarray, List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Load multiple sources from a manifest and concatenate their feature matrices."""

    records = load_source_manifest(manifest_path)
    extract_keys = {
        "chrom",
        "start",
        "end",
        "processes",
        "min_sv_size",
        "max_sv_size",
        "min_mapq",
        "min_read_len",
        "min_siglength",
        "merge_del_threshold",
        "merge_ins_threshold",
        "max_split_parts",
        "region_size",
    }

    feature_chunks: List[np.ndarray] = []
    source_stats: List[Dict[str, Any]] = []
    metadata_chunks: Dict[str, List[Any]] = {
        "platforms": [],
        "svtypes": [],
        "contigs": [],
        "starts": [],
        "ends": [],
        "svlens": [],
        "source_ids": [],
    }
    base_extract_kwargs = dict(default_extract_kwargs or {})

    for idx, record in enumerate(records):
        source_name = str(record.get("name") or record.get("sample_id") or f"source_{idx + 1}")
        source_extract_kwargs = dict(base_extract_kwargs)
        record_extract_kwargs = record.get("extract_kwargs", {})
        if isinstance(record_extract_kwargs, dict):
            source_extract_kwargs.update(record_extract_kwargs)
        for key in extract_keys:
            if key in record and record[key] is not None:
                source_extract_kwargs[key] = record[key]

        data_path = record.get("data_path")
        bam_path = record.get("bam_path")
        save_path = record.get("save_signatures_path")
        feature_bam_path = record.get("feature_bam_path") or bam_path
        bin_size = int(record.get("coverage_bin_size", default_bin_size))
        source_platform = normalize_platform(record.get("platform"))

        signatures = load_or_extract_signatures(
            data_path=data_path,
            bam_path=bam_path,
            save_path=save_path,
            **source_extract_kwargs,
        )
        for sig in signatures:
            if source_platform:
                setattr(sig, "platform", source_platform)
        features = build_feature_matrix(
            signatures,
            bam_path=feature_bam_path,
            bin_size=bin_size,
            platform=source_platform,
            feature_dim=feature_dim,
        )
        if len(features) > 0:
            feature_chunks.append(features)
            metadata_chunks["platforms"].extend([(source_platform or "unknown")] * len(signatures))
            metadata_chunks["svtypes"].extend(
                [
                    normalize_svtype(getattr(sig, "svtype", None) or guess_svtype_from_signature(sig)) or "UNK"
                    for sig in signatures
                ]
            )
            metadata_chunks["contigs"].extend([str(getattr(sig, "contig", "")) for sig in signatures])
            metadata_chunks["starts"].extend([int(getattr(sig, "tstart", 0)) for sig in signatures])
            metadata_chunks["ends"].extend([int(getattr(sig, "tend", getattr(sig, "tstart", 0) + 1)) for sig in signatures])
            metadata_chunks["svlens"].extend([int(getattr(sig, "svlen", 0) or 0) for sig in signatures])
            metadata_chunks["source_ids"].extend([idx] * len(signatures))

        source_stats.append(
            {
                "name": source_name,
                "platform": record.get("platform"),
                "coverage": record.get("coverage"),
                "n_signatures": int(len(signatures)),
                "feature_shape": [int(dim) for dim in features.shape],
                "data_path": data_path,
                "bam_path": bam_path,
                "feature_bam_path": feature_bam_path,
            }
        )

    if not feature_chunks:
        raise ValueError(f"No features could be built from manifest: {manifest_path}")
    features = np.concatenate(feature_chunks, axis=0)
    if not return_metadata:
        return features, source_stats

    metadata = {
        "platforms": np.asarray(metadata_chunks["platforms"], dtype=object),
        "svtypes": np.asarray(metadata_chunks["svtypes"], dtype=object),
        "contigs": np.asarray(metadata_chunks["contigs"], dtype=object),
        "starts": np.asarray(metadata_chunks["starts"], dtype=np.int64),
        "ends": np.asarray(metadata_chunks["ends"], dtype=np.int64),
        "svlens": np.asarray(metadata_chunks["svlens"], dtype=np.int64),
        "source_ids": np.asarray(metadata_chunks["source_ids"], dtype=np.int64),
    }
    return features, source_stats, metadata


def build_coverage_bins_from_bam(
    bam_path: str,
    contigs: Iterable[str],
    bin_size: int = 1000,
    regions: Optional[Dict[str, tuple[int, int]]] = None,
) -> Dict[str, tuple[int, np.ndarray]]:
    """
    Build approximate per-bin depth arrays directly from BAM alignments.

    Each primary mapped read contributes one covered interval `[reference_start, reference_end)`.
    """

    contig_names = sorted({str(contig) for contig in contigs if contig})
    if not contig_names:
        return {}

    coverage: Dict[str, tuple[int, np.ndarray]] = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam_file:
        for contig in contig_names:
            try:
                contig_len = bam_file.get_reference_length(contig)
            except (KeyError, ValueError):
                continue

            fetch_start = 0
            fetch_end = contig_len
            if regions and contig in regions:
                region_start, region_end = regions[contig]
                fetch_start = max(0, min(contig_len, int(region_start)))
                fetch_end = max(fetch_start, min(contig_len, int(region_end)))
            n_bins = max(1, (max(fetch_end - fetch_start, 1) + bin_size - 1) // bin_size)
            diff = np.zeros(n_bins + 1, dtype=np.int32)

            for read in bam_file.fetch(contig, fetch_start, fetch_end):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                if read.reference_end is None or read.reference_end <= read.reference_start:
                    continue

                start_pos = max(fetch_start, int(read.reference_start))
                end_pos = min(fetch_end, int(read.reference_end))
                if end_pos <= start_pos:
                    continue

                start_bin = max(0, (start_pos - fetch_start) // bin_size)
                end_bin = min(n_bins - 1, (end_pos - fetch_start - 1) // bin_size)
                diff[start_bin] += 1
                diff[end_bin + 1] -= 1

            coverage[contig] = (fetch_start, np.cumsum(diff[:-1]).astype(np.int32))

    return coverage


def build_feature_matrix(
    signatures: Sequence[Any],
    bam_path: Optional[str] = None,
    bin_size: int = 1000,
    platform: Optional[str] = None,
    feature_dim: int = NODE_FEAT_DIM,
    coverage_cache_path: Optional[str] = None,
) -> np.ndarray:
    """Convert signatures to a `(N, D)` float32 feature matrix."""

    signatures = list(signatures)
    ensure_fields(signatures)
    if feature_dim not in {BASE_NODE_FEAT_DIM, NODE_FEAT_DIM}:
        raise ValueError(f"Unsupported feature_dim={feature_dim}; expected {BASE_NODE_FEAT_DIM} or {NODE_FEAT_DIM}.")
    source_platform = normalize_platform(platform)
    for sig in signatures:
        if source_platform and not normalize_platform(getattr(sig, "platform", None)):
            setattr(sig, "platform", source_platform)
    regions: Dict[str, tuple[int, int]] = {}
    pad = max(1, 3 * int(bin_size))
    for sig in signatures:
        contig = str(getattr(sig, "contig", ""))
        start = int(getattr(sig, "tstart", 0))
        end = int(getattr(sig, "tend", start + 1))
        left = max(0, min(start, end) - pad)
        right = max(left + 1, max(start, end) + pad)
        if contig not in regions:
            regions[contig] = (left, right)
        else:
            prev_left, prev_right = regions[contig]
            regions[contig] = (min(prev_left, left), max(prev_right, right))
    cov_bins_by_contig: Dict[str, tuple[int, np.ndarray]] = {}
    resolved_cache_path = os.path.expanduser(coverage_cache_path) if coverage_cache_path else None
    if bam_path:
        loaded_from_cache = False
        if resolved_cache_path and os.path.exists(resolved_cache_path):
            try:
                with open(resolved_cache_path, "rb") as handle:
                    cached_payload = pickle.load(handle)
                cached_meta = cached_payload.get("meta", {}) if isinstance(cached_payload, dict) else {}
                cached_cov = cached_payload.get("coverage", {}) if isinstance(cached_payload, dict) else {}
                if (
                    cached_meta.get("bam_path") == os.path.expanduser(bam_path)
                    and int(cached_meta.get("bin_size", -1)) == int(bin_size)
                    and cached_meta.get("regions") == {key: [int(v[0]), int(v[1])] for key, v in regions.items()}
                ):
                    cov_bins_by_contig = cached_cov
                    loaded_from_cache = True
            except Exception:
                loaded_from_cache = False
        if not loaded_from_cache:
            cov_bins_by_contig = build_coverage_bins_from_bam(
                bam_path,
                contigs={getattr(sig, "contig", "") for sig in signatures},
                bin_size=bin_size,
                regions=regions,
            )
            if resolved_cache_path:
                os.makedirs(os.path.dirname(resolved_cache_path) or ".", exist_ok=True)
                with open(resolved_cache_path, "wb") as handle:
                    pickle.dump(
                        {
                            "meta": {
                                "bam_path": os.path.expanduser(bam_path),
                                "bin_size": int(bin_size),
                                "regions": {key: [int(v[0]), int(v[1])] for key, v in regions.items()},
                            },
                            "coverage": cov_bins_by_contig,
                        },
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

    feats = np.zeros((len(signatures), feature_dim), dtype=np.float32)
    for idx, sig in enumerate(signatures):
        contig = str(getattr(sig, "contig", ""))
        feats[idx] = signature_to_node_feat(
            sig,
            cov_bins=cov_bins_by_contig.get(contig),
            bin_size=bin_size,
            platform=source_platform,
            feature_dim=feature_dim,
        )
    return feats


__all__ = [
    "NODE_FEAT_DIM",
    "Signature",
    "build_feature_matrix",
    "build_coverage_bins_from_bam",
    "extract_signatures_from_bam",
    "load_feature_matrix_from_manifest",
    "load_or_extract_signatures",
    "load_source_manifest",
    "load_signatures",
    "save_signatures",
]

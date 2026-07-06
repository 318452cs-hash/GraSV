# Signature feature utilities.
"""
Utilities for signature post-processing + feature engineering (Node feats) + BED filtering
+ coverage bin (depth/VAF) support.

Current mainline feature spec:
- `BASE_NODE_FEAT_DIM = 24` for legacy compatibility
- `NODE_FEAT_DIM = 27` for the current primary pipeline

The active training/inference path uses 27-dimensional node features produced by:
- collect_signature_v2.py (INV strand pattern ++/--)
- caller_collect_signatures_v5.py (fine-grained source tags + read_infos coverage)

Key exports used by your pipeline:
- ensure_fields(signatures)
- signature_center(sig)
- signature_len_est(sig)
- sig_support(sig)
- load_bed_intervals(bed_path)
- filter_signatures_by_bed(signatures, bed_path, contig=None)
- build_coverage_bins(read_infos, chrom_len, chrom=None, bin_size=1000)
- local_depth_from_bins(cov_bins, pos0, bin_size=1000, win_bins=2)
- signature_to_node_feat(sig, cov_bins=None, bin_size=1000)
"""

from __future__ import annotations

import bisect
import gzip
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .signature import guess_svtype_from_signature, normalize_svtype, safe_interval


# -------------------------
# Feature spec
# -------------------------
BASE_NODE_FEAT_DIM = 24
PLATFORM_FEAT_DIM = 3
NODE_FEAT_VERSION = "v3_27d"

NODE_FEAT_NAMES: List[str] = [
    # 0-4: SVTYPE one-hot
    "is_DEL", "is_INS", "is_DUP", "is_INV", "is_TRA",
    # 5-9: core numeric
    "log1p_span", "log1p_svlen", "log1p_support", "mapq_norm", "confidence",
    # 10-12: coarse source (backward compatible)
    "src_cigar", "src_split", "src_clip",
    # 13: inserted sequence exists (if any)
    "has_ins_seq",
    # 14-15: INV strand patterns
    "inv_pp", "inv_mm",
    # 16-21: fine-grained source one-hot (new)
    "src_CIGAR", "src_SA_TAG", "src_SA_TAG_MULTI", "src_SA_TAG_MULTI_INS", "src_CLIP_LEFT", "src_CLIP_RIGHT",
    # 22-23: coverage / VAF
    "log1p_local_depth", "vaf_support_over_depth",
    # 24-26: platform one-hot
    "platform_ont", "platform_ccs", "platform_clr",
]
NODE_FEAT_DIM = BASE_NODE_FEAT_DIM + PLATFORM_FEAT_DIM


# -------------------------
# Signature field helpers
# -------------------------
def ensure_fields(signatures: Sequence[Any]) -> None:
    """
    In-place: ensure every signature has minimal required attributes.

    We DO NOT assume a strict signature class. We only set missing fields
    that downstream code expects (sid, contig, tstart, tend, svlen, source, strand, mapq, confidence, num_splits).
    """
    for i, s in enumerate(signatures):
        if getattr(s, "sid", None) is None:
            setattr(s, "sid", i)
        # common coordinates
        if getattr(s, "contig", None) is None:
            setattr(s, "contig", getattr(s, "chrom", getattr(s, "CHROM", "NA")))
        if getattr(s, "tstart", None) is None:
            # sometimes named pos/start
            setattr(s, "tstart", int(getattr(s, "start", getattr(s, "pos", 0))))
        if getattr(s, "tend", None) is None:
            # allow INS point event => tend = tstart
            setattr(s, "tend", int(getattr(s, "end", getattr(s, "tstart", 0))))
        # svlen/source/strand
        if getattr(s, "svlen", None) is None:
            setattr(s, "svlen", int(getattr(s, "SVLEN", 0) or 0))
        if getattr(s, "source", None) is None:
            setattr(s, "source", str(getattr(s, "SRC", "")) if getattr(s, "SRC", None) is not None else "")
        if getattr(s, "strand", None) is None:
            setattr(s, "strand", str(getattr(s, "STRAND", "")) if getattr(s, "STRAND", None) is not None else "")
        if getattr(s, "mapq", None) is None:
            setattr(s, "mapq", float(getattr(s, "MAPQ", 0.0) or 0.0))
        if getattr(s, "confidence", None) is None:
            # some scripts use PROB
            conf = getattr(s, "prob", None)
            if conf is None:
                conf = getattr(s, "PROB", None)
            if conf is None:
                conf = getattr(s, "score", None)
            if conf is None:
                conf = 0.0
            setattr(s, "confidence", float(conf))
        if getattr(s, "num_splits", None) is None:
            setattr(s, "num_splits", int(getattr(s, "NUM_SPLITS", 0) or 0))
        if getattr(s, "support", None) is None:
            # your code uses INFO/SUPPORT in VCF; signature may keep support
            setattr(s, "support", int(getattr(s, "SUPPORT", 0) or 0))


def signature_center(sig: Any) -> int:
    s0, e0 = safe_interval(int(getattr(sig, "tstart", 0)), int(getattr(sig, "tend", 0)))
    # INS can have zero span; keep as point
    return (s0 + e0) // 2


def signature_len_est(sig: Any) -> int:
    svlen = getattr(sig, "svlen", 0)
    try:
        svlen = abs(int(svlen))
    except Exception:
        svlen = 0
    if svlen > 0:
        return svlen
    s0, e0 = safe_interval(int(getattr(sig, "tstart", 0)), int(getattr(sig, "tend", 0)))
    return max(1, e0 - s0)


def sig_support(sig: Any) -> int:
    """
    Support definition:
    - prefer explicit sig.support if set
    - incorporate num_splits (multi-split -> higher evidence)
    - incorporate reads list length if available
    """
    sup = 0
    for k in ("support", "SUPPORT", "sup", "count"):
        v = getattr(sig, k, None)
        if v is None:
            continue
        try:
            sup = max(sup, int(v))
        except Exception:
            pass

    ns = getattr(sig, "num_splits", 0)
    try:
        ns = int(ns)
    except Exception:
        ns = 0
    if ns > 0:
        sup = max(sup, ns)

    reads = getattr(sig, "reads", None)
    if reads is not None:
        try:
            sup = max(sup, len(reads))
        except Exception:
            pass

    return max(1, sup)


# -------------------------
# BED loading + filtering
# -------------------------
def _bed_alt_contig_names(name: str) -> List[str]:
    """Return alternative contig names (chr1 <-> 1)."""
    name = str(name)
    alts = {name}
    if name.startswith("chr") and len(name) > 3:
        alts.add(name[3:])
    else:
        alts.add("chr" + name)
    return list(alts)


def load_bed_intervals(bed_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Load BED (optionally .gz) -> dict(contig -> merged intervals).
    BED is 0-based, half-open [start, end).
    """
    intervals: Dict[str, List[Tuple[int, int]]] = {}
    opener = gzip.open if bed_path.endswith(".gz") else open
    with opener(bed_path, "rt") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip().split()
            if len(parts) < 3:
                continue
            chrom, s, e = parts[0], parts[1], parts[2]
            try:
                s = int(s)
                e = int(e)
            except Exception:
                continue
            if e <= s:
                continue
            for c in _bed_alt_contig_names(chrom):
                intervals.setdefault(c, []).append((s, e))

    # sort + merge
    for c, xs in list(intervals.items()):
        xs.sort()
        merged: List[Tuple[int, int]] = []
        for s, e in xs:
            if (not merged) or (s > merged[-1][1]):
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        intervals[c] = merged
    return intervals


def interval_overlaps_bed(contig: str, start: int, end: int, bed: Optional[Dict[str, List[Tuple[int, int]]]]) -> bool:
    """
    True if [start,end) overlaps any interval in bed[contig].
    If bed is None -> True.
    """
    if bed is None:
        return True
    xs = bed.get(str(contig))
    if not xs:
        return False

    # find rightmost interval with start < end
    i = bisect.bisect_left(xs, (end, -1)) - 1
    if i >= 0 and xs[i][1] > start:
        return True
    j = i + 1
    if 0 <= j < len(xs) and xs[j][0] < end and xs[j][1] > start:
        return True
    return False


def filter_signatures_by_bed(
    signatures: Sequence[Any],
    bed_path: Optional[str],
    contig: Optional[str] = None,
) -> List[Any]:
    """
    Keep signatures whose interval overlaps a BED interval set.
    If bed_path is None -> return signatures as list.
    If contig is provided, will check bed for that contig, but still uses signature.contig by default.
    """
    if bed_path is None:
        return list(signatures)

    bed = load_bed_intervals(bed_path)
    out: List[Any] = []
    for s in signatures:
        c = str(contig) if contig is not None else str(getattr(s, "contig", ""))
        if contig is None:
            c = str(getattr(s, "contig", ""))
        s0, e0 = safe_interval(int(getattr(s, "tstart", 0)), int(getattr(s, "tend", 0)))
        if e0 <= s0:
            e0 = s0 + 1
        if interval_overlaps_bed(c, s0, e0, bed):
            out.append(s)
    logging.info(f"[bed] filtered signatures: {len(signatures)} -> {len(out)} using {bed_path}")
    return out


# -------------------------
# Coverage bins (depth)
# -------------------------
def _iter_intervals_from_read_infos(read_infos: Any, chrom: Optional[str] = None) -> Iterable[Tuple[str, int, int]]:
    """
    Robust iterator over (contig, start0, end0) intervals from read_infos.

    Supported formats (best-effort):
    - dict: {contig: [(s,e), ...]} or {contig: [{"s":..,"e":..}, ...]}
    - list/tuple of:
        * (contig, s, e)
        * {"contig"/"chrom":..., "start":..., "end":...}
        * {"rname":..., "s":..., "e":...}
    """
    if read_infos is None:
        return

    # dict[contig] -> intervals
    if isinstance(read_infos, dict):
        for c, xs in read_infos.items():
            if chrom is not None and str(c) != str(chrom):
                continue
            if xs is None:
                continue
            for it in xs:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    s, e = int(it[0]), int(it[1])
                    yield str(c), s, e
                elif isinstance(it, dict):
                    s = it.get("start", it.get("s", None))
                    e = it.get("end", it.get("e", None))
                    if s is None or e is None:
                        continue
                    yield str(c), int(s), int(e)
        return

    # list of interval objects
    if isinstance(read_infos, (list, tuple)):
        for it in read_infos:
            if isinstance(it, (list, tuple)):
                if len(it) == 3:
                    c, s, e = it[0], it[1], it[2]
                    if chrom is not None and str(c) != str(chrom):
                        continue
                    yield str(c), int(s), int(e)
                elif len(it) == 2:
                    # no contig => assume provided chrom
                    if chrom is None:
                        continue
                    s, e = it[0], it[1]
                    yield str(chrom), int(s), int(e)
            elif isinstance(it, dict):
                c = it.get("contig", it.get("chrom", it.get("rname", chrom)))
                if c is None:
                    continue
                if chrom is not None and str(c) != str(chrom):
                    continue
                s = it.get("start", it.get("s", None))
                e = it.get("end", it.get("e", None))
                if s is None or e is None:
                    continue
                yield str(c), int(s), int(e)
        return


def build_coverage_bins(
    read_infos: Any,
    chrom_len: int,
    chrom: Optional[str] = None,
    bin_size: int = 1000,
) -> np.ndarray:
    """
    Build per-bin depth array for one chromosome.

    Returns:
        cov_bins: np.int32 array of length nbins, where cov_bins[i] is the depth
        (number of read coverage intervals covering that bin).
    """
    chrom_len = int(chrom_len)
    if chrom_len <= 0:
        return np.zeros((0,), dtype=np.int32)
    bin_size = int(bin_size)
    nbins = (chrom_len + bin_size - 1) // bin_size
    diff = np.zeros((nbins + 1,), dtype=np.int32)

    n_intervals = 0
    for c, s, e in _iter_intervals_from_read_infos(read_infos, chrom=chrom):
        # clamp
        if e <= s:
            continue
        s = max(0, int(s))
        e = min(chrom_len, int(e))
        if e <= s:
            continue
        bs = s // bin_size
        be = (e - 1) // bin_size
        diff[bs] += 1
        diff[be + 1] -= 1
        n_intervals += 1

    cov = np.cumsum(diff[:-1]).astype(np.int32)
    logging.info(f"[cov] built coverage bins chrom={chrom} len={chrom_len:,} bin={bin_size} nbins={len(cov)} intervals={n_intervals}")
    return cov


def local_depth_from_bins(
    cov_bins: Optional[np.ndarray],
    pos0: int,
    bin_size: int = 1000,
    win_bins: int = 2,
) -> float:
    """
    Average depth around pos0 over +/- win_bins bins.
    If cov_bins is None or empty -> return 0.
    """
    if cov_bins is None:
        return 0.0
    offset = 0
    if isinstance(cov_bins, tuple):
        if len(cov_bins) != 2:
            return 0.0
        offset, cov_bins = cov_bins
    if len(cov_bins) == 0:
        return 0.0
    pos0 = int(pos0)
    bin_size = int(bin_size)
    offset = int(offset)
    i = (pos0 - offset) // bin_size
    lo = max(0, i - int(win_bins))
    hi = min(len(cov_bins), i + int(win_bins) + 1)
    if hi <= lo:
        return float(cov_bins[min(max(i, 0), len(cov_bins) - 1)])
    return float(np.mean(cov_bins[lo:hi]))


# -------------------------
# Source tag parsing
# -------------------------
def _split_source_tokens(src: Any) -> List[str]:
    """
    Split a 'source' field into tokens.
    Accepts string/list/tuple; returns upper-cased tokens.
    """
    if src is None:
        return []
    if isinstance(src, (list, tuple)):
        toks = []
        for x in src:
            if x is None:
                continue
            toks.append(str(x))
        src_s = ",".join(toks)
    else:
        src_s = str(src)

    # common separators
    for sep in ["|", ";", ",", " "]:
        src_s = src_s.replace(sep, "\t")
    toks = [t.strip().upper() for t in src_s.split("\t") if t.strip()]
    return toks


def _has_token(tokens: List[str], key: str) -> bool:
    key = key.upper()
    if key in tokens:
        return True
    # fallback to substring (for backward compatibility like "SA" in "SA_TAG")
    for t in tokens:
        if key in t:
            return True
    return False


def normalize_platform(platform: Optional[str]) -> Optional[str]:
    if platform is None:
        return None
    token = str(platform).strip().lower()
    if not token:
        return None
    if token in {"ont", "nanopore", "oxfordnanopore"}:
        return "ont"
    if token in {"ccs", "hifi", "pacbio_ccs", "pacbio-hifi"}:
        return "ccs"
    if token in {"clr", "pacbio_clr", "pacbio"}:
        return "clr"
    return token


# -------------------------
# Feature engineering
# -------------------------
def signature_to_node_feat(
    sig: Any,
    cov_bins: Optional[np.ndarray] = None,
    bin_size: int = 1000,
    platform: Optional[str] = None,
    feature_dim: int = NODE_FEAT_DIM,
) -> np.ndarray:
    """
    Convert signature -> node feature vector (float32).

    IMPORTANT:
    - This function is the ONLY place you should change when adding/removing node features.
    - Train and infer MUST use the same implementation.
    """
    svt = normalize_svtype(guess_svtype_from_signature(sig))
    if svt is None:
        svt = "UNK"

    # 0-4: type onehot
    type_map = {"DEL": 0, "INS": 1, "DUP": 2, "INV": 3, "TRA": 4}
    if feature_dim not in {BASE_NODE_FEAT_DIM, NODE_FEAT_DIM}:
        raise ValueError(f"Unsupported feature_dim={feature_dim}; expected {BASE_NODE_FEAT_DIM} or {NODE_FEAT_DIM}.")

    feat = np.zeros((feature_dim,), dtype=np.float32)
    if svt in type_map:
        feat[type_map[svt]] = 1.0

    # coords
    s0, e0 = safe_interval(int(getattr(sig, "tstart", 0)), int(getattr(sig, "tend", 0)))
    if e0 <= s0:
        e0 = s0 + 1
    center = (s0 + e0) // 2
    span = max(1, e0 - s0)

    # svlen/support
    svlen = signature_len_est(sig)
    support = sig_support(sig)

    # mapq/confidence
    try:
        mapq = float(getattr(sig, "mapq", 0.0) or 0.0)
    except Exception:
        mapq = 0.0
    mapq_norm = max(0.0, min(1.0, mapq / 60.0))

    try:
        conf = float(getattr(sig, "confidence", 0.0) or 0.0)
    except Exception:
        conf = 0.0
    # conf sometimes is probability; clip to [0,1]
    conf = max(0.0, min(1.0, conf))

    # 5-9 core numeric
    feat[5] = np.log1p(float(span))
    feat[6] = np.log1p(float(max(1, svlen)))
    feat[7] = np.log1p(float(max(1, support)))
    feat[8] = float(mapq_norm)
    feat[9] = float(conf)

    # source tokens
    tokens = _split_source_tokens(getattr(sig, "source", ""))

    # 10-12 coarse source
    # - cigar: "CIGAR" or "GAP" etc.
    src_cigar = 1.0 if (_has_token(tokens, "CIGAR") or _has_token(tokens, "GAP")) else 0.0
    src_split = 1.0 if (_has_token(tokens, "SA") or _has_token(tokens, "SPLIT")) else 0.0
    src_clip = 1.0 if (_has_token(tokens, "CLIP")) else 0.0
    feat[10] = src_cigar
    feat[11] = src_split
    feat[12] = src_clip

    # 13: inserted sequence exists
    has_seq = 0.0
    for k in ("ins_seq", "insert_seq", "seq", "INSSEQ"):
        v = getattr(sig, k, None)
        if v is None:
            continue
        try:
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8", errors="ignore")
            if isinstance(v, str) and len(v) > 0:
                has_seq = 1.0
                break
        except Exception:
            pass
    feat[13] = has_seq

    # 14-15: INV strand patterns ++ / --
    strand = str(getattr(sig, "strand", "") or "").strip()
    feat[14] = 1.0 if strand == "++" else 0.0  # inv_pp
    feat[15] = 1.0 if strand == "--" else 0.0  # inv_mm

    # 16-21: refined source tags
    feat[16] = 1.0 if _has_token(tokens, "CIGAR") else 0.0
    feat[17] = 1.0 if _has_token(tokens, "SA_TAG") else 0.0
    feat[18] = 1.0 if _has_token(tokens, "SA_TAG_MULTI") else 0.0
    feat[19] = 1.0 if _has_token(tokens, "SA_TAG_MULTI_INS") else 0.0
    feat[20] = 1.0 if (_has_token(tokens, "CLIP_LEFT") or _has_token(tokens, "CLIP_L")) else 0.0
    feat[21] = 1.0 if (_has_token(tokens, "CLIP_RIGHT") or _has_token(tokens, "CLIP_R")) else 0.0

    # 22-23: depth + vaf
    depth = local_depth_from_bins(cov_bins, center, bin_size=bin_size, win_bins=2)
    feat[22] = np.log1p(float(max(0.0, depth)))
    if depth <= 0.0:
        vaf = 0.0
    else:
        vaf = float(support) / float(depth)
        # clip to reasonable range; extreme values are mostly noise
        vaf = max(0.0, min(2.0, vaf))
    feat[23] = vaf

    if feature_dim == NODE_FEAT_DIM:
        platform_key = normalize_platform(platform) or normalize_platform(getattr(sig, "platform", None))
        platform_map = {"ont": 24, "ccs": 25, "clr": 26}
        if platform_key in platform_map:
            feat[platform_map[platform_key]] = 1.0

    return feat


# -------------------------
# Convenience: batch features
# -------------------------
def signatures_to_node_feats(
    signatures: Sequence[Any],
    cov_bins: Optional[np.ndarray] = None,
    bin_size: int = 1000,
    platform: Optional[str] = None,
    feature_dim: int = NODE_FEAT_DIM,
) -> np.ndarray:
    """
    Batch conversion -> (N, D) float32.
    """
    feats = np.zeros((len(signatures), feature_dim), dtype=np.float32)
    for i, s in enumerate(signatures):
        feats[i] = signature_to_node_feat(s, cov_bins=cov_bins, bin_size=bin_size, platform=platform, feature_dim=feature_dim)
    return feats


CLUSTER_SCORER_FEATURE_NAMES: Tuple[str, ...] = (
    "log_support",
    "log_cluster_size",
    "support_ratio",
    "log_svlen",
    "log_span",
    "start_std",
    "end_std",
    "start_std_norm",
    "end_std_norm",
    "length_cv",
    "mean_cosine",
    "median_mapq",
    "prob_base",
    "split_fraction",
    "cigar_fraction",
    "source_diversity",
    "global_coverage",
    "log_global_coverage",
    "support_over_global_coverage",
    "coverage_low",
    "coverage_mid",
    "coverage_high",
    "platform_ont",
    "platform_ccs",
    "platform_clr",
    "strict_support_threshold",
    "support_margin_to_strict",
    "below_strict_support",
    "failed_compactness",
    "compactness_fail_count",
    "failed_mapq",
    "failed_start_std",
    "failed_end_std",
    "failed_length_cv",
    "failed_cosine",
    "is_del",
    "is_ins",
    "is_dup",
    "is_inv",
    "is_tra",
)


def assign_call_ids(calls: Sequence[Any]) -> None:
    for idx, call in enumerate(calls, start=1):
        if not getattr(call, "call_id", None):
            setattr(call, "call_id", f"call_{idx}")


def vectorize_call_features(
    feature_dict: Dict[str, Any],
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    names = tuple(feature_names or CLUSTER_SCORER_FEATURE_NAMES)
    return np.asarray([float(feature_dict.get(name, 0.0) or 0.0) for name in names], dtype=np.float32)


def save_call_features(
    calls: Sequence[Any],
    output_path: str,
    platform: str | None = None,
    global_coverage: float | None = None,
    domain: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for call in calls:
            payload = {
                "call_id": getattr(call, "call_id", None),
                "platform": platform,
                "domain": domain,
                "global_coverage": None if global_coverage is None else float(global_coverage),
                "contig": getattr(call, "contig", None),
                "start": int(getattr(call, "start", 0) or 0),
                "end": int(getattr(call, "end", 0) or 0),
                "svtype": getattr(call, "svtype", None),
                "svlen": int(getattr(call, "svlen", 0) or 0),
                "support": int(getattr(call, "support", 0) or 0),
                "prob": float(getattr(call, "prob", 0.0) or 0.0),
                "score_features": dict(getattr(call, "score_features", {}) or {}),
            }
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def load_call_features(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# =========================
# v4 compatibility shims
# =========================
def group_signatures_by_type(signatures):
    """
    Group signatures by normalized SVTYPE.
    Returns: dict(svtype -> list[signature])
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for s in signatures:
        svt = normalize_svtype(getattr(s, "svtype", None) or guess_svtype_from_signature(s))
        buckets[svt].append(s)
    return dict(buckets)


def dedup_signatures(signatures, round_bp=10):
    """
    Deduplicate signatures with coarse rounding.

    Key idea: signatures are considered duplicates if they share:
      (contig, svtype, rounded_center, rounded_span, inv_strand_pattern)
    Keep the best one by (support, mapq, confidence).
    """
    if not signatures:
        return []

    # make sure required fields exist
    ensure_fields(signatures)

    best = {}  # key -> (score_tuple, sig)
    rb = max(1, int(round_bp))

    for s in signatures:
        contig = str(getattr(s, "contig", ""))
        svt = normalize_svtype(getattr(s, "svtype", None) or guess_svtype_from_signature(s))

        cen = signature_center(s)
        span = signature_len_est(s)

        strand = str(getattr(s, "strand", "") or "")
        # only preserve INV strand patterns if they look like "++" or "--"
        strand_key = strand if (svt == "INV" and strand in ("++", "--")) else ""

        key = (contig, svt, int(cen // rb), int(span // rb), strand_key)

        score = (
            sig_support(s),
            float(getattr(s, "mapq", 0.0) or 0.0),
            float(getattr(s, "confidence", 0.0) or 0.0),
        )

        if (key not in best) or (score > best[key][0]):
            best[key] = (score, s)

    return [v[1] for v in best.values()]

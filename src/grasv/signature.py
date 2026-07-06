#!/usr/bin/env python3
"""
Core signature data structures used across extraction, training, and inference.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class SignatureType(enum.Enum):
    """Supported structural-variant signature types."""

    DEL = "DEL"
    INS = "INS"
    DUP = "DUP"
    INV = "INV"
    TRA = "TRA"
    UNKNOWN = "UNKNOWN"


SUPPORTED_SVTYPES = ("DEL", "INS", "DUP", "INV", "TRA")


def normalize_svtype(value: Optional[str]) -> Optional[str]:
    """Normalize SVTYPE strings to the canonical project set."""

    if value is None:
        return None

    svtype = str(value).upper().strip()
    if svtype in SUPPORTED_SVTYPES:
        return svtype
    if svtype in {"BND", "CTX", "TRANSLOCATION"}:
        return "TRA"

    aliases = {
        "DELETION": "DEL",
        "INSERTION": "INS",
        "DUPLICATION": "DUP",
        "TANDEM": "DUP",
        "INVERSION": "INV",
        "TRA": "TRA",
    }
    if svtype in aliases:
        return aliases[svtype]

    for prefix in SUPPORTED_SVTYPES:
        if svtype.startswith(prefix + ":") or svtype.startswith(prefix + "_"):
            return prefix
        if svtype.endswith(":" + prefix) or svtype.endswith("_" + prefix):
            return prefix

    return None


def safe_interval(start: int, end: int) -> tuple[int, int]:
    """Return a non-negative half-open interval with minimum span 1."""

    left = int(start or 0)
    right = int(end or 0)
    if right < left:
        left, right = right, left
    if right <= left:
        right = left + 1
    if left < 0:
        shift = -left
        left = 0
        right += shift
    return left, right


def guess_svtype_from_signature(sig: object) -> str:
    """Infer SVTYPE from a signature object when it is missing."""

    explicit = normalize_svtype(getattr(sig, "svtype", None))
    if explicit:
        return explicit

    legacy = normalize_svtype(getattr(sig, "type", None))
    if legacy:
        return legacy

    source = str(getattr(sig, "source", "") or "").upper()
    if "TRA" in source or "BND" in source:
        return "TRA"
    if "INV" in source:
        return "INV"
    if "DUP" in source:
        return "DUP"
    if "DEL" in source:
        return "DEL"
    if "INS" in source or "CLIP" in source:
        return "INS"

    svlen = abs(int(getattr(sig, "svlen", 0) or 0))
    start = int(getattr(sig, "tstart", getattr(sig, "start", 0)) or 0)
    end = int(getattr(sig, "tend", getattr(sig, "end", start)) or start)
    left, right = safe_interval(start, end)
    span = right - left

    if svlen > 0 and span <= 5:
        return "INS"
    if svlen == 0 and getattr(sig, "sa_contigs", None):
        return "TRA"
    if span <= 5:
        return "INS"
    return "DEL"


@dataclass
class Signature:
    """
    Single-read SV evidence.

    Coordinates are stored as 0-based, half-open intervals [tstart, tend).
    """

    sid: int = 0
    contig: str = ""
    tstart: int = 0
    tend: int = 0
    svtype: str = SignatureType.UNKNOWN.value
    svlen: int = 0
    qname: str = ""
    mapq: int = 0
    strand: str = "+"
    source: str = "UNKNOWN"

    insert_seq: Optional[str] = None
    num_splits: int = 1
    sa_contigs: List[str] = field(default_factory=list)

    sorted_aligns: List[Dict[str, Any]] = field(default_factory=list)
    bkps: List[List[int]] = field(default_factory=list)

    svm_score: float = 0.0
    confidence: float = 0.5
    support: int = 1

    def __post_init__(self) -> None:
        self.tstart = int(self.tstart)
        self.tend = int(self.tend)
        if self.tend < self.tstart:
            self.tstart, self.tend = self.tend, self.tstart
        if self.tend <= self.tstart:
            self.tend = self.tstart + 1
        self.svlen = int(self.svlen or 0)
        self.mapq = int(self.mapq or 0)
        self.num_splits = int(self.num_splits or 0)
        self.support = int(self.support or 1)

    @property
    def span(self) -> int:
        return max(1, self.tend - self.tstart)

    @property
    def center(self) -> int:
        return (self.tstart + self.tend) // 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sid": self.sid,
            "contig": self.contig,
            "tstart": self.tstart,
            "tend": self.tend,
            "svtype": self.svtype,
            "svlen": self.svlen,
            "qname": self.qname,
            "mapq": self.mapq,
            "strand": self.strand,
            "source": self.source,
            "insert_seq": self.insert_seq,
            "num_splits": self.num_splits,
            "sa_contigs": list(self.sa_contigs),
            "sorted_aligns": list(self.sorted_aligns),
            "bkps": list(self.bkps),
            "svm_score": self.svm_score,
            "confidence": self.confidence,
            "support": self.support,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Signature":
        return cls(
            sid=payload.get("sid", 0),
            contig=payload.get("contig", payload.get("chrom", "")),
            tstart=payload.get("tstart", payload.get("start", payload.get("pos", 0))),
            tend=payload.get("tend", payload.get("end", payload.get("tstart", 0))),
            svtype=payload.get("svtype", payload.get("type", SignatureType.UNKNOWN.value)),
            svlen=payload.get("svlen", payload.get("SVLEN", 0)),
            qname=payload.get("qname", payload.get("read_name", "")),
            mapq=payload.get("mapq", payload.get("MAPQ", 0)),
            strand=payload.get("strand", payload.get("STRAND", "+")),
            source=payload.get("source", payload.get("SRC", "UNKNOWN")),
            insert_seq=payload.get("insert_seq", payload.get("ins_seq")),
            num_splits=payload.get("num_splits", payload.get("NUM_SPLITS", 1)),
            sa_contigs=payload.get("sa_contigs", []),
            sorted_aligns=payload.get("sorted_aligns", []),
            bkps=payload.get("bkps", []),
            svm_score=payload.get("svm_score", 0.0),
            confidence=payload.get("confidence", payload.get("prob", 0.5)),
            support=payload.get("support", payload.get("SUPPORT", 1)),
        )

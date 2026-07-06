# signature_extraction.py
"""
Alignment-based SV signature collection with Signature object output.

This module extracts split-read and CIGAR-derived SV signatures from BAM files
and emits Signature objects compatible with the downstream feature engineering
pipeline.

Source tags:
- "CIGAR" for CIGAR-based insertion/deletion signals
- "SA" for split-alignment signals
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pysam

from .signature import Signature, SignatureType


LOGGER = logging.getLogger(__name__)


def _parse_cigar_items(cigar_string: str) -> List[Tuple[int, str]]:
    """Parse a CIGAR string into `(length, op)` tuples."""

    if not cigar_string:
        return []
    return [(int(length), op) for length, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar_string)]


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""

    trans = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(trans)[::-1]


# ============================================================
# Chromosome parsing utilities
# ============================================================

def parse_chrom_input(chrom_input: str, bam: pysam.AlignmentFile) -> List[str]:
    """
    Parse chromosome input string and return list of valid chromosome names.

    Supports multiple formats:
    - Single: "chr22" or "22"
    - Range: "1-5" or "chr1-chr5"
    - List: "1,2,3,4,5" or "chr1,chr2,chr3"
    - Mixed: "1-3,5,7-9"

    Args:
        chrom_input: Chromosome specification string
        bam: Open BAM file to validate chromosome names

    Returns:
        List of valid chromosome names found in BAM
    """
    # Get available chromosomes from BAM
    available_chroms = set(bam.references)

    # Build a mapping for flexible matching (e.g., "1" -> "chr1" or "chr1" -> "1")
    chrom_map = {}
    for chrom in available_chroms:
        chrom_map[chrom] = chrom
        # Handle chr prefix variations
        if chrom.startswith('chr'):
            chrom_map[chrom[3:]] = chrom  # "1" -> "chr1"
        else:
            chrom_map['chr' + chrom] = chrom  # "chr1" -> "1"

    def resolve_chrom(name: str) -> Optional[str]:
        """Resolve a chromosome name to its BAM reference name."""
        name = name.strip()
        if name in chrom_map:
            return chrom_map[name]
        return None

    def expand_range(range_str: str) -> List[str]:
        """Expand a range like '1-5' or 'chr1-chr5' to individual chromosomes."""
        parts = range_str.split('-')
        if len(parts) != 2:
            return [range_str]  # Not a valid range, return as-is

        start_str, end_str = parts[0].strip(), parts[1].strip()

        # Extract numeric parts
        start_prefix = ''
        end_prefix = ''

        # Handle 'chr' prefix
        if start_str.lower().startswith('chr'):
            start_prefix = start_str[:3]
            start_str = start_str[3:]
        if end_str.lower().startswith('chr'):
            end_prefix = end_str[:3]
            end_str = end_str[3:]

        # Try to parse as integers
        try:
            start_num = int(start_str)
            end_num = int(end_str)

            # Use the prefix from start (or none if no prefix)
            prefix = start_prefix or end_prefix

            # Generate chromosome names in range
            result = []
            for i in range(start_num, end_num + 1):
                chrom_name = f"{prefix}{i}" if prefix else str(i)
                result.append(chrom_name)
            return result
        except ValueError:
            # Not numeric, return as-is (handles X, Y, M cases)
            return [range_str]

    result_chroms = []

    # Split by comma first
    parts = chrom_input.split(',')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if it's a range (contains '-' but not at start for negative numbers)
        if '-' in part and not part.startswith('-'):
            # Could be a range like "1-5" or "chr1-chr5"
            expanded = expand_range(part)
            for chrom in expanded:
                resolved = resolve_chrom(chrom)
                if resolved and resolved not in result_chroms:
                    result_chroms.append(resolved)
        else:
            # Single chromosome
            resolved = resolve_chrom(part)
            if resolved and resolved not in result_chroms:
                result_chroms.append(resolved)

    return result_chroms


def get_chrom_length(bam: pysam.AlignmentFile, chrom: str) -> int:
    """Get the length of a chromosome from BAM header."""
    for ref, length in zip(bam.references, bam.lengths):
        if ref == chrom:
            return length
    return 0


# ============================================================
# Alignment signal constants
# ============================================================

dic_starnd = {1: '+', 2: '-'}

signal = {
    1 << 2: 0,       # unmapped
    1 >> 1: 1,       # normal forward (flag 0)
    1 << 4: 2,       # reverse complement (flag 16)
    1 << 11: 3,      # supplementary
    1 << 4 | 1 << 11: 4,  # supplementary + reverse
}

# CIGAR operation tables
OPLIST = [
    pysam.CBACK, pysam.CDEL, pysam.CDIFF, pysam.CEQUAL, pysam.CHARD_CLIP,
    pysam.CINS, pysam.CMATCH, pysam.CPAD, pysam.CREF_SKIP, pysam.CSOFT_CLIP
]

CHANGETABLE = {
    pysam.CMATCH:    (True, True),
    pysam.CINS:      (True, False),
    pysam.CDEL:      (False, True),
    pysam.CREF_SKIP: (False, True),
    pysam.CPAD:      (False, False),
    pysam.CEQUAL:    (True, True),
    pysam.CDIFF:     (True, True)
}

_max_op = max(OPLIST) + 1
CHANGEOP = [CHANGETABLE[i] if i in CHANGETABLE else (False, False) for i in range(_max_op)]
REFCHANGEOP = [CHANGETABLE[i][1] if i in CHANGETABLE else False for i in range(_max_op)]
INDELOP = [(i == pysam.CDEL or i == pysam.CINS) for i in range(_max_op)]


# ============================================================
# Alignment signal helper functions
# ============================================================

def detect_flag(flag: int) -> int:
    """Detect read type from flag."""
    return signal.get(flag, 0)


def acquire_clip_pos(deal_cigar: str) -> List[int]:
    """Get clip positions and reference bias from CIGAR string."""
    seq = _parse_cigar_items(deal_cigar)
    first_pos = seq[0][0] if seq[0][1] == 'S' else 0
    last_pos = seq[-1][0] if seq[-1][1] == 'S' else 0

    bias = 0
    for length, op in seq:
        if op in ('M', 'D', '=', 'X'):
            bias += length
    return [first_pos, last_pos, bias]


def generate_combine_sigs(sigs: List, chr_name: str, read_name: str,
                          svtype: str, candidate: List, merge_dis: int) -> None:
    """
    Combine nearby signals from the same read.
    Merge nearby CIGAR-derived insertion/deletion signals.
    """
    if len(sigs) == 0:
        return
    elif len(sigs) == 1:
        if svtype == 'INS':
            candidate.append((sigs[0][0], sigs[0][1], read_name, sigs[0][2], svtype, chr_name))
        else:
            candidate.append((sigs[0][0], sigs[0][1], read_name, svtype, chr_name))
    else:
        temp_sig = list(sigs[0])
        if svtype == "INS":
            temp_sig.append(sigs[0][0])  # track last position
            for i in sigs[1:]:
                if i[0] - temp_sig[3] <= merge_dis:
                    temp_sig[1] += i[1]
                    temp_sig[2] += i[2]
                    temp_sig[3] = i[0]
                else:
                    candidate.append((temp_sig[0], temp_sig[1], read_name, temp_sig[2], svtype, chr_name))
                    temp_sig = list(i)
                    temp_sig.append(i[0])
            candidate.append((temp_sig[0], temp_sig[1], read_name, temp_sig[2], svtype, chr_name))
        else:
            temp_sig.append(sum(sigs[0]))  # track end position
            for i in sigs[1:]:
                if i[0] - temp_sig[2] <= merge_dis:
                    temp_sig[1] += i[1]
                    temp_sig[2] = sum(i)
                else:
                    candidate.append((temp_sig[0], temp_sig[1], read_name, svtype, chr_name))
                    temp_sig = list(i)
                    temp_sig.append(sum(i))
            candidate.append((temp_sig[0], temp_sig[1], read_name, svtype, chr_name))


def analysis_inv(ele_1: List, ele_2: List, read_name: str,
                 candidate: List, sv_size: int, max_size: int) -> None:
    """Analyze inversion from split alignments."""
    def _keep(start: int, end: int) -> bool:
        size = abs(int(end) - int(start))
        return size >= sv_size and (size <= max_size or max_size == -1)

    if ele_1[5] == '+':
        # +-
        if ele_1[3] - ele_2[3] >= sv_size:
            if ele_2[0] + 0.5 * (ele_1[3] - ele_2[3]) >= ele_1[1] and _keep(ele_2[3], ele_1[3]):
                candidate.append(("++", ele_2[3], ele_1[3], read_name, "INV", ele_1[4]))
        if ele_2[3] - ele_1[3] >= sv_size:
            if ele_2[0] + 0.5 * (ele_2[3] - ele_1[3]) >= ele_1[1] and _keep(ele_1[3], ele_2[3]):
                candidate.append(("++", ele_1[3], ele_2[3], read_name, "INV", ele_1[4]))
    else:
        # -+
        if ele_2[2] - ele_1[2] >= sv_size:
            if ele_2[0] + 0.5 * (ele_2[2] - ele_1[2]) >= ele_1[1] and _keep(ele_1[2], ele_2[2]):
                candidate.append(("--", ele_1[2], ele_2[2], read_name, "INV", ele_1[4]))
        if ele_1[2] - ele_2[2] >= sv_size:
            if ele_2[0] + 0.5 * (ele_1[2] - ele_2[2]) >= ele_1[1] and _keep(ele_2[2], ele_1[2]):
                candidate.append(("--", ele_2[2], ele_1[2], read_name, "INV", ele_1[4]))


def analysis_bnd(ele_1: List, ele_2: List, read_name: str, candidate: List) -> None:
    """
    Analyze translocation (BND) from split alignments.
    BND types: A (N[chr:pos[), B (N]chr:pos]), C ([chr:pos[N), D (]chr:pos]N)
    """
    if ele_2[0] - ele_1[1] <= 100:
        if ele_1[5] == '+':
            if ele_2[5] == '+':
                if ele_1[4] < ele_2[4]:
                    candidate.append(('A', ele_1[3], ele_2[4], ele_2[2], read_name, "TRA", ele_1[4]))
                else:
                    candidate.append(('D', ele_2[2], ele_1[4], ele_1[3], read_name, "TRA", ele_2[4]))
            else:
                if ele_1[4] < ele_2[4]:
                    candidate.append(('B', ele_1[3], ele_2[4], ele_2[3], read_name, "TRA", ele_1[4]))
                else:
                    candidate.append(('B', ele_2[3], ele_1[4], ele_1[3], read_name, "TRA", ele_2[4]))
        else:
            if ele_2[5] == '+':
                if ele_1[4] < ele_2[4]:
                    candidate.append(('C', ele_1[2], ele_2[4], ele_2[2], read_name, "TRA", ele_1[4]))
                else:
                    candidate.append(('C', ele_2[2], ele_1[4], ele_1[2], read_name, "TRA", ele_2[4]))
            else:
                if ele_1[4] < ele_2[4]:
                    candidate.append(('D', ele_1[2], ele_2[4], ele_2[3], read_name, "TRA", ele_1[4]))
                else:
                    candidate.append(('A', ele_2[3], ele_1[4], ele_1[2], read_name, "TRA", ele_2[4]))


def analysis_split_read(split_read: List, sv_size: int, r_length: int,
                        read_name: str, candidate: Dict, max_size: int, query: str) -> None:
    """
    Analyze split reads to detect SVs.
    Core split-alignment analysis logic.

    split_read format: [read_start, read_end, ref_start, ref_end, chr, strand]
    """
    sp_list = sorted(split_read, key=lambda x: x[0])
    trigger_ins_tra = 0

    if len(sp_list) == 2:
        ele_1 = sp_list[0]
        ele_2 = sp_list[1]

        if ele_1[4] == ele_2[4]:  # Same chromosome
            if ele_1[5] != ele_2[5]:  # Different strands -> INV
                analysis_inv(ele_1, ele_2, read_name, candidate["INV"], sv_size, max_size)
            else:
                # Same strand: DUP, INS, DEL
                a = 0
                if ele_1[5] == '-':
                    ele_1 = [r_length - sp_list[a+1][1], r_length - sp_list[a+1][0]] + sp_list[a+1][2:]
                    ele_2 = [r_length - sp_list[a][1], r_length - sp_list[a][0]] + sp_list[a][2:]
                    query = _reverse_complement(query)

                # DUP detection
                if ele_1[3] - ele_2[2] >= sv_size:
                    if ele_2[0] - ele_1[1] >= ele_1[3] - ele_2[2]:
                        # INS from overlap
                        delta = ele_2[0] + ele_1[3] - ele_2[2] - ele_1[1]
                        if delta >= sv_size and (delta <= max_size or max_size == -1):
                            ins_seq = str(query[ele_1[1] + int((ele_1[3] - ele_2[2]) / 2):ele_2[0] - int((ele_1[3] - ele_2[2]) / 2)])
                            candidate["INS"].append(((ele_1[3] + ele_2[2]) / 2, delta, read_name, ins_seq, "INS", ele_2[4]))
                    elif ele_1[3] - ele_2[2] <= max_size or max_size == -1:
                        candidate["DUP"].append((ele_2[2], ele_1[3], read_name, "DUP", ele_2[4]))

                # INS detection
                delta_length = ele_2[0] + ele_1[3] - ele_2[2] - ele_1[1]
                if ele_1[3] - ele_2[2] < max(sv_size, delta_length / 5) and delta_length >= sv_size:
                    if ele_2[2] - ele_1[3] <= max(100, delta_length / 5) and (delta_length <= max_size or max_size == -1):
                        ins_seq = str(query[ele_1[1] + int((ele_2[2] - ele_1[3]) / 2):ele_2[0] - int((ele_2[2] - ele_1[3]) / 2)])
                        candidate["INS"].append(((ele_2[2] + ele_1[3]) / 2, delta_length, read_name, ins_seq, "INS", ele_2[4]))

                # DEL detection
                delta_length = ele_2[2] - ele_2[0] + ele_1[1] - ele_1[3]
                if ele_1[3] - ele_2[2] < max(sv_size, delta_length / 5) and delta_length >= sv_size:
                    if ele_2[0] - ele_1[1] <= max(100, delta_length / 5) and (delta_length <= max_size or max_size == -1):
                        candidate["DEL"].append((ele_1[3], delta_length, read_name, "DEL", ele_2[4]))
        else:
            # Different chromosomes -> TRA
            analysis_bnd(ele_1, ele_2, read_name, candidate["TRA"])

    elif len(sp_list) >= 3:
        # Complex split reads with 3+ segments
        for a in range(len(sp_list) - 2):
            ele_1 = sp_list[a]
            ele_2 = sp_list[a + 1]
            ele_3 = sp_list[a + 2]

            if ele_1[4] == ele_2[4] == ele_3[4]:  # All on same chromosome
                # Complex INV patterns
                if ele_1[5] == ele_3[5] and ele_1[5] != ele_2[5]:
                    if ele_2[5] == '-':
                        # +-+
                        if ele_2[0] + 0.5 * (ele_3[2] - ele_1[3]) >= ele_1[1] and ele_3[0] + 0.5 * (ele_3[2] - ele_1[3]) >= ele_2[1]:
                            if ele_2[2] >= ele_1[3] and ele_3[2] >= ele_2[3]:
                                if ele_2[3] - ele_1[3] <= max_size or max_size == -1:
                                    candidate["INV"].append(("++", ele_1[3], ele_2[3], read_name, "INV", ele_1[4]))
                                if ele_3[2] - ele_2[2] <= max_size or max_size == -1:
                                    candidate["INV"].append(("--", ele_2[2], ele_3[2], read_name, "INV", ele_1[4]))
                    else:
                        # -+-
                        if ele_1[1] <= ele_2[0] + 0.5 * (ele_1[2] - ele_3[3]) and ele_3[0] + 0.5 * (ele_1[2] - ele_3[3]) >= ele_2[1]:
                            if ele_2[2] - ele_3[3] >= -50 and ele_1[2] - ele_2[3] >= -50:
                                if ele_2[3] - ele_3[3] <= max_size or max_size == -1:
                                    candidate["INV"].append(("++", ele_3[3], ele_2[3], read_name, "INV", ele_1[4]))
                                if ele_1[2] - ele_2[2] <= max_size or max_size == -1:
                                    candidate["INV"].append(("--", ele_2[2], ele_1[2], read_name, "INV", ele_1[4]))

                # Same strand processing for DUP/INS/DEL
                if ele_1[5] == ele_2[5] == ele_3[5]:
                    if ele_1[5] == '-':
                        ele_1 = [r_length - sp_list[a + 2][1], r_length - sp_list[a + 2][0]] + sp_list[a + 2][2:]
                        ele_2 = [r_length - sp_list[a + 1][1], r_length - sp_list[a + 1][0]] + sp_list[a + 1][2:]
                        ele_3 = [r_length - sp_list[a][1], r_length - sp_list[a][0]] + sp_list[a][2:]
                        query_res = _reverse_complement(query)
                    else:
                        query_res = query

                    # DUP
                    if ele_2[3] - ele_3[2] >= sv_size and ele_2[2] < ele_3[3] and (ele_2[3] - ele_3[2] <= max_size or max_size == -1):
                        candidate["DUP"].append((ele_3[2], ele_2[3], read_name, "DUP", ele_2[4]))

                    if a == 0:
                        if ele_1[3] - ele_2[2] >= sv_size and (ele_1[3] - ele_2[2] <= max_size or max_size == -1):
                            candidate["DUP"].append((ele_2[2], ele_1[3], read_name, "DUP", ele_2[4]))

                    # INS
                    delta_length = ele_2[0] + ele_1[3] - ele_2[2] - ele_1[1]
                    if ele_1[3] - ele_2[2] < max(sv_size, delta_length / 5) and delta_length >= sv_size:
                        if ele_2[2] - ele_1[3] <= max(100, delta_length / 5) and (delta_length <= max_size or max_size == -1):
                            if ele_3[2] >= ele_2[3]:
                                ins_seq = str(query_res[ele_1[1] + int((ele_2[2] - ele_1[3]) / 2):ele_2[0] - int((ele_2[2] - ele_1[3]) / 2)])
                                candidate["INS"].append(((ele_2[2] + ele_1[3]) / 2, delta_length, read_name, ins_seq, "INS", ele_2[4]))

                    # DEL
                    delta_length = ele_2[2] - ele_2[0] + ele_1[1] - ele_1[3]
                    if ele_1[3] - ele_2[2] < max(sv_size, delta_length / 5) and delta_length >= sv_size:
                        if ele_2[0] - ele_1[1] <= max(100, delta_length / 5) and (delta_length <= max_size or max_size == -1):
                            if ele_3[2] >= ele_2[3]:
                                candidate["DEL"].append((ele_1[3], delta_length, read_name, "DEL", ele_2[4]))

            elif ele_1[4] != ele_2[4]:
                trigger_ins_tra = 1
                analysis_bnd(ele_1, ele_2, read_name, candidate["TRA"])
                if a == len(sp_list) - 3 and ele_2[4] != ele_3[4]:
                    analysis_bnd(ele_2, ele_3, read_name, candidate["TRA"])

    # Handle INS involved in TRA (complex case)
    if len(sp_list) >= 3 and trigger_ins_tra == 1:
        if sp_list[0][4] == sp_list[-1][4] and sp_list[0][5] == sp_list[-1][5]:
            if sp_list[0][5] == '+':
                ele_1 = sp_list[0]
                ele_2 = sp_list[-1]
                query_res = query
            else:
                ele_1 = [r_length - sp_list[-1][1], r_length - sp_list[-1][0]] + sp_list[-1][2:]
                ele_2 = [r_length - sp_list[0][1], r_length - sp_list[0][0]] + sp_list[0][2:]
                query_res = _reverse_complement(query)

            dis_ref = ele_2[2] - ele_1[3]
            dis_read = ele_2[0] - ele_1[1]
            if abs(dis_ref) < max(sv_size, (dis_read - dis_ref) / 5) and dis_read - dis_ref >= sv_size:
                if dis_read - dis_ref <= max_size or max_size == -1:
                    ins_seq = str(query_res[ele_1[1] + int(dis_ref / 2):ele_2[0] - int(dis_ref / 2)])
                    candidate["INS"].append((min(ele_2[2], ele_1[3]), dis_read - dis_ref, read_name, ins_seq, "INS", ele_2[4]))

            if dis_ref <= -sv_size and (-dis_ref <= max_size or max_size == -1):
                candidate["DUP"].append((ele_2[2], ele_1[3], read_name, "DUP", ele_2[4]))


def organize_split_signal(primary_info: List, supplementary_info: List, total_l: int,
                          sv_size: int, min_mapq: int, max_split_parts: int,
                          read_name: str, candidate: Dict, max_size: int, query: str) -> None:
    """Organize split alignment signals from SA tag."""
    split_read = []
    if len(primary_info) > 0:
        split_read.append(primary_info)

    for sa_entry in supplementary_info:
        seq = sa_entry.split(',')
        local_chr = seq[0]
        local_start = int(seq[1])
        local_cigar = seq[3]
        local_strand = seq[2]
        local_mapq = int(seq[4])

        if local_mapq >= min_mapq:
            local_set = acquire_clip_pos(local_cigar)
            if local_strand == '+':
                split_read.append([local_set[0], total_l - local_set[1], local_start,
                                   local_start + local_set[2], local_chr, local_strand])
            else:
                try:
                    split_read.append([local_set[1], total_l - local_set[0], local_start,
                                       local_start + local_set[2], local_chr, local_strand])
                except:
                    pass

    if len(split_read) <= max_split_parts or max_split_parts == -1:
        analysis_split_read(split_read, sv_size, total_l, read_name, candidate, max_size, query)


# ============================================================
# Main collection function - outputs Signature objects
# ============================================================

def collect_signatures_region(
    bam_path: str,
    contig: str,
    start: int,
    end: int,
    min_sv_size: int = 30,
    max_sv_size: int = 100000,
    min_mapq: int = 20,
    min_read_len: int = 500,
    min_siglength: int = 30,
    merge_del_threshold: int = 0,
    merge_ins_threshold: int = 100,
    max_split_parts: int = 7
) -> List[Signature]:
    """
    Collect SV signatures from a BAM region using alignment evidence.

    Args:
        bam_path: Path to BAM file
        contig: Chromosome name
        start: Start position (0-based)
        end: End position
        min_sv_size: Minimum SV size
        max_sv_size: Maximum SV size
        min_mapq: Minimum mapping quality
        min_read_len: Minimum read length
        min_siglength: Minimum signal length for CIGAR
        merge_del_threshold: Merge distance for DEL
        merge_ins_threshold: Merge distance for INS
        max_split_parts: Maximum split parts

    Returns:
        List[Signature]: Signatures compatible with feature engineering pipeline
    """
    # Separate candidates for CIGAR and SA sources
    candidate_cigar = {"DEL": [], "INS": [], "DUP": [], "INV": [], "TRA": []}
    candidate_sa = {"DEL": [], "INS": [], "DUP": [], "INV": [], "TRA": []}

    bam = pysam.AlignmentFile(bam_path, "rb")

    for read in bam.fetch(contig, start, end):
        # Only extract signatures from primary mapped alignments.
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue

        # Ensure read starts in region
        if read.reference_start < start:
            continue

        # Skip short reads
        if read.query_length is None or read.query_length < min_read_len:
            continue

        # Skip reads without CIGAR
        if read.cigar is None or len(read.cigar) == 0:
            continue

        process_signal = detect_flag(read.flag)

        # ========== 1. CIGAR-based signals ==========
        if read.mapq >= min_mapq:
            combine_ins = []
            combine_del = []

            sig_start = read.reference_start
            shift_ins_read = 0

            # Handle leading clip
            if read.cigar[0][0] == 4:  # soft clip
                pass
            elif read.cigar[0][0] == 5:  # hard clip
                shift_ins_read = -read.cigar[0][1]

            for op, oplen in read.cigartuples:
                if op != 2:  # Not DEL
                    shift_ins_read += oplen

                if oplen >= min_siglength and INDELOP[op]:
                    if op == 2:  # DEL
                        combine_del.append([sig_start, oplen])
                        sig_start += oplen
                    else:  # INS
                        ins_seq = ""
                        if read.query_sequence:
                            ins_seq = str(read.query_sequence[shift_ins_read - oplen:shift_ins_read])
                        combine_ins.append([sig_start, oplen, ins_seq])
                else:
                    if REFCHANGEOP[op]:
                        sig_start += oplen

            # Merge signals from same read
            generate_combine_sigs(combine_ins, contig, read.query_name,
                                  "INS", candidate_cigar["INS"], merge_ins_threshold)
            generate_combine_sigs(combine_del, contig, read.query_name,
                                  "DEL", candidate_cigar["DEL"], merge_del_threshold)

        # ========== 2. SA tag based signals ==========
        if process_signal == 1 or process_signal == 2:  # flag 0 or 16
            tags = dict(read.get_tags())

            # Calculate primary alignment info
            softclip_left = 0
            softclip_right = 0
            hardclip_left = 0
            hardclip_right = 0
            pos_start = read.reference_start
            pos_end = read.reference_end

            if read.cigar[0][0] == 4:
                softclip_left = read.cigar[0][1]
            elif read.cigar[0][0] == 5:
                hardclip_left = read.cigar[0][1]
            if read.cigar[-1][0] == 4:
                softclip_right = read.cigar[-1][1]
            elif read.cigar[-1][0] == 5:
                hardclip_right = read.cigar[-1][1]

            if hardclip_left != 0:
                softclip_left = hardclip_left
            if hardclip_right != 0:
                softclip_right = hardclip_right

            if read.mapq >= min_mapq:
                if process_signal == 1:
                    primary_info = [softclip_left, read.query_length - softclip_right,
                                    pos_start, pos_end, contig, dic_starnd[process_signal]]
                else:
                    primary_info = [softclip_right, read.query_length - softclip_left,
                                    pos_start, pos_end, contig, dic_starnd[process_signal]]
            else:
                primary_info = []

            if 'SA' in tags:
                if process_signal == 1:
                    query_seq = read.query_sequence or ""
                else:
                    query_seq = _reverse_complement(read.query_sequence or "")

                supplementary_info = tags['SA'].split(';')[:-1]
                organize_split_signal(primary_info, supplementary_info, read.query_length,
                                      min_sv_size, min_mapq, max_split_parts,
                                      read.query_name, candidate_sa, max_sv_size, query_seq)

    bam.close()

    # ========== Convert to Signature objects ==========
    signatures = []
    sid = 0

    # CIGAR-based DEL
    for sig in candidate_cigar["DEL"]:
        # Format: (pos, length, read_name, "DEL", chr)
        signatures.append(Signature(
            sid=sid,
            contig=sig[-1],
            tstart=int(sig[0]),
            tend=int(sig[0]) + int(sig[1]),
            svtype="DEL",
            svlen=int(sig[1]),
            qname=sig[2],
            mapq=min_mapq,
            source="CIGAR",
            confidence=0.5
        ))
        sid += 1

    # CIGAR-based INS
    for sig in candidate_cigar["INS"]:
        # Format: (pos, length, read_name, insert_seq, "INS", chr)
        signatures.append(Signature(
            sid=sid,
            contig=sig[-1],
            tstart=int(sig[0]),
            tend=int(sig[0]) + 1,
            svtype="INS",
            svlen=int(sig[1]),
            qname=sig[2],
            mapq=min_mapq,
            source="CIGAR",
            insert_seq=sig[3] if len(sig) > 3 else None,
            confidence=0.5
        ))
        sid += 1

    # SA-based DEL
    for sig in candidate_sa["DEL"]:
        signatures.append(Signature(
            sid=sid,
            contig=sig[-1],
            tstart=int(sig[0]),
            tend=int(sig[0]) + int(sig[1]),
            svtype="DEL",
            svlen=int(sig[1]),
            qname=sig[2],
            mapq=min_mapq,
            source="SA_TAG",
            confidence=0.5
        ))
        sid += 1

    # SA-based INS
    for sig in candidate_sa["INS"]:
        signatures.append(Signature(
            sid=sid,
            contig=sig[-1],
            tstart=int(sig[0]),
            tend=int(sig[0]) + 1,
            svtype="INS",
            svlen=int(sig[1]),
            qname=sig[2],
            mapq=min_mapq,
            source="SA_TAG",
            insert_seq=sig[3] if len(sig) > 3 else None,
            confidence=0.5
        ))
        sid += 1

    # SA-based DUP
    for sig in candidate_sa["DUP"]:
        # Format: (start, end, read_name, "DUP", chr)
        dup_start = int(sig[0])
        dup_end = int(sig[1])
        signatures.append(Signature(
            sid=sid,
            contig=sig[-1],
            tstart=dup_start,
            tend=dup_end,
            svtype="DUP",
            svlen=dup_end - dup_start,
            qname=sig[2],
            mapq=min_mapq,
            source="SA_TAG",
            confidence=0.5
        ))
        sid += 1

    # SA-based INV
    for sig in candidate_sa["INV"]:
        # Format: (strand_pattern, start, end, read_name, "INV", chr)
        inv_start = int(sig[1])
        inv_end = int(sig[2])
        signatures.append(Signature(
            sid=sid,
            contig=sig[-1],
            tstart=inv_start,
            tend=inv_end,
            svtype="INV",
            svlen=inv_end - inv_start,
            qname=sig[3],
            mapq=min_mapq,
            strand=sig[0],  # ++/-- pattern
            source="SA_TAG",
            confidence=0.5
        ))
        sid += 1

    # SA-based TRA
    for sig in candidate_sa["TRA"]:
        # Format: (type, pos1, chr2, pos2, read_name, "TRA", chr1)
        signatures.append(Signature(
            sid=sid,
            contig=sig[-1],
            tstart=int(sig[1]),
            tend=int(sig[1]) + 1,
            svtype="TRA",
            svlen=0,
            qname=sig[4],
            mapq=min_mapq,
            source="SA_TAG",
            sa_contigs=[sig[2]],  # chr2
            confidence=0.5
        ))
        sid += 1

    return signatures


def collect_signatures_multi_chrom(
    bam_path: str,
    chrom_input: str,
    min_sv_size: int = 30,
    max_sv_size: int = 100000,
    min_mapq: int = 20,
    min_read_len: int = 500,
    min_siglength: int = 30,
    merge_del_threshold: int = 0,
    merge_ins_threshold: int = 100,
    max_split_parts: int = 7
) -> List[Signature]:
    """
    Collect SV signatures from multiple chromosomes.

    Args:
        bam_path: Path to BAM file
        chrom_input: Chromosome specification string. Supports:
            - Single: "chr22" or "22"
            - Range: "1-5" or "chr1-chr5"
            - List: "1,2,3,4,5" or "chr1,chr2,chr3"
            - Mixed: "1-3,5,7-9"
        min_sv_size: Minimum SV size
        max_sv_size: Maximum SV size
        min_mapq: Minimum mapping quality
        min_read_len: Minimum read length
        min_siglength: Minimum signal length for CIGAR
        merge_del_threshold: Merge distance for DEL
        merge_ins_threshold: Merge distance for INS
        max_split_parts: Maximum split parts

    Returns:
        List[Signature]: Combined signatures from all specified chromosomes

    Examples:
        # Single chromosome
        sigs = collect_signatures_multi_chrom("data.bam", "chr22")

        # Range of chromosomes
        sigs = collect_signatures_multi_chrom("data.bam", "1-5")

        # List of chromosomes
        sigs = collect_signatures_multi_chrom("data.bam", "1,2,3,4,5")

        # Mixed format
        sigs = collect_signatures_multi_chrom("data.bam", "1-3,5,chr7-chr9,X,Y")
    """
    bam = pysam.AlignmentFile(bam_path, "rb")

    # Parse chromosome input
    chroms = parse_chrom_input(chrom_input, bam)

    if not chroms:
        bam.close()
        raise ValueError(f"No valid chromosomes found for input: {chrom_input}")

    all_signatures = []

    for chrom in chroms:
        chrom_length = get_chrom_length(bam, chrom)
        if chrom_length == 0:
            continue

        # Collect signatures for this chromosome
        sigs = collect_signatures_region(
            bam_path=bam_path,
            contig=chrom,
            start=0,
            end=chrom_length,
            min_sv_size=min_sv_size,
            max_sv_size=max_sv_size,
            min_mapq=min_mapq,
            min_read_len=min_read_len,
            min_siglength=min_siglength,
            merge_del_threshold=merge_del_threshold,
            merge_ins_threshold=merge_ins_threshold,
            max_split_parts=max_split_parts
        )
        all_signatures.extend(sigs)

    bam.close()

    # Re-assign sequential sids
    for i, sig in enumerate(all_signatures):
        sig.sid = i

    return all_signatures


# ============================================================
# Convenience function with SimpleNamespace options
# ============================================================

def analyze_alignments(
    alignments: List[pysam.AlignedSegment],
    bam: pysam.AlignmentFile,
    opt: Any,
    part_num: int = 0
) -> List[Signature]:
    """
    Analyze alignments with options object (compatible with other collect modules).

    This is a wrapper that accepts alignments list instead of BAM path.
    """
    min_sv_size = getattr(opt, 'min_sv_size', 30)
    max_sv_size = getattr(opt, 'max_sv_size', 100000)
    min_mapq = getattr(opt, 'min_mapping_quality', 20)
    min_read_len = getattr(opt, 'min_read_len', 500)
    min_siglength = getattr(opt, 'min_siglength', 30)
    merge_del_threshold = getattr(opt, 'merge_del_threshold', 0)
    merge_ins_threshold = getattr(opt, 'merge_ins_threshold', 100)
    max_split_parts = getattr(opt, 'max_split_parts', 7)

    candidate_cigar = {"DEL": [], "INS": [], "DUP": [], "INV": [], "TRA": []}
    candidate_sa = {"DEL": [], "INS": [], "DUP": [], "INV": [], "TRA": []}

    for read in alignments:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue

        if read.query_length is None or read.query_length < min_read_len:
            continue

        if read.cigar is None or len(read.cigar) == 0:
            continue

        contig = read.reference_name
        process_signal = detect_flag(read.flag)

        # CIGAR-based
        if read.mapq >= min_mapq:
            combine_ins = []
            combine_del = []
            sig_start = read.reference_start
            shift_ins_read = 0

            if read.cigar[0][0] == 4:
                pass
            elif read.cigar[0][0] == 5:
                shift_ins_read = -read.cigar[0][1]

            for op, oplen in read.cigartuples:
                if op != 2:
                    shift_ins_read += oplen

                if oplen >= min_siglength and INDELOP[op]:
                    if op == 2:
                        combine_del.append([sig_start, oplen])
                        sig_start += oplen
                    else:
                        ins_seq = ""
                        if read.query_sequence:
                            ins_seq = str(read.query_sequence[shift_ins_read - oplen:shift_ins_read])
                        combine_ins.append([sig_start, oplen, ins_seq])
                else:
                    if REFCHANGEOP[op]:
                        sig_start += oplen

            generate_combine_sigs(combine_ins, contig, read.query_name,
                                  "INS", candidate_cigar["INS"], merge_ins_threshold)
            generate_combine_sigs(combine_del, contig, read.query_name,
                                  "DEL", candidate_cigar["DEL"], merge_del_threshold)

        # SA-based
        if process_signal == 1 or process_signal == 2:
            tags = dict(read.get_tags())

            softclip_left = 0
            softclip_right = 0
            hardclip_left = 0
            hardclip_right = 0
            pos_start = read.reference_start
            pos_end = read.reference_end

            if read.cigar[0][0] == 4:
                softclip_left = read.cigar[0][1]
            elif read.cigar[0][0] == 5:
                hardclip_left = read.cigar[0][1]
            if read.cigar[-1][0] == 4:
                softclip_right = read.cigar[-1][1]
            elif read.cigar[-1][0] == 5:
                hardclip_right = read.cigar[-1][1]

            if hardclip_left != 0:
                softclip_left = hardclip_left
            if hardclip_right != 0:
                softclip_right = hardclip_right

            if read.mapq >= min_mapq:
                if process_signal == 1:
                    primary_info = [softclip_left, read.query_length - softclip_right,
                                    pos_start, pos_end, contig, dic_starnd[process_signal]]
                else:
                    primary_info = [softclip_right, read.query_length - softclip_left,
                                    pos_start, pos_end, contig, dic_starnd[process_signal]]
            else:
                primary_info = []

            if 'SA' in tags:
                if process_signal == 1:
                    query_seq = read.query_sequence or ""
                else:
                    query_seq = _reverse_complement(read.query_sequence or "")

                supplementary_info = tags['SA'].split(';')[:-1]
                organize_split_signal(primary_info, supplementary_info, read.query_length,
                                      min_sv_size, min_mapq, max_split_parts,
                                      read.query_name, candidate_sa, max_sv_size, query_seq)

    # Convert to Signatures
    signatures = []
    sid = 0

    for sig in candidate_cigar["DEL"]:
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=int(sig[0]), tend=int(sig[0]) + int(sig[1]),
            svtype="DEL", svlen=int(sig[1]), qname=sig[2], mapq=min_mapq, source="CIGAR"
        ))
        sid += 1

    for sig in candidate_cigar["INS"]:
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=int(sig[0]), tend=int(sig[0]) + 1,
            svtype="INS", svlen=int(sig[1]), qname=sig[2], mapq=min_mapq, source="CIGAR",
            insert_seq=sig[3] if len(sig) > 3 else None
        ))
        sid += 1

    for sig in candidate_sa["DEL"]:
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=int(sig[0]), tend=int(sig[0]) + int(sig[1]),
            svtype="DEL", svlen=int(sig[1]), qname=sig[2], mapq=min_mapq, source="SA_TAG"
        ))
        sid += 1

    for sig in candidate_sa["INS"]:
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=int(sig[0]), tend=int(sig[0]) + 1,
            svtype="INS", svlen=int(sig[1]), qname=sig[2], mapq=min_mapq, source="SA_TAG",
            insert_seq=sig[3] if len(sig) > 3 else None
        ))
        sid += 1

    for sig in candidate_sa["DUP"]:
        dup_start, dup_end = int(sig[0]), int(sig[1])
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=dup_start, tend=dup_end,
            svtype="DUP", svlen=dup_end - dup_start, qname=sig[2], mapq=min_mapq, source="SA_TAG"
        ))
        sid += 1

    for sig in candidate_sa["INV"]:
        inv_start, inv_end = int(sig[1]), int(sig[2])
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=inv_start, tend=inv_end,
            svtype="INV", svlen=inv_end - inv_start, qname=sig[3], mapq=min_mapq, strand=sig[0], source="SA_TAG"
        ))
        sid += 1

    for sig in candidate_sa["TRA"]:
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=int(sig[1]), tend=int(sig[1]) + 1,
            svtype="TRA", svlen=0, qname=sig[4], mapq=min_mapq, source="SA_TAG", sa_contigs=[sig[2]]
        ))
        sid += 1

    return signatures


# ============================================================
# Multiprocessing support for fast collection
# ============================================================

# 全局变量用于worker进程共享参数
_worker_params = {}


def _init_worker(bam_path: str, min_sv_size: int, max_sv_size: int, min_mapq: int,
                 min_read_len: int, min_siglength: int, merge_del_threshold: int,
                 merge_ins_threshold: int, max_split_parts: int):
    """Initialize worker process with shared parameters."""
    global _worker_params
    _worker_params = {
        'bam_path': bam_path,
        'min_sv_size': min_sv_size,
        'max_sv_size': max_sv_size,
        'min_mapq': min_mapq,
        'min_read_len': min_read_len,
        'min_siglength': min_siglength,
        'merge_del_threshold': merge_del_threshold,
        'merge_ins_threshold': merge_ins_threshold,
        'max_split_parts': max_split_parts,
        'bam': None  # Will be opened per-worker
    }


def _collect_region_worker(task: Tuple[str, int, int]) -> Tuple[List, List, List, List, List, List, List, List, List, List]:
    """
    Worker function - processes a single region and returns raw candidates.
    Returns tuples instead of Signature objects to avoid serialization overhead.
    """
    global _worker_params
    chrom, start, end = task

    # Open BAM once per worker (lazy initialization)
    if _worker_params.get('bam') is None:
        _worker_params['bam'] = pysam.AlignmentFile(_worker_params['bam_path'], "rb")

    bam = _worker_params['bam']
    min_sv_size = _worker_params['min_sv_size']
    max_sv_size = _worker_params['max_sv_size']
    min_mapq = _worker_params['min_mapq']
    min_read_len = _worker_params['min_read_len']
    min_siglength = _worker_params['min_siglength']
    merge_del_threshold = _worker_params['merge_del_threshold']
    merge_ins_threshold = _worker_params['merge_ins_threshold']
    max_split_parts = _worker_params['max_split_parts']

    # Raw candidate lists (not Signature objects yet)
    cigar_del = []
    cigar_ins = []
    sa_del = []
    sa_ins = []
    sa_dup = []
    sa_inv = []
    sa_tra = []

    try:
        for read in bam.fetch(chrom, start, end):
            # Only extract signatures from primary mapped alignments.
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            # Skip reads starting before region
            if read.reference_start < start:
                continue

            # Skip short reads
            if read.query_length is None or read.query_length < min_read_len:
                continue

            # Skip reads without CIGAR
            if read.cigar is None or len(read.cigar) == 0:
                continue

            process_signal = detect_flag(read.flag)

            # ========== CIGAR-based signals ==========
            if read.mapq >= min_mapq:
                combine_ins = []
                combine_del = []
                sig_start = read.reference_start
                shift_ins_read = 0

                if read.cigar[0][0] == 4:
                    pass
                elif read.cigar[0][0] == 5:
                    shift_ins_read = -read.cigar[0][1]

                for op, oplen in read.cigartuples:
                    if op != 2:
                        shift_ins_read += oplen

                    if oplen >= min_siglength and INDELOP[op]:
                        if op == 2:  # DEL
                            combine_del.append([sig_start, oplen])
                            sig_start += oplen
                        else:  # INS
                            ins_seq = ""
                            if read.query_sequence:
                                ins_seq = str(read.query_sequence[shift_ins_read - oplen:shift_ins_read])
                            combine_ins.append([sig_start, oplen, ins_seq])
                    else:
                        if REFCHANGEOP[op]:
                            sig_start += oplen

                # Collect into local candidates
                _generate_combine_sigs_fast(combine_ins, chrom, read.query_name, "INS", cigar_ins, merge_ins_threshold)
                _generate_combine_sigs_fast(combine_del, chrom, read.query_name, "DEL", cigar_del, merge_del_threshold)

            # ========== SA tag based signals ==========
            if process_signal == 1 or process_signal == 2:
                tags = dict(read.get_tags())

                softclip_left = 0
                softclip_right = 0
                hardclip_left = 0
                hardclip_right = 0
                pos_start = read.reference_start
                pos_end = read.reference_end

                if read.cigar[0][0] == 4:
                    softclip_left = read.cigar[0][1]
                elif read.cigar[0][0] == 5:
                    hardclip_left = read.cigar[0][1]
                if read.cigar[-1][0] == 4:
                    softclip_right = read.cigar[-1][1]
                elif read.cigar[-1][0] == 5:
                    hardclip_right = read.cigar[-1][1]

                if hardclip_left != 0:
                    softclip_left = hardclip_left
                if hardclip_right != 0:
                    softclip_right = hardclip_right

                if read.mapq >= min_mapq:
                    if process_signal == 1:
                        primary_info = [softclip_left, read.query_length - softclip_right,
                                        pos_start, pos_end, chrom, dic_starnd[process_signal]]
                    else:
                        primary_info = [softclip_right, read.query_length - softclip_left,
                                        pos_start, pos_end, chrom, dic_starnd[process_signal]]
                else:
                    primary_info = []

                if 'SA' in tags:
                    if process_signal == 1:
                        query_seq = read.query_sequence or ""
                    else:
                        query_seq = _reverse_complement(read.query_sequence or "")

                    supplementary_info = tags['SA'].split(';')[:-1]
                    # Use local candidate dict
                    local_candidate = {"DEL": sa_del, "INS": sa_ins, "DUP": sa_dup, "INV": sa_inv, "TRA": sa_tra}
                    organize_split_signal(primary_info, supplementary_info, read.query_length,
                                          min_sv_size, min_mapq, max_split_parts,
                                          read.query_name, local_candidate, max_sv_size, query_seq)
    except Exception as e:
        # Silently handle region fetch errors (e.g., invalid regions)
        pass

    return (cigar_del, cigar_ins, sa_del, sa_ins, sa_dup, sa_inv, sa_tra)


def _generate_combine_sigs_fast(sigs: List, chr_name: str, read_name: str,
                                 svtype: str, candidate: List, merge_dis: int) -> None:
    """Fast version of generate_combine_sigs - appends directly to candidate list."""
    if len(sigs) == 0:
        return
    elif len(sigs) == 1:
        if svtype == 'INS':
            candidate.append((sigs[0][0], sigs[0][1], read_name, sigs[0][2], svtype, chr_name))
        else:
            candidate.append((sigs[0][0], sigs[0][1], read_name, svtype, chr_name))
    else:
        temp_sig = list(sigs[0])
        if svtype == "INS":
            temp_sig.append(sigs[0][0])
            for i in sigs[1:]:
                if i[0] - temp_sig[3] <= merge_dis:
                    temp_sig[1] += i[1]
                    temp_sig[2] += i[2]
                    temp_sig[3] = i[0]
                else:
                    candidate.append((temp_sig[0], temp_sig[1], read_name, temp_sig[2], svtype, chr_name))
                    temp_sig = list(i)
                    temp_sig.append(i[0])
            candidate.append((temp_sig[0], temp_sig[1], read_name, temp_sig[2], svtype, chr_name))
        else:
            temp_sig.append(sum(sigs[0]))
            for i in sigs[1:]:
                if i[0] - temp_sig[2] <= merge_dis:
                    temp_sig[1] += i[1]
                    temp_sig[2] = sum(i)
                else:
                    candidate.append((temp_sig[0], temp_sig[1], read_name, svtype, chr_name))
                    temp_sig = list(i)
                    temp_sig.append(sum(i))
            candidate.append((temp_sig[0], temp_sig[1], read_name, svtype, chr_name))


def collect_signatures_parallel(
    bam_path: str,
    chrom_input: str,
    num_processes: int = 4,
    min_sv_size: int = 30,
    max_sv_size: int = 100000,
    min_mapq: int = 20,
    min_read_len: int = 500,
    min_siglength: int = 30,
    merge_del_threshold: int = 0,
    merge_ins_threshold: int = 100,
    max_split_parts: int = 7,
    region_size: int = 5000000  # 5Mb per region
) -> List[Signature]:
    """
    Collect SV signatures using multiprocessing with region-based parallelization.

    This uses fixed-size region splitting for better load balancing.

    Args:
        bam_path: Path to BAM file
        chrom_input: Chromosome specification string (e.g., "1-22", "1,2,3", "chr1-chr5")
        num_processes: Number of parallel processes (default: 4)
        min_sv_size: Minimum SV size
        max_sv_size: Maximum SV size
        min_mapq: Minimum mapping quality
        min_read_len: Minimum read length
        min_siglength: Minimum signal length for CIGAR
        merge_del_threshold: Merge distance for DEL
        merge_ins_threshold: Merge distance for INS
        max_split_parts: Maximum split parts
        region_size: Size of each parallel region in bp (default: 5Mb)

    Returns:
        List[Signature]: Combined signatures from all specified chromosomes
    """
    import multiprocessing as mp
    from functools import partial

    bam = pysam.AlignmentFile(bam_path, "rb")
    chroms = parse_chrom_input(chrom_input, bam)

    if not chroms:
        bam.close()
        raise ValueError(f"No valid chromosomes found for input: {chrom_input}")

    # Create fixed-size region tasks
    tasks = []
    for chrom in chroms:
        chrom_length = get_chrom_length(bam, chrom)
        if chrom_length == 0:
            continue

        # Split chromosome into regions
        for start in range(0, chrom_length, region_size):
            end = min(start + region_size, chrom_length)
            tasks.append((chrom, start, end))

    bam.close()

    if not tasks:
        return []

    LOGGER.info(
        "signature_extraction parallel_start chromosomes=%s regions=%s processes=%s",
        len(chroms),
        len(tasks),
        num_processes,
    )

    # Use multiprocessing pool with initializer
    with mp.Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(bam_path, min_sv_size, max_sv_size, min_mapq, min_read_len,
                  min_siglength, merge_del_threshold, merge_ins_threshold, max_split_parts)
    ) as pool:
        # Use imap_unordered for progress tracking
        results = []
        completed = 0
        for result in pool.imap_unordered(_collect_region_worker, tasks):
            results.append(result)
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                LOGGER.info(
                    "signature_extraction parallel_progress completed=%s total=%s percent=%.1f",
                    completed,
                    len(tasks),
                    100 * completed / len(tasks),
                )

    # Merge all results
    all_cigar_del = []
    all_cigar_ins = []
    all_sa_del = []
    all_sa_ins = []
    all_sa_dup = []
    all_sa_inv = []
    all_sa_tra = []

    for result in results:
        cigar_del, cigar_ins, sa_del, sa_ins, sa_dup, sa_inv, sa_tra = result
        all_cigar_del.extend(cigar_del)
        all_cigar_ins.extend(cigar_ins)
        all_sa_del.extend(sa_del)
        all_sa_ins.extend(sa_ins)
        all_sa_dup.extend(sa_dup)
        all_sa_inv.extend(sa_inv)
        all_sa_tra.extend(sa_tra)

    # Convert to Signature objects
    signatures = []
    sid = 0

    # CIGAR DEL
    for sig in all_cigar_del:
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=int(sig[0]), tend=int(sig[0]) + int(sig[1]),
            svtype="DEL", svlen=int(sig[1]), qname=sig[2], mapq=min_mapq, source="CIGAR"
        ))
        sid += 1

    # CIGAR INS
    for sig in all_cigar_ins:
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=int(sig[0]), tend=int(sig[0]) + 1,
            svtype="INS", svlen=int(sig[1]), qname=sig[2], mapq=min_mapq, source="CIGAR",
            insert_seq=sig[3] if len(sig) > 3 else None
        ))
        sid += 1

    # SA DEL
    for sig in all_sa_del:
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=int(sig[0]), tend=int(sig[0]) + int(sig[1]),
            svtype="DEL", svlen=int(sig[1]), qname=sig[2], mapq=min_mapq, source="SA_TAG"
        ))
        sid += 1

    # SA INS
    for sig in all_sa_ins:
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=int(sig[0]), tend=int(sig[0]) + 1,
            svtype="INS", svlen=int(sig[1]), qname=sig[2], mapq=min_mapq, source="SA_TAG",
            insert_seq=sig[3] if len(sig) > 3 else None
        ))
        sid += 1

    # SA DUP
    for sig in all_sa_dup:
        dup_start, dup_end = int(sig[0]), int(sig[1])
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=dup_start, tend=dup_end,
            svtype="DUP", svlen=dup_end - dup_start, qname=sig[2], mapq=min_mapq, source="SA_TAG"
        ))
        sid += 1

    # SA INV
    for sig in all_sa_inv:
        inv_start, inv_end = int(sig[1]), int(sig[2])
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=inv_start, tend=inv_end,
            svtype="INV", svlen=inv_end - inv_start, qname=sig[3], mapq=min_mapq, strand=sig[0], source="SA_TAG"
        ))
        sid += 1

    # SA TRA
    for sig in all_sa_tra:
        signatures.append(Signature(
            sid=sid, contig=sig[-1], tstart=int(sig[1]), tend=int(sig[1]) + 1,
            svtype="TRA", svlen=0, qname=sig[4], mapq=min_mapq, source="SA_TAG", sa_contigs=[sig[2]]
        ))
        sid += 1

    LOGGER.info("signature_extraction collected n_signatures=%s", len(signatures))
    return signatures


def save_signatures_to_pickle(
    signatures: List[Signature],
    output_path: str,
    params: Optional[Dict] = None
) -> None:
    """
    Save signatures to a pickle file.

    Args:
        signatures: List of Signature objects
        output_path: Path to output pickle file
        params: Optional dictionary of parameters to save with the signatures
    """
    import pickle

    params_payload = dict(params or {})
    params_payload.setdefault("signature_record_class", "Signature")
    params_payload.setdefault("normalized_fields_version", 1)

    payload = {
        'signatures': signatures,
        'num_signatures': len(signatures),
        'params': params_payload,
        'signature_record_class': "Signature",
        'normalized_fields_version': 1,
    }

    # Count by type and source
    type_counts = {}
    source_counts = {}
    for sig in signatures:
        type_counts[sig.svtype] = type_counts.get(sig.svtype, 0) + 1
        source_counts[sig.source] = source_counts.get(sig.source, 0) + 1

    payload['type_counts'] = type_counts
    payload['source_counts'] = source_counts

    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)

    LOGGER.info("saved signatures path=%s n_signatures=%s", output_path, len(signatures))
    LOGGER.info("signature type counts=%s", type_counts)
    LOGGER.info("signature source counts=%s", source_counts)


# ============================================================
# Command-line interface
# ============================================================

def main():
    """Command-line entry point for signature collection."""
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="GraSV SV signature collection with multiprocessing support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single chromosome
  python -m grasv.signature_extraction --bam data.bam --chrom chr22 --output sigs.pkl

  # Multiple chromosomes (range)
  python -m grasv.signature_extraction --bam data.bam --chrom 1-22 --output sigs.pkl -p 8

  # Multiple chromosomes (list)
  python -m grasv.signature_extraction --bam data.bam --chrom 1,2,3,4,5 --output sigs.pkl

  # Mixed format with multiprocessing
  python -m grasv.signature_extraction --bam data.bam --chrom "1-5,10,chr15-chr20,X,Y" --output sigs.pkl -p 16

Chromosome input formats:
  - Single: "chr22" or "22"
  - Range: "1-5" or "chr1-chr5"
  - List: "1,2,3,4,5" or "chr1,chr2,chr3"
  - Mixed: "1-3,5,chr7-chr9,X,Y"
"""
    )

    # Required arguments
    parser.add_argument("--bam", "-b", required=True,
                        help="Input BAM file (must be indexed)")
    parser.add_argument("--chrom", "-c", required=True,
                        help="Chromosome(s) to process (e.g., '1-22', 'chr1,chr2', '1-5,10,X')")
    parser.add_argument("--output", "-o", required=True,
                        help="Output pickle file path")

    # Multiprocessing options
    parser.add_argument("--processes", "-p", type=int, default=4,
                        help="Number of parallel processes (default: 4)")
    parser.add_argument("--region-size", "-r", type=int, default=5000000,
                        help="Region size for parallel processing in bp (default: 5000000 = 5Mb)")

    # SV detection parameters
    parser.add_argument("--min-sv-size", type=int, default=30,
                        help="Minimum SV size (default: 30)")
    parser.add_argument("--max-sv-size", type=int, default=100000,
                        help="Maximum SV size (default: 100000)")
    parser.add_argument("--min-mapq", type=int, default=20,
                        help="Minimum mapping quality (default: 20)")
    parser.add_argument("--min-read-len", type=int, default=500,
                        help="Minimum read length (default: 500)")
    parser.add_argument("--min-siglength", type=int, default=30,
                        help="Minimum CIGAR signal length (default: 30)")
    parser.add_argument("--merge-del-threshold", type=int, default=0,
                        help="Merge distance for DEL (default: 0)")
    parser.add_argument("--merge-ins-threshold", type=int, default=100,
                        help="Merge distance for INS (default: 100)")
    parser.add_argument("--max-split-parts", type=int, default=7,
                        help="Maximum split alignment parts (default: 7)")

    args = parser.parse_args()

    # Validate input BAM exists
    import os
    if not os.path.exists(args.bam):
        LOGGER.error("BAM file not found: %s", args.bam)
        return 1

    # Check for BAM index
    if not os.path.exists(args.bam + ".bai") and not os.path.exists(args.bam.replace(".bam", ".bai")):
        LOGGER.warning("BAM index not found. Please run: samtools index %s", args.bam)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        LOGGER.info("created output directory: %s", output_dir)

    LOGGER.info("BAM file: %s", args.bam)
    LOGGER.info("chromosomes: %s", args.chrom)
    LOGGER.info("output: %s", args.output)
    LOGGER.info("processes: %s", args.processes)
    LOGGER.info("region size: %.1f Mb", args.region_size / 1000000)

    # Collect signatures
    if args.processes > 1:
        signatures = collect_signatures_parallel(
            bam_path=args.bam,
            chrom_input=args.chrom,
            num_processes=args.processes,
            min_sv_size=args.min_sv_size,
            max_sv_size=args.max_sv_size,
            min_mapq=args.min_mapq,
            min_read_len=args.min_read_len,
            min_siglength=args.min_siglength,
            merge_del_threshold=args.merge_del_threshold,
            merge_ins_threshold=args.merge_ins_threshold,
            max_split_parts=args.max_split_parts,
            region_size=args.region_size
        )
    else:
        signatures = collect_signatures_multi_chrom(
            bam_path=args.bam,
            chrom_input=args.chrom,
            min_sv_size=args.min_sv_size,
            max_sv_size=args.max_sv_size,
            min_mapq=args.min_mapq,
            min_read_len=args.min_read_len,
            min_siglength=args.min_siglength,
            merge_del_threshold=args.merge_del_threshold,
            merge_ins_threshold=args.merge_ins_threshold,
            max_split_parts=args.max_split_parts
        )

    # Save to pickle
    params = {
        'bam': args.bam,
        'chrom': args.chrom,
        'min_sv_size': args.min_sv_size,
        'max_sv_size': args.max_sv_size,
        'min_mapq': args.min_mapq,
        'min_read_len': args.min_read_len,
        'min_siglength': args.min_siglength,
        'merge_del_threshold': args.merge_del_threshold,
        'merge_ins_threshold': args.merge_ins_threshold,
        'max_split_parts': args.max_split_parts,
        'processes': args.processes
    }

    save_signatures_to_pickle(signatures, args.output, params)

    LOGGER.info("done")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

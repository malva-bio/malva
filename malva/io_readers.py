# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

"""
Transparent input adapters for SRA archives and BAM files.

Provides the same chunk-based output as FASTQ processing so the rest of the
indexing pipeline is unaware of the input format.  Performance is maintained by
batch-level Cython functions (process_sra_batch / process_bam_batch) that keep
the inner k-mer extraction loop in C.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np


_BATCH_SIZE = 50_000
_INITIAL_CAP = 50_000_000


def detect_input_type(reads_in: list) -> str:
    """
    Infer the input format from the reads_in list.

    Returns ``'fastq'``, ``'sra'``, or ``'bam'``.
    Two-element lists (and bulk-mode integer first elements) are always FASTQ.
    Single-element lists are dispatched by file extension.
    """
    if len(reads_in) == 2:
        return 'fastq'
    if isinstance(reads_in[0], int):
        return 'fastq'

    ext = os.path.splitext(str(reads_in[0]).lower())[1]
    if ext == '.sra':
        return 'sra'
    if ext == '.bam':
        return 'bam'
    raise ValueError(
        f"Cannot determine input type from '{reads_in[0]}'. "
        "Provide a single .sra or .bam file, or two FASTQ files for R1/R2."
    )


# ---------------------------------------------------------------------------
# SRA
# ---------------------------------------------------------------------------

def process_sra_reads(
    sra_path: str,
    output_dir: str,
    spatial_index,
    kmer_size: int = 24,
    l_prefix: int = 12,
    jump_amount: int = 0,
    trim_start: int = 0,
    trim_end: int = 28,
    chunksize: int = 100_000_000,
    n_report: int = 1_000_000,
    threads: int = 1,
    barcode_segment: Optional[int] = None,
    cdna_segment: Optional[int] = None,
) -> list:
    """
    Process an SRA archive and populate index chunks.

    Uses ``SRAReader.iter_raw_batched()`` (the fastest SRAReader iterator) and
    the Cython ``process_sra_batch`` function for k-mer extraction.

    When *barcode_segment* or *cdna_segment* are ``None``, layout auto-detection
    is performed by sampling the first 1 000 spots.

    Returns a list of chunk file paths (empty when a single radix-build sufficed).
    """
    try:
        from sra_reader import SRAReader, detect_layout
    except ImportError:
        raise ImportError(
            "sra-reader is required for SRA input.\n"
            "  Install with: pip install sra-reader\n"
            "  Or within the malva extras: pip install 'malva[sra]'"
        )

    from malva.fastq_processing import process_sra_batch
    from malva.indexes import (
        write_sorted_chunk_py, build_index_radix_py, _write_empty_index
    )

    if jump_amount == 0:
        jump_amount = kmer_size

    # ------------------------------------------------------------------
    # Auto-detect read layout when segment indices are not explicit
    # ------------------------------------------------------------------
    if barcode_segment is None or cdna_segment is None:
        logging.info("Auto-detecting SRA read layout (sampling first 1 000 spots)…")
        layout = detect_layout(sra_path)
        logging.info("Detected layout:\n%s", layout.describe())
        if barcode_segment is None:
            if layout.barcode_segment is None:
                raise ValueError(
                    "Could not auto-detect the barcode segment from the SRA file. "
                    "Specify it with --sra-barcode-segment."
                )
            barcode_segment = layout.barcode_segment
        if cdna_segment is None:
            if layout.cdna_segment is None:
                raise ValueError(
                    "Could not auto-detect the cDNA segment from the SRA file. "
                    "Specify it with --sra-cdna-segment."
                )
            cdna_segment = layout.cdna_segment

    logging.info(
        "SRA mode: barcode=segment[%d][%d:%d], cDNA=segment[%d]",
        barcode_segment, trim_start, trim_end, cdna_segment,
    )

    chunk_dir = os.path.join(output_dir, '_chunks')
    os.makedirs(chunk_dir, exist_ok=True)

    cap = min(chunksize, _INITIAL_CAP)
    ak = np.empty(cap, dtype=np.uint64)
    ac = np.empty(cap, dtype=np.uint32)
    tp: int = 0
    chunk_num = 0
    total_pairs = 0
    ns = 0
    chunk_paths: list[str] = []
    t0 = time.time()

    reader = SRAReader(sra_path, include_quality=False)
    with reader:
        for segs_batch in reader.iter_raw_batched(batch_size=_BATCH_SIZE):
            ns += len(segs_batch[0])

            # Retry loop handles buffer-full (-1) by doubling capacity
            while True:
                new_tp = process_sra_batch(
                    segs_batch,
                    barcode_segment, cdna_segment,
                    spatial_index,
                    ak, ac, tp,
                    trim_start, trim_end,
                    kmer_size, jump_amount,
                    cap,
                )
                if new_tp == -1:
                    cap *= 2
                    ak, ac = _grow(ak, ac, cap)
                    continue
                tp = new_tp
                break

            if tp >= chunksize:
                cp = os.path.join(chunk_dir, f'chunk_{chunk_num}.bin')
                logging.info(
                    "  Writing chunk %d (%s pairs, total %s)",
                    chunk_num, f"{tp:,}", f"{total_pairs + tp:,}",
                )
                write_sorted_chunk_py(ak, ac, tp, cp)
                chunk_paths.append(cp)
                total_pairs += tp
                chunk_num += 1
                tp = 0
                t0 = time.time()

            if ns % n_report < _BATCH_SIZE:
                dt = time.time() - t0
                if dt > 0:
                    logging.info(
                        "  Read %s spots, %s pairs (%.0f spots/s)",
                        f"{ns:,}", f"{total_pairs + tp:,}", n_report / dt,
                    )
                t0 = time.time()

    logging.info(
        "SRA: %s spots, %s pairs, %d chunks",
        f"{ns:,}", f"{total_pairs + tp:,}", chunk_num,
    )

    return _finalise(ak, ac, tp, chunk_num, chunk_paths, chunk_dir,
                     output_dir, kmer_size, l_prefix)


# ---------------------------------------------------------------------------
# BAM
# ---------------------------------------------------------------------------

def process_bam_reads(
    bam_path: str,
    output_dir: str,
    spatial_index,
    kmer_size: int = 24,
    l_prefix: int = 12,
    jump_amount: int = 0,
    chunksize: int = 100_000_000,
    n_report: int = 1_000_000,
    threads: int = 1,
    barcode_tag: str = 'CB',
    sequence_tag: Optional[str] = None,
) -> list:
    """
    Process a BAM file and populate index chunks.

    Args:
        barcode_tag: BAM tag carrying the cell barcode (default ``'CB'``).
                     Common alternatives: ``'CR'`` (raw barcode), ``'XC'``.
        sequence_tag: BAM tag for the cDNA sequence.  When ``None`` (the
                      default), the main ``SEQ`` field is used.

    Returns a list of chunk file paths (empty when a single radix-build sufficed).
    """
    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is required for BAM input.\n"
            "  Install with: pip install pysam\n"
            "  Or via conda: conda install -c bioconda pysam\n"
            "  Or within the malva extras: pip install 'malva[bam]'"
        )

    from malva.fastq_processing import process_bam_batch
    from malva.indexes import (
        write_sorted_chunk_py, build_index_radix_py, _write_empty_index
    )

    if jump_amount == 0:
        jump_amount = kmer_size

    logging.info(
        "BAM mode: barcode tag='%s', sequence='%s'",
        barcode_tag,
        "SEQ field" if sequence_tag is None else f"tag={sequence_tag!r}",
    )

    chunk_dir = os.path.join(output_dir, '_chunks')
    os.makedirs(chunk_dir, exist_ok=True)

    cap = min(chunksize, _INITIAL_CAP)
    ak = np.empty(cap, dtype=np.uint64)
    ac = np.empty(cap, dtype=np.uint32)
    tp: int = 0
    chunk_num = 0
    total_pairs = 0
    ns = 0
    chunk_paths: list[str] = []
    t0 = time.time()

    bc_batch: list = []
    seq_batch: list = []

    def _flush_bam_batch():
        nonlocal tp, cap, ak, ac
        while True:
            new_tp = process_bam_batch(
                bc_batch, seq_batch,
                spatial_index,
                ak, ac, tp,
                kmer_size, jump_amount,
                cap,
            )
            if new_tp == -1:
                cap *= 2
                ak, ac = _grow(ak, ac, cap)
                continue
            tp = new_tp
            break
        bc_batch.clear()
        seq_batch.clear()

    with pysam.AlignmentFile(bam_path, 'rb', threads=threads) as bam:
        for read in bam.fetch(until_eof=True):
            ns += 1

            try:
                bc = read.get_tag(barcode_tag)
            except KeyError:
                bc = None

            if sequence_tag is not None:
                try:
                    seq = read.get_tag(sequence_tag)
                except KeyError:
                    seq = None
            else:
                seq = read.query_sequence  # None for reads with no SEQ

            bc_batch.append(bc)
            seq_batch.append(seq)

            if len(bc_batch) >= _BATCH_SIZE:
                _flush_bam_batch()

                if tp >= chunksize:
                    cp = os.path.join(chunk_dir, f'chunk_{chunk_num}.bin')
                    logging.info(
                        "  Writing chunk %d (%s pairs, total %s)",
                        chunk_num, f"{tp:,}", f"{total_pairs + tp:,}",
                    )
                    write_sorted_chunk_py(ak, ac, tp, cp)
                    chunk_paths.append(cp)
                    total_pairs += tp
                    chunk_num += 1
                    tp = 0
                    t0 = time.time()

                if ns % n_report < _BATCH_SIZE:
                    dt = time.time() - t0
                    if dt > 0:
                        logging.info(
                            "  Read %s reads, %s pairs (%.0f reads/s)",
                            f"{ns:,}", f"{total_pairs + tp:,}", n_report / dt,
                        )
                    t0 = time.time()

        if bc_batch:
            _flush_bam_batch()

    logging.info(
        "BAM: %s reads, %s pairs, %d chunks",
        f"{ns:,}", f"{total_pairs + tp:,}", chunk_num,
    )

    return _finalise(ak, ac, tp, chunk_num, chunk_paths, chunk_dir,
                     output_dir, kmer_size, l_prefix)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _grow(ak: np.ndarray, ac: np.ndarray, new_cap: int):
    """Double-and-copy numpy accumulation buffers."""
    ak_new = np.empty(new_cap, dtype=np.uint64)
    ac_new = np.empty(new_cap, dtype=np.uint32)
    ak_new[:len(ak)] = ak
    ac_new[:len(ac)] = ac
    return ak_new, ac_new


def _finalise(ak, ac, tp, chunk_num, chunk_paths, chunk_dir,
              output_dir, kmer_size, l_prefix):
    """Handle the remaining data after the read loop and return chunk paths."""
    from malva.indexes import (
        write_sorted_chunk_py, build_index_radix_py, _write_empty_index
    )

    if tp == 0 and chunk_num == 0:
        _write_empty_index(output_dir, kmer_size, l_prefix,
                           kmer_size - l_prefix, 0)
        return []

    if chunk_num == 0:
        logging.info("Single chunk (%s pairs) — using radix build", f"{tp:,}")
        build_index_radix_py(ak, ac, tp, output_dir, kmer_size, l_prefix)
        return []

    if tp > 0:
        cp = os.path.join(chunk_dir, f'chunk_{chunk_num}.bin')
        logging.info("  Writing final chunk %d (%s pairs)", chunk_num, f"{tp:,}")
        write_sorted_chunk_py(ak, ac, tp, cp)
        chunk_paths.append(cp)

    return chunk_paths

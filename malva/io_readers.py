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

import json
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

    path = str(reads_in[0]).lower()
    # Use endswith rather than splitext so .sralite is matched correctly
    # (splitext('.sralite') → ('.sralite',) but that still works — just
    # be explicit to avoid any ambiguity).
    if path.endswith('.sra') or path.endswith('.sralite'):
        return 'sra'
    if path.endswith('.bam'):
        return 'bam'
    raise ValueError(
        f"Cannot determine input type from '{reads_in[0]}'. "
        "Provide a single .sra / .sralite or .bam file, or two FASTQ files for R1/R2."
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
    extra_meta: Optional[dict] = None,
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
        from sra_reader import SRAReader
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

    if barcode_segment is None or cdna_segment is None:
        raise ValueError(
            "barcode_segment and cdna_segment must be specified for SRA input. "
            "Pass --sra-barcode-segment and --sra-cdna-segment on the command line, "
            "or use generate_manifest.py to detect them automatically."
        )

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

    n_cells = spatial_index.num_items() if spatial_index is not None else 0
    return _finalise(ak, ac, tp, chunk_num, chunk_paths, chunk_dir,
                     output_dir, kmer_size, l_prefix,
                     n_cells=n_cells, extra_meta=extra_meta)


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
    extra_meta: Optional[dict] = None,
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

    # ------------------------------------------------------------------
    # Pre-scan first reads to verify barcode tag is present
    # ------------------------------------------------------------------
    _N_PROBE = 200
    _probe_bc_missing = 0
    _probe_tags: dict[str, int] = {}
    with pysam.AlignmentFile(bam_path, 'rb', threads=threads) as _bam_probe:
        for _i, _read in enumerate(_bam_probe.fetch(until_eof=True)):
            if _i >= _N_PROBE:
                break
            try:
                _read.get_tag(barcode_tag)
            except KeyError:
                _probe_bc_missing += 1
            for _tag, _val in _read.get_tags():
                _probe_tags[_tag] = _probe_tags.get(_tag, 0) + 1

    if _probe_tags:
        _probe_total = min(_N_PROBE, max(_probe_tags.values()))
        _bc_miss_rate = _probe_bc_missing / max(_probe_total, 1)
        if _bc_miss_rate > 0.5:
            _present = sorted(_probe_tags, key=lambda t: -_probe_tags[t])
            logging.warning(
                "BAM: barcode tag '%s' absent in %.0f%% of first %d reads. "
                "Tags present (by frequency): %s. "
                "Use --bam-barcode-tag to specify the correct tag.",
                barcode_tag, _bc_miss_rate * 100, _probe_total,
                ", ".join(f"'{t}' ({_probe_tags[t]}x)" for t in _present[:10]),
            )
        else:
            logging.debug(
                "BAM probe (%d reads): tag '%s' found in %.0f%% of reads.",
                _probe_total, barcode_tag, (1 - _bc_miss_rate) * 100,
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
    ns_bc_missing: int = 0

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
                ns_bc_missing += 1

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
    if ns > 0 and ns_bc_missing / ns > 0.5:
        logging.warning(
            "BAM: tag '%s' was missing in %s/%s reads (%.0f%%). "
            "The index will likely be empty. "
            "Use --bam-barcode-tag to specify the correct tag.",
            barcode_tag, f"{ns_bc_missing:,}", f"{ns:,}",
            ns_bc_missing / ns * 100,
        )

    n_cells = spatial_index.num_items() if spatial_index is not None else 0
    return _finalise(ak, ac, tp, chunk_num, chunk_paths, chunk_dir,
                     output_dir, kmer_size, l_prefix,
                     n_cells=n_cells, extra_meta=extra_meta)


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


def _patch_meta(output_dir: str, n_cells: int, extra: dict) -> None:
    """Merge correct values into the meta.json written by the Cython build.

    Cython build functions (_build_index_radix, build_from_sorted_chunks) write
    a minimal 9-field meta.json with n_cells=0 and no index-state fields
    (merge_projects, project_id_shift, cell_id_mask, …).  This helper reads
    that file, overwrites n_cells with the real value, merges in any extra
    fields, and writes it back — preserving magic, version, n_kmers, etc.
    """
    meta_path = os.path.join(output_dir, 'meta.json')
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        meta['n_cells'] = int(n_cells)
        meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Could not patch meta.json: %s", exc)


def _finalise(ak, ac, tp, chunk_num, chunk_paths, chunk_dir,
              output_dir, kmer_size, l_prefix,
              n_cells: int = 0, extra_meta: Optional[dict] = None):
    """Handle the remaining data after the read loop and return chunk paths."""
    from malva.indexes import (
        write_sorted_chunk_py, build_index_radix_py, _write_empty_index
    )

    extra = extra_meta or {}

    if tp == 0 and chunk_num == 0:
        logging.warning(
            "No k-mer pairs were extracted — the index will be empty (n_cells=0). "
            "Likely causes: "
            "(1) wrong barcode segment/tag — check --sra-barcode-segment or --bam-barcode-tag; "
            "(2) all barcodes are absent from the spatial barcode index; "
            "(3) the file contains no reads with a valid barcode+cDNA pair."
        )
        _write_empty_index(output_dir, kmer_size, l_prefix,
                           kmer_size - l_prefix, 0)
        return []

    if chunk_num == 0:
        logging.info("Single chunk (%s pairs) — using radix build", f"{tp:,}")
        build_index_radix_py(ak, ac, tp, output_dir, kmer_size, l_prefix)
        # build_index_radix_py always writes n_cells=0 and omits index-state
        # fields — patch meta.json immediately with the correct values.
        _patch_meta(output_dir, n_cells, extra)
        return []

    if tp > 0:
        cp = os.path.join(chunk_dir, f'chunk_{chunk_num}.bin')
        logging.info("  Writing final chunk %d (%s pairs)", chunk_num, f"{tp:,}")
        write_sorted_chunk_py(ak, ac, tp, cp)
        chunk_paths.append(cp)

    return chunk_paths

import json
import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_malva_to_prefix(malva_index_dir, output_dir, l_prefix=12,
                            chunk_size=5_000_000, verbose=True):
    """Convert a legacy HDF5 index to the current prefix-bucketed format.

    .. deprecated::
        This function requires the legacy HDF5-based MalvaIndex which is no longer
        part of the package. It is kept for historical reference only.
    """
    try:
        from malva._legacy_index import MalvaIndex  # no longer distributed
    except ImportError:
        raise ImportError(
            "The legacy HDF5-based MalvaIndex is no longer part of Malva. "
            "If you need to convert an old index, check out the git history "
            "of the 'prefix_bucketed_index' branch for the original convert.py."
        )
    from malva.indexes import convert_h5_to_prefix

    logger = logging.getLogger("convert_to_prefix")
    logger.info(f"Opening existing index at {malva_index_dir}")
    mindex = MalvaIndex(malva_index_dir)
    mindex.open(mode='r')

    kmer_size = mindex.kmer_size
    n_spatial = mindex.n_spatial
    logger.info(f"  kmer_size={kmer_size}, n_spatial={n_spatial:,}")

    chunk_id = 0
    indices_ds = mindex.index[f'index_{chunk_id}_indices']
    indptr_ds = mindex.index[f'index_{chunk_id}_indptr']
    data_ds = mindex.index[f'index_{chunk_id}_data']

    # Delegate to Cython for fast conversion
    n_kmers, n_buckets = convert_h5_to_prefix(
        indices_ds, indptr_ds, data_ds,
        output_dir, kmer_size=kmer_size, l_prefix=l_prefix,
        n_cells=n_spatial, chunk_size=chunk_size, verbose=verbose,
    )

    # Copy spatial coords
    if 'spatial_coord' in mindex.index:
        np.save(os.path.join(output_dir, 'spatial_coord.npy'),
                np.array(mindex.index['spatial_coord'], dtype=np.float32))

    # Update metadata with project info
    meta_path = os.path.join(output_dir, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    for attr, key in [('project_mapping', 'project_mapping'),
                       ('has_merged_projects', 'merge_projects'),
                       ('project_id_shift', 'project_id_shift'),
                       ('cell_id_mask', 'cell_id_mask')]:
        if attr in mindex.index.attrs:
            v = mindex.index.attrs[attr]
            if attr == 'project_mapping': v = json.loads(v)
            elif attr == 'has_merged_projects': v = bool(v)
            else: v = int(v)
            meta[key] = v
    if hasattr(mindex, 'coord_lims') and mindex.coord_lims:
        meta['coord_lims'] = list(mindex.coord_lims)

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    mindex.close()


def verify_indices(old_index_dir, new_index_dir, n_sample=100000, seed=42, verbose=True):
    """Compare a legacy HDF5 index against a converted prefix-bucketed index.

    .. deprecated::
        Requires the legacy HDF5-based MalvaIndex which is no longer distributed.
    """
    try:
        from malva._legacy_index import MalvaIndex  # no longer distributed
    except ImportError:
        raise ImportError(
            "The legacy HDF5-based MalvaIndex is no longer part of Malva. "
            "See convert_malva_to_prefix() for details."
        )
    from malva.indexes import PrefixIndex

    logger = logging.getLogger("verify_indices")
    logger.info(f"Opening old index: {old_index_dir}")
    old = MalvaIndex(old_index_dir)
    old.open(mode='r')

    chunk_id = 0
    indices_ds = old.index[f'index_{chunk_id}_indices']
    indptr_ds = old.index[f'index_{chunk_id}_indptr']
    data_ds = old.index[f'index_{chunk_id}_data']
    total_indices = len(indices_ds)
    total_indptr = len(indptr_ds)
    total_data = len(data_ds)
    logger.info(f"  Old: {total_indices:,} kmers")

    logger.info(f"Opening new index: {new_index_dir}")
    new_idx = PrefixIndex()
    new_idx.open(new_index_dir)

    rng = np.random.RandomState(seed)
    if n_sample <= 0 or n_sample >= total_indices:
        sample_idx = np.arange(total_indices)
    else:
        sample_idx = np.sort(rng.choice(total_indices, size=n_sample, replace=False))
    logger.info(f"Verifying {len(sample_idx):,} kmers")

    n_ok = n_mismatch = n_missing_new = n_extra_new = n_checked = 0
    batch_size = min(50000, len(sample_idx))
    t0 = time.time()
    last_report = t0

    for batch_start in range(0, len(sample_idx), batch_size):
        batch_end = min(batch_start + batch_size, len(sample_idx))
        bidx = sample_idx[batch_start:batch_end]

        idx_min, idx_max = int(bidx[0]), int(bidx[-1])
        range_indices = np.asarray(indices_ds[idx_min:idx_max + 1], dtype=np.uint64)
        ip_max = min(idx_max + 2, total_indptr)
        range_indptr = np.asarray(indptr_ds[idx_min:ip_max], dtype=np.uint64)

        batch_kmers = range_indices[bidx - idx_min]
        new_results = new_idx.query(batch_kmers, count_at_most=999999999, count_at_least=0)

        for j, idx in enumerate(bidx):
            local = idx - idx_min
            kmer_val = int(range_indices[local])
            ip_s = int(range_indptr[local])
            ip_e = int(range_indptr[local + 1]) if local + 1 < len(range_indptr) else total_data
            old_cells = np.unique(np.asarray(data_ds[ip_s:ip_e], dtype=np.uint32)) if ip_e > ip_s else np.array([], dtype=np.uint32)
            new_cells = np.sort(new_results[kmer_val]) if kmer_val in new_results else np.array([], dtype=np.uint32)

            n_checked += 1
            if np.array_equal(old_cells, new_cells):
                n_ok += 1
            elif len(new_cells) == 0 and len(old_cells) > 0:
                n_missing_new += 1
                if n_missing_new <= 3:
                    logger.warning(f"  MISSING: kmer={kmer_val:#x} old={len(old_cells)} cells")
            elif len(old_cells) == 0 and len(new_cells) > 0:
                n_extra_new += 1
                if n_extra_new <= 3:
                    logger.warning(f"  EXTRA: kmer={kmer_val:#x} new={len(new_cells)} cells")
            else:
                n_mismatch += 1
                if n_mismatch <= 3:
                    o = set(old_cells.tolist()); n = set(new_cells.tolist())
                    logger.warning(f"  MISMATCH: kmer={kmer_val:#x} "
                                   f"old={len(old_cells)} new={len(new_cells)} "
                                   f"only_old={len(o-n)} only_new={len(n-o)}")

        now = time.time()
        if verbose and (now - last_report > 3.0 or batch_end >= len(sample_idx)):
            dt = now - t0
            rate = n_checked / dt if dt > 0 else 0
            logger.info(f"  {n_checked:,}/{len(sample_idx):,} "
                        f"ok={n_ok} mis={n_mismatch} miss={n_missing_new} extra={n_extra_new} "
                        f"({rate:.0f}/s)")
            last_report = now

    old.close()
    new_idx.close()
    logger.info(f"\nResult: {n_ok:,} OK, {n_mismatch:,} mismatch, "
                f"{n_missing_new:,} missing, {n_extra_new:,} extra out of {n_checked:,}")
    if n_mismatch == 0 and n_missing_new == 0 and n_extra_new == 0:
        logger.info("PASS — indices are identical")
        return True
    else:
        logger.error("FAIL — indices differ")
        return False


def _run_convert(args):
    if args.command == 'convert':
        convert_malva_to_prefix(args.input, args.output, l_prefix=args.l_prefix, chunk_size=args.chunk_size)
    elif args.command == 'verify':
        verify_indices(args.old, args.new, n_sample=args.n_sample, seed=args.seed)

if __name__ == "__main__":
    from malva.cli import get_convert_parser
    args = get_convert_parser().parse_args()
    _run_convert(args)

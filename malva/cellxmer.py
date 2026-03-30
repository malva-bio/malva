# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

import logging
import os
import tempfile
import shutil
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from malva.utils import check_directory_exists, SUCCESS_MSG
from malva.kmer_processing import decode_kmer


def _get_csr_arrays(kmer_index, verbose=True):
    """
    Extract CSR-compatible arrays (indices, indptr, data) from a MalvaIndex.

    Returns:
        (indices, indptr, data, num_cells)
    """
    kmer_index._ensure_index_open()
    pi = kmer_index._prefix_index
    indices, indptr, data = pi.iterate_all_kmers(verbose=verbose)
    num_cells = kmer_index.n_spatial
    return indices, indptr.astype(np.int64), data, num_cells


def _get_csr_arrays_chunk(kmer_index, chunk_start, chunk_end, verbose=True):
    """
    Extract CSR arrays for a slice of k-mers [chunk_start, chunk_end).

    Returns:
        (indices_chunk, indptr_chunk, data_chunk, num_cells, total_kmers)
    """
    kmer_index._ensure_index_open()
    pi = kmer_index._prefix_index
    all_indices, all_indptr, all_data = pi.iterate_all_kmers(verbose=verbose)
    num_cells = kmer_index.n_spatial
    total_kmers = len(all_indices)

    actual_end = min(chunk_end, total_kmers)
    indices_chunk = all_indices[chunk_start:actual_end]
    ip_start = int(all_indptr[chunk_start])
    ip_end = int(all_indptr[actual_end])
    data_chunk = all_data[ip_start:ip_end]
    indptr_chunk = all_indptr[chunk_start:actual_end + 1] - ip_start

    return indices_chunk, indptr_chunk.astype(np.int64), data_chunk, num_cells, total_kmers


def malva_to_cellxmer(
    kmer_index,
    count_at_most: int = 10_000,
    count_at_least: int = 10,
    verbose=True
):
    if verbose:
        logging.info(f"Opening malva index and loading k-mer information")

    indices, indptr, data, num_cells = _get_csr_arrays(kmer_index, verbose=verbose)

    diff_counts = np.diff(indptr)
    _diff_counts_idx = diff_counts > -1

    n_kmer_filter = diff_counts[_diff_counts_idx].shape
    interesting_kmers = indices[np.append(_diff_counts_idx, np.array([False]))]

    if verbose:
        logging.debug(f"There are {n_kmer_filter[0]:,} {kmer_index.kmer_size}-mers with "
                      f"{count_at_least:,} < counts < {count_at_most:,}")
        logging.debug(f"Creating sparse matrix")

    adata_X_or = csr_matrix(
        (np.ones_like(data), data, indptr),
        shape=(n_kmer_filter[0], num_cells)
    )
    data_Ts = adata_X_or[_diff_counts_idx].indices
    indptr_Ts = adata_X_or[_diff_counts_idx].indptr

    adata_X_tr = csr_matrix(
        (np.ones_like(data_Ts), data_Ts, indptr_Ts),
        shape=(n_kmer_filter[0], num_cells)
    ).T
    adata_X_tr = adata_X_tr + adata_X_tr
    adata_X_tr = (adata_X_tr * 0.5).astype(np.uint32)

    if verbose:
        logging.debug(f"Creating AnnData object from cell-by-kmer sparse matrix")

    adata = ad.AnnData(X=adata_X_tr.tocsr())
    adata.var_names = [decode_kmer(v, int(kmer_index.kmer_size)) for v in interesting_kmers]

    coord_path = os.path.join(kmer_index.index_dir, 'spatial_coord.npy')
    if os.path.exists(coord_path):
        adata.obsm['spatial'] = np.load(coord_path)

    return adata


def malva_to_filtered_cellxmer_chunked(
    kmer_index,
    count_at_most=10_000,
    count_at_least=10,
    k_size=None,
    w_size=16,
    num_buckets=100000,
    chunk_size=10_000_000,
    temp_dir=None,
    verbose=True
):
    """
    Process a MalvaIndex to create a filtered cell-by-bucket matrix in chunks.
    """
    from malva.filter_minimizers import KmerFilter

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="cellxmer_")
        delete_temp = True
    else:
        temp_dir = tempfile.mkdtemp(prefix="cellxmer_", dir=temp_dir)
        delete_temp = False

    if verbose:
        logging.debug(f"Using temporary directory: {temp_dir}")
        logging.debug(f"Opening malva index and loading k-mer information")

    if k_size is None:
        k_size = int(kmer_index.kmer_size)

    kmer_filter = KmerFilter(k_size, w_size, num_buckets)

    kmer_index._ensure_index_open()
    import json
    meta_path = os.path.join(kmer_index.index_dir, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    len_indices = meta.get('n_kmers', 0)
    num_cells = kmer_index.n_spatial

    if verbose:
        logging.info(f"Reading all {len_indices:,} kmers from index...")
    pi = kmer_index._prefix_index
    all_indices, all_indptr, all_data = pi.iterate_all_kmers(verbose=verbose)
    len_indices = len(all_indices)

    total_chunks = (len_indices + chunk_size - 1) // chunk_size
    chunk_files = []

    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len_indices)

        if verbose:
            logging.info(f"Processing chunk {chunk_idx+1}/{total_chunks}: "
                         f"k-mers {chunk_start:,} to {chunk_end:,}")

        indices_c = all_indices[chunk_start:chunk_end]
        ip_s = int(all_indptr[chunk_start])
        ip_e = int(all_indptr[chunk_end])
        data_c = all_data[ip_s:ip_e]
        indptr_c = (all_indptr[chunk_start:chunk_end + 1] - ip_s).astype(np.int64)

        adata_X_or = csr_matrix(
            (np.ones_like(data_c), data_c, indptr_c),
            shape=(len(indptr_c) - 1, num_cells)
        ).T

        bucket_assignments = kmer_filter.filter_stream(indices_c)

        kmer_to_bucket = csr_matrix(
            (np.ones(len(bucket_assignments), dtype=np.int8),
             (np.arange(len(bucket_assignments)), bucket_assignments)),
            shape=(len(bucket_assignments), num_buckets)
        )

        cell_by_bucket_chunk = adata_X_or.dot(kmer_to_bucket)

        chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx}.npz")
        save_npz(chunk_file, cell_by_bucket_chunk)
        chunk_files.append(chunk_file)

    if verbose:
        logging.info(f"Merging {len(chunk_files)} chunks into final cell-by-bucket matrix")

    merged_cell_by_bucket = load_npz(chunk_files[0])
    for i, chunk_file in enumerate(chunk_files[1:], 1):
        if verbose and i % 10 == 0:
            logging.info(f"Merging chunk {i}/{len(chunk_files)-1}")
        merged_cell_by_bucket = merged_cell_by_bucket + load_npz(chunk_file)

    if verbose:
        logging.info(f"Creating AnnData object from merged cell-by-bucket sparse matrix")

    adata = ad.AnnData(X=merged_cell_by_bucket.tocsr())
    adata.var_names = [f"bucket_{i}" for i in range(num_buckets)]

    coord_path = os.path.join(kmer_index.index_dir, 'spatial_coord.npy')
    if os.path.exists(coord_path):
        adata.obsm['spatial'] = np.load(coord_path)

    if delete_temp:
        if verbose:
            logging.debug(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

    return adata


def _run_cellxmer(args):
    from malva.index import MalvaIndex
    from malva.spacemake import create_meshed_adata

    kmer_index = MalvaIndex(args.index_in)
    check_directory_exists(args.h5ad_out, except_when=False)

    logging.info(f"Converting malva index to filtered cell-by-kmer object with chunked processing")

    adata_cellxmer = malva_to_filtered_cellxmer_chunked(
        kmer_index,
        count_at_most=args.kmer_max,
        count_at_least=args.kmer_min,
        k_size=None,
        w_size=args.w_size,
        num_buckets=args.num_buckets,
        chunk_size=args.chunk_size,
        temp_dir=args.tmp_dir,
        verbose=True
    )

    output_path = os.path.join(args.h5ad_out,
                               f'cellxmer_filtered_w{args.w_size}_b{args.num_buckets}.h5ad')
    adata_cellxmer.write_h5ad(output_path)

    if args.bin_size > 0:
        logging.info(f"Meshing AnnData into {args.bin_size} spatial unit-side hexagons")
        mesh_adata_cellxmer = create_meshed_adata(
            adata_cellxmer, 1,
            spot_diameter_um=args.bin_size,
            spot_distance_um=args.bin_size,
            bead_diameter_um=args.bin_size,
            mesh_type="hexagon"
        )
        mesh_output_path = os.path.join(
            args.h5ad_out,
            f'cellxmer_filtered_w{args.w_size}_b{args.num_buckets}_bin{args.bin_size}.h5ad'
        )
        mesh_adata_cellxmer.write_h5ad(mesh_output_path)

    if args.save_kmer:
        _run_cellxmer_non_chunked(args)

    logging.info(f"Final filtered AnnData object was stored at {args.h5ad_out}")
    logging.info(SUCCESS_MSG)


def _run_cellxmer_non_chunked(args):
    from malva.index import MalvaIndex
    from malva.spacemake import create_meshed_adata

    kmer_index = MalvaIndex(args.index_in)
    check_directory_exists(args.h5ad_out, except_when=False)

    logging.info(f"Converting malva index to cell-by-kmer object")
    adata_cellxmer = malva_to_cellxmer(
        kmer_index,
        count_at_most=args.kmer_max,
        count_at_least=args.kmer_min
    )

    adata_cellxmer.write_h5ad(os.path.join(args.h5ad_out, 'cellxmer.h5ad'))
    if args.bin_size > 0:
        logging.info(f"Meshing AnnData into {args.bin_size} spatial unit-side hexagons")
        mesh_adata_cellxmer = create_meshed_adata(
            adata_cellxmer, 1,
            spot_diameter_um=args.bin_size,
            spot_distance_um=args.bin_size,
            bead_diameter_um=args.bin_size,
            mesh_type="hexagon"
        )
        mesh_adata_cellxmer.write_h5ad(
            os.path.join(args.h5ad_out, f'cellxmer_bin{args.bin_size}.h5ad')
        )


if __name__ == "__main__":
    from malva.cli import get_cellxmer_parser
    args = get_cellxmer_parser().parse_args()
    _run_cellxmer(args)

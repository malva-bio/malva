import logging
import os
import tempfile
import shutil
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from malva.index import MalvaIndex
from malva.spacemake import create_meshed_adata
from malva.utils import check_directory_exists
from malva.filter_minimizers import KmerFilter
from malva.kmer_processing import decode_kmer

def malva_to_cellxmer(
    kmer_index,
    count_at_most: int = 10_000,
    count_at_least: int = 10,
    verbose=True
):
    if verbose:
        logging.info(f"Opening malva index and loading k-mer information")

    kmer_index.open(mode='r')
    indices, data, indptr = kmer_index.index['index_0_indices'][:], kmer_index.index['index_0_data'][:], kmer_index.index['index_0_indptr'][:]
    diff_counts = np.diff(indptr)
    _diff_counts_idx = diff_counts > -1 #(diff_counts>count_at_least) & (diff_counts < count_at_most)

    if 'spatial_coord' not in kmer_index.index:
        num_cells = kmer_index.n_spatial + 1 # so we don't have out of bounds issues...
    else:
        # otherwise, we use the number of spatial coordinates
        num_cells = kmer_index.n_spatial # so we don't have out of bounds issues...

    n_kmer_filter = diff_counts[_diff_counts_idx].shape
    interesting_kmers = indices[np.append(_diff_counts_idx, np.array([False]))]
    kmer_index.close()
    
    if verbose:
        logging.info(f"There are {n_kmer_filter[0]:,} {kmer_index.kmer_size}-mers with {count_at_least:,} < counts < {count_at_most:,}")

    kmer_index.open(mode='r')
    
    if verbose:
        logging.info(f"Creating sparse matrix")

    adata_X_or = csr_matrix((np.ones_like(data), data, indptr), shape=(n_kmer_filter[0], num_cells))
    data_Ts = adata_X_or[_diff_counts_idx].indices
    indptr_Ts = adata_X_or[_diff_counts_idx].indptr

    adata_X_tr = csr_matrix((np.ones_like(data_Ts), data_Ts, indptr_Ts), shape=(n_kmer_filter[0], num_cells)).T
    # we need to apply this to keep the common items between matrices, then rescale to unit
    adata_X_tr = adata_X_tr + adata_X_tr
    adata_X_tr = (adata_X_tr * 0.5).astype(np.uint32)

    if verbose:
        logging.info(f"Creating AnnData object from cell-by-kmer sparse matrix")
    
    adata = ad.AnnData(X=adata_X_tr.tocsr())
    adata.var_names = [decode_kmer(v, int(kmer_index.kmer_size)) for v in interesting_kmers]

    kmer_index.open(mode='r')
    if 'spatial_coord' in kmer_index.index:
        adata.obsm['spatial'] = kmer_index.spatial_coord[:]
    kmer_index.close()

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
    Process a malva index to create a filtered cell-by-bucket matrix in chunks
    following the same approach as the original implementation but with chunked processing.
    """
    # Create a temporary directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="cellxmer_")
        delete_temp = True
    else:
        temp_dir = tempfile.mkdtemp(prefix="cellxmer_", dir=temp_dir)
        delete_temp = False
    
    if verbose:
        logging.info(f"Using temporary directory: {temp_dir}")
        logging.info(f"Opening malva index and loading k-mer information")
    
    # First pass: Identify k-mers that meet the count criteria
    kmer_index.open(mode='r')
    
    if k_size is None:
        k_size = int(kmer_index.kmer_size)
    
    kmer_filter = KmerFilter(k_size, w_size, num_buckets)

    len_indices = kmer_index.index['index_0_indices'].shape[0]
    if 'spatial_coord' not in kmer_index.index:
        num_cells = kmer_index.n_spatial + 1 # so we don't have out of bounds issues...
    else:
        # otherwise, we use the number of spatial coordinates
        num_cells = kmer_index.n_spatial # so we don't have out of bounds issues...
    
    total_chunks =  (len_indices + chunk_size - 1) // chunk_size
    chunk_files = []
    
    # Process in chunks
    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len_indices)
        
        if verbose:
            logging.info(f"Processing chunk {chunk_idx+1}/{total_chunks}: k-mers {chunk_start:,} to {chunk_end:,}")
        
        # Process this chunk following the original algorithm
        kmer_index.open(mode='r')
        
        # Load necessary data for this chunk
        indices = kmer_index.index['index_0_indices'][chunk_start:chunk_end]
        indptr = kmer_index.index['index_0_indptr'][chunk_start:chunk_end]
        if chunk_end == len_indices:
            indptr = np.concatenate([indptr, np.array([len(kmer_index.index['index_0_data'])])]).astype(int)
        else:
            indptr = np.concatenate([indptr, np.array([kmer_index.index['index_0_indptr'][chunk_end+1]])]).astype(int)

        data = kmer_index.index['index_0_data'][indptr[0]:indptr[-1]]
        indptr -= indptr[0]
        kmer_index.close()


        adata_X_or = csr_matrix((np.ones_like(data), data, indptr), shape=(len(indptr) - 1, num_cells)).T

        # Get bucket assignments for the k-mers in this chunk
        # We need to get the correct k-mer values for bucket assignment
        bucket_assignments = kmer_filter.filter_stream(indices)
        
        # Create mapper from k-mers to buckets
        kmer_to_bucket = csr_matrix(
            (np.ones(len(bucket_assignments), dtype=np.int8),
             (np.arange(len(bucket_assignments)), bucket_assignments)),
            shape=(len(bucket_assignments), num_buckets)
        )
        
        # Calculate cell-by-bucket matrix for this chunk
        cell_by_bucket_chunk = adata_X_or.dot(kmer_to_bucket)
        
        # We don't need to make binary, defeats the purpose - we need a spectrum!
        # cell_by_bucket_chunk.data = np.ones_like(cell_by_bucket_chunk.data, dtype=np.int8)
        
        # Save this chunk to a temporary file
        chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx}.npz")
        save_npz(chunk_file, cell_by_bucket_chunk)
        chunk_files.append(chunk_file)
    
    # Merge all chunks
    if verbose:
        logging.info(f"Merging {len(chunk_files)} chunks into final cell-by-bucket matrix")
    
    # Initialize with the first chunk
    merged_cell_by_bucket = load_npz(chunk_files[0])
    
    # Add subsequent chunks
    for i, chunk_file in enumerate(chunk_files[1:], 1):
        if verbose and i % 10 == 0:
            logging.info(f"Merging chunk {i}/{len(chunk_files)-1}")
        
        next_chunk = load_npz(chunk_file)
        
        # Add matrices, but no binary values - again, would defeat the whole purpose
        merged_cell_by_bucket = merged_cell_by_bucket + next_chunk
        # merged_cell_by_bucket.data = np.ones_like(merged_cell_by_bucket.data, dtype=np.int8)
    
    # Create AnnData object
    if verbose:
        logging.info(f"Creating AnnData object from merged cell-by-bucket sparse matrix")
    
    adata = ad.AnnData(X=merged_cell_by_bucket.tocsr())
    
    # Set bucket names as variable names
    adata.var_names = [f"bucket_{i}" for i in range(num_buckets)]
    
    # Add spatial coordinates if available
    kmer_index.open(mode='r')
    if 'spatial_coord' in kmer_index.index:
        adata.obsm['spatial'] = kmer_index.spatial_coord[:]
    kmer_index.close()
    
    # Clean up temporary files
    if delete_temp:
        if verbose:
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    
    return adata


def _run_cellxmer(args):
    """
    Run the cell-by-kmer conversion with k-mer filtering and chunked processing.
    """
    kmer_index = MalvaIndex(args.index_in)
    check_directory_exists(args.h5ad_out, except_when=False)
    
    logging.info(f"Converting malva index to filtered cell-by-kmer object with chunked processing")
    
    adata_cellxmer = malva_to_filtered_cellxmer_chunked(
        kmer_index,
        count_at_most=args.kmer_max,
        count_at_least=args.kmer_min,
        k_size=None,  # Use size from index
        w_size=args.w_size,
        num_buckets=args.num_buckets,
        chunk_size=args.chunk_size,
        temp_dir=args.tmp_dir,
        verbose=True
    )
    
    output_path = os.path.join(args.h5ad_out, f'cellxmer_filtered_w{args.w_size}_b{args.num_buckets}.h5ad')
    adata_cellxmer.write_h5ad(output_path)
    
    if args.bin_size > 0:
        logging.info(f"Meshing AnnData into {args.bin_size} spatial unit-side hexagons")
        
        mesh_adata_cellxmer = create_meshed_adata(
            adata_cellxmer,
            1,  # assume the user provides bin_size in the correct units (no rescaling!)
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
    logging.info("SUCCESS!")

def _run_cellxmer_non_chunked(args):
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
            adata_cellxmer,
            1,
            spot_diameter_um=args.bin_size,
            spot_distance_um=args.bin_size,
            bead_diameter_um=args.bin_size,
            mesh_type="hexagon"
        )

        mesh_adata_cellxmer.write_h5ad(os.path.join(args.h5ad_out, f'cellxmer_bin{args.bin_size}.h5ad'))

if __name__ == "__main__":
    from malva.cli import get_cellxmer_parser
    
    parser = get_cellxmer_parser()
    args = parser.parse_args()
    _run_cellxmer(args)
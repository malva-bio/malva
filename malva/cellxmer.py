import logging
import os

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix

from malva.index import MalvaIndex
from malva.kmer_processing import decode_kmer
from malva.spacemake import create_meshed_adata
from malva.utils import check_directory_exists

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
    _diff_counts_idx = (diff_counts>count_at_least) & (diff_counts < count_at_most)

    n_kmer_filter = diff_counts[_diff_counts_idx].shape
    interesting_kmers = indices[np.append(_diff_counts_idx, np.array([False]))]
    kmer_index.close()
    
    if verbose:
        logging.info(f"There are {n_kmer_filter[0]:,} {kmer_index.kmer_size}-mers with {count_at_least:,} < counts < {count_at_most:,}")

    kmer_index.open(mode='r')
    
    if verbose:
        logging.info(f"Creating sparse matrix")

    adata_X_or = csr_matrix((np.ones_like(data), data, indptr))
    data_Ts = adata_X_or[_diff_counts_idx].indices
    indptr_Ts = adata_X_or[_diff_counts_idx].indptr

    adata_X_tr = csr_matrix((np.ones_like(data_Ts), data_Ts, indptr_Ts)).T
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


def _run_cellxmer(args):
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
            1, # assume the user provides bin_size in the correct units (no rescaling!)
            spot_diameter_um=args.bin_size,
            spot_distance_um=args.bin_size,
            bead_diameter_um=args.bin_size,
            mesh_type="hexagon"
        )

        mesh_adata_cellxmer.write_h5ad(os.path.join(args.h5ad_out, f'cellxmer_bin{args.bin_size}.h5ad'))

    logging.info(f"Final AnnData object with cell-by-kmer matrix was stored at {args.h5ad_out}")
    logging.info("SUCCESS!")


if __name__ == "__main__":
    from malva.cli import get_cellxmer_parser

    args = get_cellxmer_parser().parse_args()
    _run_cellxmer()

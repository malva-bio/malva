import gzip
import logging
import os

import numpy as np
from rich.progress import track

from malva.index import MalvaIndex
from malva.reader import iterate_fasta
from malva.show import MalvaPlot
from malva.utils import (check_directory_exists, check_file_exists,
                         get_reference_cache)


class MalvaReferenceNotFound(Exception):
    pass


def write_mtx_header(file, shape):
    file.write(b"%%MatrixMarket matrix coordinate real general\n")
    file.write(b"%\n")
    file.write(f"{shape[0]:>20} {shape[1]:>20} {shape[2]:>20}\n".encode())  # We'll update the nnz at the end


def process_gene(
    kmer_index,
    utrs_gene,
    current_gene,
    mtx_file,
    feature_file,
    current_col,
    sliding_size: int = 128,
    pct_threshold: float = 0.65,
    count_at_most: int = 10_000,
    count_at_least: int = 10,
    single_count: bool = False,
):
    locs, counts, _ = kmer_index.where(
        utrs_gene,
        sliding_size=sliding_size,
        pct_threshold=pct_threshold,
        count_at_most=count_at_most,
        count_at_least=count_at_least,
        single_count=single_count,
    )
    counts = np.clip((counts / len(utrs_gene)).astype(int), 1, 10_000)

    for loc, count in zip(locs, counts):
        mtx_file.write(f"{loc+1} {current_col} {count}\n".encode())

    feature_file.write(f"{current_gene}\n".encode())

    return len(locs)


def resave_h5ad(folder, kmer_index):
    try:
        import anndata as ad
    except ImportError:
        # TODO: decide if we make a dependency, or if we import the code here (we don't use full functionality...)
        raise ImportError("Please install anndata: `pip install anndata`")
    import pandas as pd

    matrix_file = os.path.join(folder, "matrix.mtx")
    check_file_exists(matrix_file, except_when=False)

    features_file = os.path.join(folder, "features.tsv.gz")
    check_file_exists(features_file, except_when=False)

    # will except if the file exists
    h5ad_file = os.path.join(folder, "pseudoquant.h5ad")
    check_file_exists(h5ad_file, except_when=True)

    adata = ad.read_mtx(matrix_file)
    adata.var_names = pd.read_csv(features_file, header=None, sep="\t")[0]

    n_spatial = (kmer_index.coord_lims[1] + 1) * (kmer_index.coord_lims[3] + 1)
    xy = np.unravel_index(
        np.arange(n_spatial), (kmer_index.coord_lims[1] + 1, kmer_index.coord_lims[3] + 1), order="C"
    )

    adata.obsm["spatial"] = np.vstack(xy).T

    adata.write_h5ad(h5ad_file)


def process_reference(
    kmer_index,
    reference_file,
    folder_out,
    verbose=True,
    sliding_size: int = 128,
    pct_threshold: float = 0.65,
    count_at_most: int = 10_000,
    count_at_least: int = 10,
    single_count: bool = False,
):
    kmer_index.verbose = False
    kmer_index.open()
    with open(os.path.join(folder_out, "matrix.mtx"), "wb") as mtx_file, gzip.open(
        os.path.join("features.tsv.gz"), "wb"
    ) as feature_file:

        current_gene = ""
        utrs_gene = []
        current_col = 0
        total_nnz = 0

        # we reserve the size of the header
        write_mtx_header(mtx_file, (0, 0, 0))

        if verbose:
            iterator = track(iterate_fasta(reference_file), description=f"Running pseudo-quantification")
        else:
            iterator = iterate_fasta(reference_file)
        for seq in iterator:
            it_gene_name = seq[0].split(":")[0]

            if it_gene_name != current_gene:
                if utrs_gene:
                    nnz = process_gene(kmer_index, utrs_gene, current_gene, mtx_file, feature_file, current_col + 1, sliding_size, pct_threshold, count_at_most, count_at_least, single_count)
                    total_nnz += nnz
                    current_col += 1

                utrs_gene = []
                current_gene = it_gene_name

            if seq[1] == "" or len(seq[1]) < 24 * 2 or len(seq[1]) > 3_000:
                continue

            utrs_gene.append(seq[1])

        if utrs_gene:
            nnz = process_gene(kmer_index, utrs_gene, current_gene, mtx_file, feature_file, current_col + 1)
            total_nnz += nnz
            current_col += 1

        # TODO: write the barcodes file, optionally it will contain the spatial coordinates...

    kmer_index.close()

    # TODO: the n_spatial is calcualted from lims, but sum one, otherwise not correct!
    # TODO: check in indexes.pyx if this is correct (how to compute n_spatial)
    n_spatial = (kmer_index.coord_lims[1] + 1) * (kmer_index.coord_lims[3] + 1)

    with open("matrix.mtx", "r+b") as mtx_file:
        mtx_file.seek(0)
        write_mtx_header(mtx_file, (n_spatial, current_col, total_nnz))

    print(f"MTX file created with shape: {n_spatial} x {current_col}, non-zero elements: {total_nnz}")


def _run_quant(args):
    kmer_index = MalvaIndex(args.index_in)

    # the output directory must not exist
    outdir_exists = check_directory_exists(args.folder_out)
    if not outdir_exists:
        logging.warn("The specified output directory did not exist. Creating...")
        os.mkdir(args.folder_out)

    reference_file = get_reference_cache(args.reference)
    logging.info(f"Will load reference '{args.reference}'")

    logging.info(f"Running pseudo-quantification")
    process_reference(
        kmer_index,
        reference_file,
        args.folder_out,
        sliding_size=args.sliding_size,
        pct_threshold=args.pct_threshold,
        count_at_most=args.kmer_max,
        count_at_least=args.kmer_min,
        single_count=args.single_count,
    )

    if args.h5ad:
        logging.info("Resaving pseudoquantification as AnnData (h5ad)")
        resave_h5ad(args.folder_out, kmer_index)

    logging.info("SUCCESS!")


if __name__ == "__main__":
    from malva.cli import get_quant_parser

    args = get_quant_parser().parse_args()
    _run_quant()

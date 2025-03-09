import gzip
import logging
import os

import numpy as np

from malva.indexes import MalvaIndex, BackgroundModel
from malva.reader import iterate_fasta
from malva.spacemake import create_meshed_adata
from malva.utils import (check_directory_exists, check_file_exists,
                         get_reference_cache)

N_EACH_REPORT = 100

class MalvaReferenceNotFound(Exception):
    pass


def write_mtx_header(file, shape):
    file.write(b"%%MatrixMarket matrix coordinate real general\n")
    file.write(b"%\n")
    file.write(f"{shape[0]:>20} {shape[1]:>20} {shape[2]:>20}\n".encode())  # We'll update the nnz at the end


def process_batch(
    kmer_index,
    seqs_batch,
    gene_names,
    mtx_file,
    feature_file,
    start_col,
    sliding_size: int = 128,
    pct_threshold: float = 0.65,
    count_at_most: int = 10_000,
    count_at_least: int = 10,
    single_count: bool = False,
    use_background_model: bool = True
):
    results = kmer_index.where(
        seqs_batch,
        sliding_size=sliding_size,
        pct_threshold=pct_threshold,
        count_at_most=count_at_most,
        count_at_least=count_at_least,
        single_count=single_count,
        max_mem="1M",
        use_background_model=use_background_model
    )

    # we have to clip otherwise we wouldn't count those that have many
    # entries in the reference (e.g., many alternative 3'UTRs) but only
    # one count was found 
    # TODO: if one of the sequences has a lot of counts but not another,
    # this will lead undercounting because of the large number of "seqs_gene"
    # counts = np.clip((counts / len(seqs_gene)), 1, 10_000).astype(int)
    
    total_nnz = 0
    for i, (locs, counts, _) in enumerate(results):
        current_col = start_col + i
        gene_name = gene_names[i]
        
        for loc, count in zip(locs, counts):
            mtx_file.write(f"{loc+1} {current_col} {count}\n".encode())
        
        feature_file.write(f"{gene_name}\n".encode())
        total_nnz += len(locs)
    
    return total_nnz


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

    # TODO: load more efficiently when too large to reduce memory usage
    kmer_index.open(mode='r')
    if 'spatial_coord' in kmer_index.index:
        adata.obsm["spatial"] = kmer_index.spatial_coord[:]
    kmer_index.close()

    adata.write_h5ad(h5ad_file)

    return adata


def process_reference(
    kmer_index,
    reference_file,
    folder_out,
    use_background_model=True,
    verbose=True,
    sliding_size: int = 128,
    pct_threshold: float = 0.65,
    count_at_most: int = 10_000,
    count_at_least: int = 10,
    single_count: bool = False,
    batch_size: int = 500
):
    kmer_index.verbose = False
    kmer_index.open(mode='r')
    with open(os.path.join(folder_out, "matrix.mtx"), "wb") as mtx_file, gzip.open(
        os.path.join(folder_out, "features.tsv.gz"), "wb"
    ) as feature_file:

        current_gene = ""
        seqs_gene = []
        current_col = 0
        total_nnz = 0

        # Batch processing containers
        batch_seqs = []  # List of lists of sequences
        batch_genes = []  # List of gene names

        # we reserve the size of the header
        write_mtx_header(mtx_file, (0, 0, 0))

        def process_current_batch():
            nonlocal current_col, total_nnz
            if batch_seqs:
                nnz = process_batch(
                    kmer_index, batch_seqs, batch_genes, mtx_file, feature_file, 
                    current_col + 1, sliding_size, pct_threshold, count_at_most, 
                    count_at_least, single_count, use_background_model
                )
                total_nnz += nnz
                current_col += len(batch_seqs)
                if verbose and (current_col % N_EACH_REPORT) < len(batch_seqs):
                    logging.info(f"Processed {current_col} entries. Last sequence ID: {batch_genes[-1]}")

        for seq in iterate_fasta(reference_file):
            it_gene_name = seq[0].split(":")[0]
            
            if it_gene_name != current_gene:
                if seqs_gene:
                    # Add current gene to batch
                    batch_seqs.append(seqs_gene)
                    batch_genes.append(current_gene)
                    
                    # Process batch if full
                    if len(batch_seqs) >= batch_size:
                        process_current_batch()
                        batch_seqs = []
                        batch_genes = []
                        
                seqs_gene = []
                current_gene = it_gene_name
            
            if seq[1] == "" or len(seq[1]) < sliding_size + kmer_index.kmer_size:
                continue
                
            seqs_gene.append(seq[1])

        # Process last gene
        if seqs_gene:
            batch_seqs.append(seqs_gene)
            batch_genes.append(current_gene)
        
        # Process final batch
        if batch_seqs:
            process_current_batch()

        # TODO: write the barcodes file, optionally it will contain the spatial coordinates...

    kmer_index.close()

    # the n_spatial is calcualted from lims, but sum one, otherwise not correct!
    if 'spatial_coord' in kmer_index.index:
        n_spatial = kmer_index.n_spatial
    # we need to add +1 because indices from mtx file start at 1, not 0 (for the scRNA data)
    else:
        n_spatial = kmer_index.n_spatial

    with open(os.path.join(folder_out, "matrix.mtx"), "r+b") as mtx_file:
        mtx_file.seek(0)
        write_mtx_header(mtx_file, (n_spatial, current_col, total_nnz))

    logging.info(f"MTX file created at {folder_out} \n\twith shape: {n_spatial} x {current_col}, non-zero elements: {total_nnz}")


def _run_quant(args):
    kmer_index = MalvaIndex(args.index_in)

    # the output directory must not exist
    outdir_exists = check_directory_exists(args.folder_out)
    if not outdir_exists:
        logging.warning("The specified output directory did not exist. Creating...")
        os.mkdir(args.folder_out)

    reference_file = get_reference_cache(args.reference)
    logging.info(f"Will load reference '{args.reference}'")


    if not check_file_exists(os.path.join(args.folder_out, "matrix.mtx")):
        background_model = None
        if args.background_model is not None:
            logging.info(f"Loading background model")
            check_file_exists(args.background_model, except_when=False)
            background_model = BackgroundModel(kmer_index.kmer_size)
            background_model.load(args.background_model)
            kmer_index.set_background_model(background_model)

        logging.info(f"Running pseudo-quantification")
        process_reference(
            kmer_index,
            reference_file,
            args.folder_out,
            use_background_model=True if background_model is not None else False,
            sliding_size=args.sliding_size,
            pct_threshold=args.pct_threshold,
            count_at_most=args.kmer_max,
            count_at_least=args.kmer_min,
            single_count=args.single_count,
        )
    else:
        logging.info(f"Quantification matrix exists at {args.folder_out}. Skipping...")

    if args.h5ad:
        logging.info("Resaving pseudoquantification as AnnData (h5ad)")
        adata = resave_h5ad(args.folder_out, kmer_index)

    if args.h5ad and args.bin_size > 0:
        logging.info(f"Meshing AnnData into {args.bin_size} spatial unit-side hexagons")
        mesh_adata = create_meshed_adata(
            adata,
            1, # assume the user provides bin_size in the correct units (no rescaling!)
            spot_diameter_um=args.bin_size,
            spot_distance_um=args.bin_size,
            bead_diameter_um=args.bin_size,
            mesh_type="hexagon"
        )

        h5ad_mesh_file = os.path.join(args.folder_out, f"pseudoquant_bin{args.bin_size}.h5ad")
        check_file_exists(h5ad_mesh_file, except_when=True)
        mesh_adata.write_h5ad(h5ad_mesh_file)

    logging.info("SUCCESS!")


if __name__ == "__main__":
    from malva.cli import get_quant_parser

    args = get_quant_parser().parse_args()
    _run_quant()

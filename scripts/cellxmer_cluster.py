import logging
import os

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

from scipy import optimize

from malva.utils import check_directory_exists

def run_clustering(adata, savefig=None, resolution=1):
    """
    Automated analysis pipeline for filtering and clustering
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing raw counts (not filtered nor normalized)
    savefig : str, default=None
        Folder where to save the plots. By default, save in current path
    resolution : float, default=1
        Resolution used for leiden clustering (higher values = more clusters)
        
    Returns
    -------
    adata_filtered : AnnData
        AnnData object containing only the called cells    
    """
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution = resolution, key_added="leiden")
    sc.tl.umap(adata)

    sc.tl.rank_genes_groups(adata, 'leiden', use_raw=False, pts=True)
    sc.tl.dendrogram(adata, 'leiden', use_raw=False)

    if savefig is not None:
        sc.pl.umap(adata, color=["total_counts", "leiden"], cmap='inferno', show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(savefig, "umap_counts_clusters.png"))
        sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, standard_scale='var', min_logfoldchange=2, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(savefig, "dotplot_markers.png"))
        
        bestmarkers = [adata.uns["rank_genes_groups"]["names"][0][i] for i in range(len(adata.uns["rank_genes_groups"]["names"][0]))]
        sc.pl.umap(adata, color=bestmarkers, legend_fontsize=4, legend_fontoutline=0.1, cmap='inferno', show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(savefig, "umap_markers.png"))

    return adata


def preprocess_adata(adata, gene_cutoff=5000, cell_cutoff=10):
    """
    Preprocesses by filtering cells (by counts) and genes (by cells),
    and then applies normalization. Copies raw counts.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing raw counts (not filtered nor normalized)
        gene_cutoff : int
        Gene count threshold used for cell filtering
    cell_cutoff : int
        Cell count threshold used for gene filtering
        
    Returns
    -------
    adata_filtered : AnnData
        AnnData object containing only the called cells
    """
    adata.raw = adata.copy()
    sc.pp.filter_cells(adata, min_genes=gene_cutoff)
    sc.pp.filter_genes(adata, min_cells=cell_cutoff)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

    return adata


def bucket_threshold_cell_counting(adata, expected_cells=None, min_cells=2, max_cells=None, 
                              percentile=99, ordmag_divisor=10, plot=True):
    """
    Implementation of a Gene threshold method for cell calling, similar to Cell Ranger's approach
    but independent of Cell Ranger and working directly with AnnData objects.

    This is specific for cellxmer objects
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing raw counts (not filtered or normalized)
    expected_cells : int, optional
        Expected number of cells in the dataset. If None, it will be estimated.
    min_cells : int, default=2
        Minimum number of cells to consider in the grid search
    max_cells : int, optional
        Maximum number of cells to consider in the grid search.
        If None, it will be set to int(n_barcodes/2) or 45,000, whichever is smaller.
    percentile : int, default=99
        Percentile used for the UMI threshold calculation (Cell Ranger uses 99th percentile)
    ordmag_divisor : int, default=10
        Divisor used in the Order of Magnitude algorithm (Cell Ranger uses 10)
    plot : bool, default=True
        Whether to plot the UMI distribution and threshold
        
    Returns
    -------
    adata_filtered : AnnData
        AnnData object containing only the called cells
    threshold : float
        UMI count threshold used for cell calling
    cell_barcodes : list
        List of barcodes that were called as cells
    """
    # Compute based on the occupancy of buckets!
    total_umi_counts = adata.obs['n_genes_by_counts'].values
    
    # Sort barcodes by UMI counts (descending)
    sorted_indices = np.argsort(total_umi_counts)[::-1]
    sorted_counts = total_umi_counts[sorted_indices]
    
    n_barcodes = len(total_umi_counts)
    barcode_rank = np.arange(1, n_barcodes + 1)
    
    # Set max_cells if not provided
    if max_cells is None:
        max_cells = min(int(n_barcodes / 2), 45000)
    
    # Define the Order of Magnitude function
    def order_magnitude(n_top):
        if n_top <= 0 or n_top >= n_barcodes:
            return 0
        
        # Get the 99th percentile UMI count of the top n_top barcodes
        top_counts = sorted_counts[:int(n_top)]
        p99_count = np.percentile(top_counts, percentile)
        
        # Count barcodes with UMI counts > p99/10
        threshold = p99_count / ordmag_divisor
        cells_above_threshold = np.sum(sorted_counts > threshold)
        
        return cells_above_threshold
    
    # Define the loss function to minimize
    def loss_function(x):
        ordmag_x = order_magnitude(int(x))
        relative_loss = ((ordmag_x - x) ** 2) / x if x > 0 else float('inf')
        return relative_loss
    
    # If expected_cells is not provided, estimate it
    if expected_cells is None:
        # Grid search to find the number of cells that minimizes the loss
        x_values = np.logspace(np.log10(min_cells), np.log10(max_cells), 100).astype(int)
        best_x = min_cells
        min_loss = float('inf')
        
        for x in x_values:
            current_loss = loss_function(x)
            if current_loss < min_loss:
                min_loss = current_loss
                best_x = x
        
        # Refine with a more focused optimization
        result = optimize.minimize_scalar(
            loss_function, 
            bounds=(max(min_cells, best_x/2), min(max_cells, best_x*2)),
            method='bounded'
        )
        expected_cells = int(result.x)
    
    # Get the 99th percentile UMI count of the top expected_cells barcodes
    top_counts = sorted_counts[:expected_cells]
    p99_count = np.percentile(top_counts, percentile)
    
    # Calculate threshold
    threshold = p99_count / ordmag_divisor
    
    # Call cells (barcodes with UMI counts > threshold)
    is_cell = total_umi_counts > threshold
    cell_barcodes = adata.obs_names[is_cell].tolist()
    
    # Filter AnnData to keep only cells
    adata_filtered = adata[is_cell].copy()
    
    # Add cell calling information to the AnnData object
    adata.obs['total_umi_counts'] = total_umi_counts
    adata.obs['is_cell'] = is_cell
    
    # Additional statistics for the user
    n_cells_called = sum(is_cell)
    mean_umis_per_cell = np.mean(total_umi_counts[is_cell])
    median_umis_per_cell = np.median(total_umi_counts[is_cell])
    
    # Print summary
    print(f"Cell calling summary:")
    print(f"  - Expected cells (estimated): {expected_cells}")
    print(f"  - UMI threshold: {threshold:.2f}")
    print(f"  - Cells called: {n_cells_called} out of {n_barcodes} barcodes ({n_cells_called/n_barcodes*100:.2f}%)")
    print(f"  - Mean UMIs per cell: {mean_umis_per_cell:.2f}")
    print(f"  - Median UMIs per cell: {median_umis_per_cell:.2f}")
    
    # Plot the UMI distribution and threshold
    if plot:
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Barcode rank plot (log-log)
        plt.subplot(1, 2, 1)
        plt.loglog(barcode_rank, sorted_counts, label='UMI counts')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.axvline(x=n_cells_called, color='g', linestyle='--', label=f'Cells called: {n_cells_called}')
        plt.xlabel('Barcode rank')
        plt.ylabel('UMI counts')
        plt.title('Barcode rank plot')
        plt.legend()
        
        # Plot 2: Histogram of UMI counts
        plt.subplot(1, 2, 2)
        plt.hist(np.log10(total_umi_counts + 1), bins=100, alpha=0.7)
        plt.axvline(x=np.log10(threshold + 1), color='r', linestyle='--', 
                   label=f'Threshold: {threshold:.2f}')
        plt.xlabel('log10(UMI counts + 1)')
        plt.ylabel('Frequency')
        plt.title('UMI count distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return adata_filtered, threshold, cell_barcodes

def _run_cellxmer_clustering(args):
    if not check_directory_exists(args.savefig, except_when=None):
        logging.info("The --savefig directory does not exist. Creating...")
        os.mkdir(args.savefig)
    
    # File loading, thresholding and QC metrics
    adata = sc.read_h5ad(args.adata_in)
    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)

    _, threshold, cells = bucket_threshold_cell_counting(
        adata, expected_cells=None, plot=True
    )

    logging.info(f"Cells called by the Bucket cutoff: {len(cells)}")
    logging.info(f"Bucket cutoff: {int(threshold)}")

    # Step 2: Prepare AnnData with non-technical gene marking
    adata = preprocess_adata(adata, gene_cutoff=int(threshold))
    
    # Step 3: Run clustering pipeline
    adata = run_clustering(adata, savefig=args.savefig)

    logging.info(f"Writing AnnData object with annotation was stored at {args.adata_out}")
    adata.write_h5ad(args.adata_out)

    logging.info("SUCCESS!")
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster cellxmer objects.")
    parser.add_argument(
        "--adata-in",
        type=str,
        required=True,
        help="cell-by-bucket matrix in AnnData format, to be annotated. It must be unfiltered.",
    )
    parser.add_argument(
        "--adata-out",
        type=str,
        required=True,
        help="Where to save the annotated, normalized and filtered AnnData object",
    )
    parser.add_argument(
        "--savefig",
        type=str,
        required=False,
        default=None,
        help="Folder where the output plots are saved. When not specified, plots are not generated nor saved.",
    )
    
    args = parser.parse_args()
    _run_cellxmer_clustering(args)
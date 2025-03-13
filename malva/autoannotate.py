import json
import logging
import os

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

import pandas as pd

from scipy import stats, optimize

from malva.utils import get_reference_cache, check_directory_exists

def load_marker_genes(json_path, exclude_technical=True):
    """
    Load marker genes from a JSON file, optionally excluding technical genes
    
    Parameters:
    -----------
    json_path : str
        Path to the marker gene JSON file
    exclude_technical : bool
        If True, exclude any gene categories that start with 'technical_'
        
    Returns:
    --------
    dict
        Dictionary of filtered marker genes
    """
    with open(json_path, 'r') as f:
        all_markers = json.load(f)
    
    # Filter out technical gene categories if requested
    if exclude_technical:
        filtered_markers = {k: v for k, v in all_markers.items() 
                           if not k.startswith('technical_')}
        
        # Log what was excluded
        excluded = set(all_markers.keys()) - set(filtered_markers.keys())
        if excluded:
            print(f"Excluded technical categories: {', '.join(excluded)}")
    else:
        filtered_markers = all_markers
    
    return filtered_markers

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
    sc.tl.pca(adata, svd_solver='arpack', mask_var='non_technical')
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

def preprocess_adata(adata, umi_cutoff=500, cell_cutoff=2):
    """
    Preprocesses by filtering cells (by counts) and genes (by cells),
    and then applies normalization. Copies raw counts.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing raw counts (not filtered nor normalized)
        umi_cutoff : int
        UMI count threshold used for cell filtering
    cell_cutoff : int
        Cell count threshold used for gene filtering
        
    Returns
    -------
    adata_filtered : AnnData
        AnnData object containing only the called cells
    """
    adata.raw = adata.copy()
    sc.pp.filter_cells(adata, min_counts=umi_cutoff)
    sc.pp.filter_genes(adata, min_cells=cell_cutoff)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    return adata


def score_cells_by_cell_type(adata, cell_markers):
    """
    Score cells for each cell type based on marker gene expression
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with cells as rows and genes as columns
    cell_markers : dict
        Dictionary mapping cell types to lists of marker genes
        
    Returns:
    --------
    adata : AnnData
        Input object with scores added to obs
    """
    # Ensure normalized and log-transformed data
    if 'log1p' not in adata.uns:
        print("Note: Data should be normalized and log-transformed")
    
    # Add a score for each cell type
    for cell_type, markers in cell_markers.items():
        if "technical_" in cell_type or "HALLMARK_" in cell_type:
            continue

        # Only use markers that are in the dataset
        available_markers = [m for m in markers if m in adata.var_names]
        
        try:
            if len(available_markers) > 0:
                # Use scanpy's score_genes function
                sc.tl.score_genes(adata, gene_list=available_markers, 
                                score_name=f"score_{cell_type.replace(' ', '_')}")
                print(f"Scored {cell_type}: {len(available_markers)}/{len(markers)} markers found")
            else:
                print(f"Warning: No markers found for {cell_type}")
        except:
            print(f"Warning: No markers found for {cell_type}")
    
    return adata

def get_top_cell_types(adata, n_types=3, score_prefix="score_"):
    """
    For each cell, get the top scoring cell types
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data with cell type scores
    n_types : int
        Number of top cell types to retrieve
    score_prefix : str
        Prefix for score columns
        
    Returns:
    --------
    top_cell_types : pd.DataFrame
        DataFrame with top cell types and scores for each cell
    """
    # Get all score columns
    score_cols = [col for col in adata.obs.columns if col.startswith(score_prefix)]
    
    # Create a DataFrame with all scores
    scores_df = adata.obs[score_cols].copy()
    
    # Prepare output DataFrame
    result = pd.DataFrame(index=adata.obs_names)
    
    # For each cell, get the top cell types
    for i in range(1, n_types + 1):
        # Get the column with the highest score
        result[f'cell_type_{i}'] = scores_df.idxmax(axis=1).apply(lambda x: x[len(score_prefix):].replace('_', ' '))
        result[f'score_{i}'] = scores_df.max(axis=1)
        
        # Remove the top cell type from consideration for the next iteration
        for cell in adata.obs_names:
            scores_df.loc[cell, scores_df.loc[cell].idxmax()] = -1
    
    return result

def annotate_clusters(adata, cell_markers, cluster_key='leiden', threshold=0.4, min_markers=3, savefig=None):
    """
    Score and annotate cell types based on cluster-level expression signatures
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with clustering results
    cell_markers : dict
        Dictionary mapping cell types to lists of marker genes
    cluster_key : str
        Key in adata.obs for cluster assignments
    threshold : float
        Minimum differential expression score threshold for a marker to be considered
    min_markers : int
        Minimum number of markers needed for a cell type to be assigned
    savefig : str
        Path where the plots will be saved. If None, then no plots are saved
        
    Returns:
    --------
    adata : AnnData
        Input object with cell type annotations added
    annotations : pd.DataFrame
        Detailed annotation information for each cluster
    """
    # Get unique clusters
    clusters = adata.obs[cluster_key].unique()
    
    # Create a DataFrame to store cluster-level annotations
    cluster_annotations = pd.DataFrame(index=clusters)
    
    # For each cluster, identify the most likely cell type
    for cluster in clusters:
        # Get cells in this cluster
        cluster_mask = adata.obs[cluster_key] == cluster
        cluster_cells = adata[cluster_mask]
        
        # Calculate average expression by cluster for all genes
        if isinstance(cluster_cells.X, np.ndarray):
            cluster_expr = np.mean(cluster_cells.X, axis=0)
        else:
            cluster_expr = np.mean(cluster_cells.X.toarray(), axis=0)
        
        # Calculate average expression for all other clusters
        other_mask = adata.obs[cluster_key] != cluster
        other_cells = adata[other_mask]
        if isinstance(other_cells.X, np.ndarray):
            other_expr = np.mean(other_cells.X, axis=0)
        else:
            other_expr = np.mean(other_cells.X.toarray(), axis=0)
        
        # Calculate fold change and differential score
        epsilon = 1e-9  # To avoid division by zero
        fold_change = (cluster_expr + epsilon) / (other_expr + epsilon)
        diff_score = cluster_expr - other_expr
        
        # Score each cell type based on its markers
        cell_type_scores = {}
        cell_type_marker_counts = {}
        cell_type_top_markers = {}
        
        for cell_type, markers in cell_markers.items():
            if "technical_" in cell_type or "HALLMARK_" in cell_type:
                continue
                
            # Only use markers that are in the dataset
            available_markers = [m for m in markers if m in adata.var_names]
            if len(available_markers) < min_markers:
                continue
                
            # Calculate marker scores for this cell type
            marker_scores = []
            for marker in available_markers:
                marker_idx = adata.var_names.get_loc(marker)
                marker_score = diff_score[marker_idx]
                marker_fc = fold_change[marker_idx]
                
                # Only count markers that are differentially expressed in this cluster
                if marker_score > threshold:
                    marker_scores.append((marker, marker_score, marker_fc))
            
            # Sort markers by score
            marker_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate overall score based on top markers
            if len(marker_scores) >= min_markers:
                # Use geometric mean of top marker scores to reduce influence of outliers
                top_scores = [score for _, score, _ in marker_scores[:10]]
                if top_scores:
                    overall_score = np.exp(np.mean(np.log(np.array(top_scores) + epsilon)))
                    cell_type_scores[cell_type] = overall_score
                    cell_type_marker_counts[cell_type] = len(marker_scores)
                    cell_type_top_markers[cell_type] = marker_scores[:5]  # Store top 5 markers
        
        # Initialize variables to handle the case where no cell types meet criteria
        cell_label = "Unknown"
        confidence = "low"
        marker_str = "No significant markers"
        best_score = 0
        best_cell_type = None  # Initialize to avoid UnboundLocalError
        
        # Find the best matching cell type
        if cell_type_scores:
            # Sort cell types by score
            sorted_cell_types = sorted(cell_type_scores.items(), key=lambda x: x[1], reverse=True)
            best_cell_type = sorted_cell_types[0][0]
            best_score = sorted_cell_types[0][1]
            
            # Check if the best match is significantly better than the second best
            if len(sorted_cell_types) > 1:
                second_best = sorted_cell_types[1][0]
                second_score = sorted_cell_types[1][1]
                score_ratio = best_score / (second_score + epsilon)
                
                # If scores are close, might be a mixed population
                if score_ratio < 1.5:  # Threshold for considering mixed population
                    cell_label = f"{best_cell_type}/{second_best}"
                    confidence = "medium"
                else:
                    cell_label = best_cell_type
                    confidence = "high"
            else:
                cell_label = best_cell_type
                confidence = "high"
                
            # Get top markers for the best cell type
            top_markers = cell_type_top_markers[best_cell_type]
            marker_str = ", ".join([f"{m} ({s:.2f}x)" for m, s, fc in top_markers])
        
        # Store annotation for this cluster
        cluster_annotations.loc[cluster, "cell_type"] = cell_label
        cluster_annotations.loc[cluster, "confidence"] = confidence
        cluster_annotations.loc[cluster, "score"] = best_score
        cluster_annotations.loc[cluster, "n_markers"] = cell_type_marker_counts.get(best_cell_type, 0) if best_cell_type else 0
        cluster_annotations.loc[cluster, "top_markers"] = marker_str
    
    # Add cell type annotations to original data
    adata.obs["cell_type"] = adata.obs[cluster_key].map(cluster_annotations["cell_type"])
    adata.obs["annotation_confidence"] = adata.obs[cluster_key].map(cluster_annotations["confidence"])
    
    # Add cluster-level metadata
    adata.uns["cluster_annotations"] = cluster_annotations

    if savefig is not None:
        sc.pl.umap(adata, color=['leiden', 'cell_type'], legend_loc='on data', legend_fontsize=8, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(savefig, "umap_clustername.png"))
    
    if "spatial" in adata.obsm and savefig is not None:
        ax = sc.pl.embedding(adata, color=['leiden'], show=False, basis='spatial')
        ax.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(savefig, "spatial_leiden.png"))
        ax = sc.pl.embedding(adata, color=['cell_type'], show=False, basis='spatial')
        ax.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(savefig, "spatial_clustername.png"))

    
    return adata, cluster_annotations

def get_detailed_cluster_annotations(adata, cluster_key='leiden'):
    """Get detailed cluster annotations with cell type distribution"""
    clusters = adata.obs[cluster_key].unique()
    results = []
    
    for cluster in clusters:
        cells_in_cluster = adata.obs[cluster_key] == cluster
        n_cells = cells_in_cluster.sum()
        
        # Count cell types in this cluster
        type_counts = adata.obs.loc[cells_in_cluster, 'cell_type_1'].value_counts()
        
        # Get the top 3 types
        top_types = []
        for i, (cell_type, count) in enumerate(type_counts.items()):
            if i >= 3:
                break
            pct = (count / n_cells) * 100
            top_types.append(f"{cell_type} ({pct:.1f}%)")
        
        # Come up with a composite name for the cluster
        if len(top_types) > 0 and type_counts.iloc[0] / n_cells > 0.7:
            # If one type is dominant (>70%), use that
            cluster_name = type_counts.index[0]
        elif len(top_types) > 1 and (type_counts.iloc[0] + type_counts.iloc[1]) / n_cells > 0.8:
            # If top 2 types make up >80%, use a compound name
            cluster_name = f"{type_counts.index[0]}/{type_counts.index[1]}"
        else:
            # Otherwise, list top types
            cluster_name = "Mixed: " + "/".join([t.split()[0] for t in top_types[:2]])
        
        results.append({
            'cluster': cluster,
            'n_cells': n_cells,
            'cluster_name': cluster_name,
            'cell_type_composition': " | ".join(top_types)
        })
    
    return pd.DataFrame(results).sort_values('cluster')

def analyze_technical_genes(adata, housekeeping_genes, savefig=None):
    """
    Analyze technical and housekeeping genes to assess data quality
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    housekeeping_genes : dict
        Dictionary of gene categories and their corresponding genes
    savefig : str
        Path where the plots will be saved. If None, then no plots are saved
    """
    # Create a DataFrame to store metrics
    metrics = pd.DataFrame(index=adata.obs_names)
    
    # Calculate metrics for each category
    for category, genes in housekeeping_genes.items():
        if "technical_" not in category:
            continue
        
        # Find genes present in the dataset
        available_genes = [g for g in genes if g in adata.var_names]
        
        if len(available_genes) > 0:
            print(f"{category}: {len(available_genes)}/{len(genes)} genes found")
            
            # Calculate mean expression
            if isinstance(adata.X, np.ndarray):
                expr = adata[:, available_genes].X
            else:
                expr = adata[:, available_genes].X.toarray()
            
            metrics[f'{category}_mean'] = np.mean(expr, axis=1)
            metrics[f'{category}_percent'] = (np.sum(expr > 0, axis=1) / len(available_genes)) * 100
            
            # Score cells for this category
            sc.tl.score_genes(adata, gene_list=available_genes, score_name=f"score_{category}")
        else:
            print(f"Warning: No {category} genes found in the dataset")
    
    # Add metrics to adata.obs
    for col in metrics.columns:
        adata.obs[col] = metrics[col]
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Housekeeping gene metrics across clusters
    categories = list(housekeeping_genes.keys())
    available_categories = [cat for cat in categories if f'score_{cat}' in adata.obs.columns and "technical_" == cat[:len("technical_")]]
    
    if available_categories:
        sc.pl.violin(adata, [f'score_{cat}' for cat in available_categories], 
                     groupby='leiden', rotation=90, stripplot=False, multi_panel=True)
    
    # UMAP colored by technical metrics
    for category in available_categories:
        if "technical_" not in category:
            continue

        sc.pl.umap(adata, color=f'score_{category}', title=f'{category} Score', 
                  cmap='viridis', s=50, alpha=0.8)
        
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(savefig, "technical_genes.png"))
    
    return adata

def umi_threshold_cell_calling(adata, expected_cells=None, min_cells=2, max_cells=None, 
                              percentile=99, ordmag_divisor=10, plot=True):
    """
    Implementation of a UMI threshold method for cell calling, similar to Cell Ranger's approach
    but independent of Cell Ranger and working directly with AnnData objects.
    
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
    # Calculate total UMIs per barcode
    if isinstance(adata.X, np.ndarray):
        total_umi_counts = np.array(adata.X.sum(axis=1)).flatten()
    else:  # For sparse matrices
        total_umi_counts = np.array(adata.X.sum(axis=1)).flatten()
    
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

def simple_good_turing_smoothing(counts):
    """
    Implements the Simple Good-Turing smoothing algorithm to estimate probabilities
    for unseen events, ensuring non-zero proportions for genes with zero counts.
    
    Parameters
    ----------
    counts : array-like
        Count data for each gene
        
    Returns
    -------
    probabilities : ndarray
        Smoothed probabilities for each gene
    """
    # Convert counts to array and ensure it's 1D
    counts = np.asarray(counts).flatten()
    
    # Calculate frequency of frequencies
    unique_counts, counts_freq = np.unique(counts, return_counts=True)
    
    # Handle the case where we have zeros in the data
    if 0 in unique_counts:
        zero_idx = np.where(unique_counts == 0)[0][0]
        unique_counts = np.delete(unique_counts, zero_idx)
        counts_freq = np.delete(counts_freq, zero_idx)
    
    # Need at least two non-zero frequency values for interpolation
    if len(unique_counts) < 2:
        # If not enough data, use simple add-one smoothing
        return (counts + 1) / (np.sum(counts) + len(counts))
    
    # Log-log linear regression for frequency estimation
    log_counts = np.log(unique_counts)
    log_freq = np.log(counts_freq)
    
    # Use linear regression to estimate the slope
    slope, intercept, _, _, _ = stats.linregress(log_counts, log_freq)
    
    # For each count r, estimate r+1 frequency
    n_r_plus_one = {}
    total_counts = np.sum(counts)
    
    for r, n_r in zip(unique_counts, counts_freq):
        if r+1 in unique_counts:
            idx = np.where(unique_counts == r+1)[0][0]
            n_r_plus_one[r] = counts_freq[idx]
        else:
            # Use the regression to estimate
            n_r_plus_one[r] = np.exp(intercept + slope * np.log(r+1))
    
    # Calculate the smoothed probabilities
    probabilities = np.zeros_like(counts, dtype=float)
    zero_prob = 0
    
    for i, c in enumerate(counts):
        if c == 0:
            probabilities[i] = 0  # Will be set to zero_prob later
        else:
            # Good-Turing estimate: (c+1) * N(c+1) / N(c) / total
            if c in n_r_plus_one and n_r_plus_one[c] > 0:
                probabilities[i] = (c + 1) * n_r_plus_one[c] / (counts_freq[np.where(unique_counts == c)[0][0]] * total_counts)
            else:
                # Fallback to simple smoothing if we can't estimate
                probabilities[i] = (c + 0.5) / (total_counts + 0.5 * len(counts))
    
    # Assign probability mass for unseen events (0 counts)
    if 1 in unique_counts:
        n1 = counts_freq[np.where(unique_counts == 1)[0][0]]
        zero_prob = n1 / total_counts / np.sum(counts == 0) if np.sum(counts == 0) > 0 else 0
    else:
        zero_prob = 0.1 / np.sum(counts == 0) if np.sum(counts == 0) > 0 else 0
    
    # Assign zero_prob to all genes with zero counts
    probabilities[counts == 0] = zero_prob
    
    # Normalize to ensure probabilities sum to 1
    probabilities = probabilities / np.sum(probabilities)
    
    return probabilities


def emptydrops_refinement(adata, adata_filtered=None, threshold=None, ambient_min_umi=1, 
                          ambient_max_umi=100, min_total_umi=500, fdr_threshold=0.01, 
                          plot=True):
    """
    Implementation of the EmptyDrops algorithm for refining cell calling by identifying
    low RNA content cells that are distinguishable from empty droplets.
    
    Parameters
    ----------
    adata : AnnData
        Original AnnData object containing all barcodes (filtered and unfiltered)
    adata_filtered : AnnData, optional
        AnnData object containing barcodes called as cells by OrdMag or another method.
        If None, it assumes all barcodes in adata are potential cells.
    threshold : float, optional
        UMI threshold used in initial cell calling. If None, it will be estimated.
    ambient_min_umi : int, default=1
        Minimum total UMI count to consider a barcode for ambient RNA profile estimation
    ambient_max_umi : int, default=100
        Maximum total UMI count to consider a barcode for ambient RNA profile estimation
    min_total_umi : int, default=500
        Minimum total UMI count for a barcode to be considered as a candidate cell
    fdr_threshold : float, default=0.01
        False discovery rate threshold for cell calling
    plot : bool, default=True
        Whether to plot diagnostic visualizations
        
    Returns
    -------
    adata_refined : AnnData
        AnnData object containing all called cells (OrdMag + EmptyDrops)
    ambient_profile : ndarray
        Estimated ambient RNA profile
    cell_barcodes : list
        List of barcodes that were called as cells
    """
    # Calculate total UMIs per barcode if not already in adata.obs
    if 'total_umi_counts' not in adata.obs:
        if isinstance(adata.X, np.ndarray):
            adata.obs['total_umi_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        else:  # For sparse matrices
            adata.obs['total_umi_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    
    # If adata_filtered is provided, identify barcodes already called as cells
    if adata_filtered is not None:
        already_called = adata.obs_names.isin(adata_filtered.obs_names)
    else:
        # If no filtered data provided, assume all are uncalled
        already_called = np.zeros(adata.shape[0], dtype=bool)
        
    # If threshold is not provided, estimate it
    if threshold is None and adata_filtered is not None:
        threshold = adata.obs.loc[already_called, 'total_umi_counts'].min()
    elif threshold is None:
        # Use a default threshold if no filtered data and no threshold provided
        threshold = ambient_max_umi
    
    # Identify ambient barcodes for background profile estimation
    is_ambient = (
        (adata.obs['total_umi_counts'] >= ambient_min_umi) & 
        (adata.obs['total_umi_counts'] <= ambient_max_umi) &
        (~already_called)
    )
    
    # Ensure we have enough ambient barcodes
    if np.sum(is_ambient) < 100:
        print(f"Warning: Only {np.sum(is_ambient)} ambient barcodes found. Consider adjusting ambient_min_umi and ambient_max_umi.")
        # Adjust ambient range if needed
        if np.sum(is_ambient) < 10:  # If critically low, expand range
            ambient_max_umi = np.percentile(adata.obs['total_umi_counts'], 10)
            is_ambient = (
                (adata.obs['total_umi_counts'] >= ambient_min_umi) & 
                (adata.obs['total_umi_counts'] <= ambient_max_umi) &
                (~already_called)
            )
            print(f"Adjusted ambient_max_umi to {ambient_max_umi:.2f}, now using {np.sum(is_ambient)} ambient barcodes.")
    
    # Extract ambient counts
    ambient_counts = adata.X[is_ambient].sum(axis=0)
    if isinstance(ambient_counts, np.matrix):
        ambient_counts = np.array(ambient_counts).flatten()
    
    # Apply Simple Good-Turing smoothing to get ambient profile
    ambient_profile = simple_good_turing_smoothing(ambient_counts)
    
    # Identify candidate barcodes for testing against the ambient profile
    # These are barcodes with UMI counts > ambient_max_umi and ≥ min_total_umi, not already called
    is_candidate = (
        (adata.obs['total_umi_counts'] > ambient_max_umi) & 
        (adata.obs['total_umi_counts'] >= min_total_umi) &
        (adata.obs['total_umi_counts'] < threshold) &  # Below the OrdMag threshold
        (~already_called)
    )
    
    # If no candidates, return the original filtered data
    if np.sum(is_candidate) == 0:
        print("No candidate barcodes found for EmptyDrops refinement.")
        if adata_filtered is not None:
            return adata_filtered, ambient_profile, adata_filtered.obs_names.tolist()
        else:
            # If no filtered data provided, return an empty AnnData object
            return adata[[]].copy(), ambient_profile, []
    
    # Calculate the log-likelihood ratio test for each candidate barcode
    pvalues = np.ones(adata.shape[0])
    test_stats = np.zeros(adata.shape[0])
    
    for i in np.where(is_candidate)[0]:
        # Get gene counts for this barcode
        if isinstance(adata.X, np.ndarray):
            barcode_counts = adata.X[i].copy()
        else:  # For sparse matrices
            barcode_counts = adata.X[i].toarray().flatten()
        
        total_umi = int(adata.obs['total_umi_counts'].iloc[i])
        
        # Skip if no counts (shouldn't happen but just in case)
        if total_umi == 0:
            continue
        
        # Calculate expected counts under the null (ambient) model
        expected_counts = ambient_profile * total_umi
        
        # Calculate test statistic using multinomial likelihood ratio test
        # We use a chi-squared approximation
        nonzero_idx = barcode_counts > 0
        if np.sum(nonzero_idx) > 0:
            # Calculate test statistic only for genes with non-zero counts
            obs = barcode_counts[nonzero_idx]
            exp = expected_counts[nonzero_idx]
            
            # Avoid division by zero
            exp = np.maximum(exp, 1e-10)
            
            # Calculate test statistic (2 * sum(O * log(O/E)))
            test_stat = 2 * np.sum(obs * np.log(obs / exp))
            df = np.sum(nonzero_idx) - 1  # Degrees of freedom
            
            if df > 0:  # Need at least 2 genes with non-zero counts
                pvalues[i] = 1 - stats.chi2.cdf(test_stat, df)
                test_stats[i] = test_stat
    
    # Apply multiple testing correction (Benjamini-Hochberg FDR)
    candidate_idx = np.where(is_candidate)[0]
    if len(candidate_idx) > 0:
        candidate_pvals = pvalues[candidate_idx]
        # Sort p-values
        sorted_idx = np.argsort(candidate_pvals)
        sorted_pvals = candidate_pvals[sorted_idx]
        
        # Calculate Benjamini-Hochberg critical values
        m = len(sorted_pvals)
        j = np.arange(1, m + 1)
        critical_values = j * fdr_threshold / m
        
        # Find the largest p-value that is <= its critical value
        significant = sorted_pvals <= critical_values
        if np.any(significant):
            max_idx = np.where(significant)[0][-1]
            significance_threshold = sorted_pvals[max_idx]
        else:
            significance_threshold = 0
        
        # Call cells with p-values <= significance threshold
        is_emptydrops_cell = np.zeros_like(pvalues, dtype=bool)
        is_emptydrops_cell[candidate_idx] = candidate_pvals <= significance_threshold
    else:
        is_emptydrops_cell = np.zeros_like(pvalues, dtype=bool)
    
    # Combine cells from OrdMag and EmptyDrops
    is_cell = already_called | is_emptydrops_cell
    
    # Create refined AnnData
    adata_refined = adata[is_cell].copy()
    
    # Add metadata to the original AnnData
    adata.obs['is_ambient'] = is_ambient
    adata.obs['is_candidate'] = is_candidate
    adata.obs['pvalue'] = pvalues
    adata.obs['test_statistic'] = test_stats
    adata.obs['is_emptydrops_cell'] = is_emptydrops_cell
    adata.obs['is_cell'] = is_cell
    adata.obs['cell_calling_method'] = 'None'
    adata.obs.loc[already_called, 'cell_calling_method'] = 'OrdMag'
    adata.obs.loc[is_emptydrops_cell, 'cell_calling_method'] = 'EmptyDrops'
    
    # Print summary
    n_ordmag = np.sum(already_called)
    n_emptydrops = np.sum(is_emptydrops_cell)
    n_total = np.sum(is_cell)
    
    print(f"EmptyDrops refinement summary:")
    print(f"  - Ambient barcodes used for background: {np.sum(is_ambient)}")
    print(f"  - Candidate barcodes tested: {np.sum(is_candidate)}")
    print(f"  - Cells called by OrdMag: {n_ordmag}")
    print(f"  - Additional cells called by EmptyDrops: {n_emptydrops}")
    print(f"  - Total cells called: {n_total}")
    
    # Create diagnostic plots
    if plot and np.sum(is_candidate) > 0:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: UMI counts vs. p-values
        scatter = axs[0].scatter(
            adata.obs['total_umi_counts'][is_candidate],
            -np.log10(pvalues[is_candidate] + 1e-10),
            c=is_emptydrops_cell[is_candidate],
            s=10,
            alpha=0.5,
            cmap='coolwarm'
        )
        axs[0].set_xscale('log')
        axs[0].set_xlabel('Total UMI counts')
        axs[0].set_ylabel('-log10(p-value)')
        axs[0].set_title('EmptyDrops p-values vs. UMI counts')
        axs[0].axhline(y=-np.log10(fdr_threshold), color='r', linestyle='--', 
                        label=f'FDR threshold: {fdr_threshold}')
        axs[0].legend()
        
        # Plot 2: Barcode rank plot with cell calls
        barcode_ranks = np.arange(1, adata.shape[0] + 1)
        sorted_umi = np.sort(adata.obs['total_umi_counts'].values)[::-1]
        
        axs[1].loglog(barcode_ranks, sorted_umi, 'grey', alpha=0.5, label='All barcodes')
        
        # Color points by calling method
        cell_ranks = np.where(is_cell[np.argsort(adata.obs['total_umi_counts'].values)[::-1]])[0] + 1
        ordmag_ranks = np.where(already_called[np.argsort(adata.obs['total_umi_counts'].values)[::-1]])[0] + 1
        emptydrops_ranks = np.where(is_emptydrops_cell[np.argsort(adata.obs['total_umi_counts'].values)[::-1]])[0] + 1
        
        if len(ordmag_ranks) > 0:
            ordmag_umis = sorted_umi[ordmag_ranks - 1]
            axs[1].scatter(ordmag_ranks, ordmag_umis, color='blue', s=10, alpha=0.7, label='OrdMag')
        
        if len(emptydrops_ranks) > 0:
            emptydrops_umis = sorted_umi[emptydrops_ranks - 1]
            axs[1].scatter(emptydrops_ranks, emptydrops_umis, color='red', s=10, alpha=0.7, label='EmptyDrops')
            
        axs[1].axhline(y=threshold, color='k', linestyle='--', label=f'OrdMag threshold: {threshold:.1f}')
        axs[1].axhline(y=ambient_max_umi, color='g', linestyle='--', label=f'Ambient max: {ambient_max_umi}')
        axs[1].axhline(y=min_total_umi, color='purple', linestyle='--', label=f'Min UMI: {min_total_umi}')
        
        axs[1].set_xlabel('Barcode rank')
        axs[1].set_ylabel('Total UMI counts')
        axs[1].set_title('Barcode rank plot with cell calls')
        axs[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    # Return the refined dataset, ambient profile, and cell barcodes
    cell_barcodes = adata.obs_names[is_cell].tolist()
    return adata_refined, ambient_profile, cell_barcodes

def load_markers(marker_source):
    """
    Load marker genes and prepare technical/non-technical gene lists
    
    Parameters:
    -----------
    marker_source : str
        Either 'human_markers', 'human_markers_hallmarks', 'mouse_markers', or a path to a custom JSON file
        
    Returns:
    --------
    tuple
        (cell_markers_nontechnical, cell_markers, nontechnical_genes)
    """
    import os
    
    # Check if source is a reference or file
    if marker_source in ['human_markers', 'human_markers_hallmarks', 'mouse_markers']:
        # It's a reference, get it from the reference cache
        marker_file = get_reference_cache(marker_source + "_json")
    elif os.path.isfile(marker_source):
        # It's a custom file
        marker_file = marker_source
    else:
        raise ValueError(f"Marker source '{marker_source}' is not a valid reference or file path")
    
    # Load markers without technical genes
    cell_markers_nontechnical = load_marker_genes(marker_file, exclude_technical=True)
    nontechnical_genes = np.concatenate(list(cell_markers_nontechnical.values()))
    
    # Load all markers including technical
    cell_markers = load_marker_genes(marker_file, exclude_technical=False)
    
    return cell_markers_nontechnical, cell_markers, nontechnical_genes

def score_annotate(adata, cell_markers, savefig=None):
    """
    Score cells for each cell type and annotate with top cell types
    
    Parameters:
    -----------
    adata : AnnData
        Clustered AnnData object
    cell_markers : dict
        Dictionary mapping cell types to marker genes
    savefig : str
        Path where the plots will be saved. If None, then no plots are saved
        
    Returns:
    --------
    AnnData
        AnnData with cell type scores and annotations
    """
    # Score cells for each cell type
    adata = score_cells_by_cell_type(adata, cell_markers)
    
    # Get top cell types for each cell
    top_cell_types = get_top_cell_types(adata, n_types=3)
    
    # Add to AnnData object
    adata.obs = pd.concat([adata.obs, top_cell_types], axis=1)
    
    # Get detailed cluster annotations
    detailed_annotations = get_detailed_cluster_annotations(adata)
    
    # Map cluster names
    cluster_name_map = detailed_annotations.set_index('cluster')['cluster_name'].to_dict()
    adata.obs['cluster_name'] = adata.obs['leiden'].map(cluster_name_map)

    
    return adata, detailed_annotations

def _run_autoannotate(args):
    if args.reference not in ['human_markers', 'human_markers_hallmarks', 'mouse_markers']:
        logging.error("The --reference has to be either 'human_markers', 'human_markers_hallmarks', or 'mouse_markers'. Others are not supported yet.")

    if not check_directory_exists(args.savefig, except_when=None):
        logging.info("The --savefig directory does not exist. Creating...")
        os.mkdir(args.savefig)
    
    # File loading, thresholding and QC metrics
    adata = sc.read_h5ad(args.adata_in)
    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)

    _, threshold, cells = umi_threshold_cell_calling(
        adata, expected_cells=None, plot=True
    )

    logging.info(f"Cells called by the UMI cutoff: {len(cells)}")
    logging.info(f"UMI cutoff: {int(threshold)}")

    # Step 1: Load markers
    _, cell_markers, nontechnical_genes = load_markers(args.reference)

    # Step 2: Prepare AnnData with non-technical gene marking
    adata.var['non_technical'] = adata.var_names.isin(nontechnical_genes)
    adata = preprocess_adata(adata, umi_cutoff=int(threshold))
    
    # Step 3: Run clustering pipeline
    adata = run_clustering(adata, savefig=args.savefig)
    
    # Step 4: Score cell types
    adata, _ = annotate_clusters(
        adata, 
        cell_markers, 
        cluster_key='leiden', 
        threshold=0.4, # TODO: we can make user-adjustable...
        min_markers=3,
        savefig=args.savefig
    )
    
    # Step 5: Analyze technical genes
    adata = analyze_technical_genes(adata, cell_markers, args.savefig)

    logging.info(f"Writing AnnData object with annotation was stored at {args.adata_out}")
    adata.write_h5ad(args.adata_out)

    logging.info("SUCCESS!")
    
if __name__ == "__main__":
    from malva.cli import get_autoannotate_parser

    args = get_autoannotate_parser().parse_args()
    _run_autoannotate()
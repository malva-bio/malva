# Malva Tools Examples

This tutorial demonstrates how to use Malva locally to index, quantify, and query single-cell RNA-seq data.

## Dataset

We use the **1k Human PBMCs** dataset from 10x Genomics (v3 chemistry, 3' gene expression). This is a standard benchmark dataset containing approximately 1,000 peripheral blood mononuclear cells from a healthy donor.

## Tutorial Structure

Run the examples in order:

| Step | File | Description |
|------|------|-------------|
| 1 | `1_run_malva.sh` | Download data, build the index, and quantify gene expression |
| 2 | `2_simple_analysis.ipynb` | Load quantification results and perform standard single-cell analysis |
| 3 | `3_sequence_search.ipynb` | Query arbitrary sequences against the index and visualize results |

## Quick Start

First, request access to the malva binaries.

Then, run ```1_run_malva.sh``` to download data and build the index, then follow the notebooks in order.

## Output

After completing the tutorial, you will have:

- A Malva index of the PBMC dataset
- Gene expression quantification in h5ad format
- UMAP visualizations with cell type clusters
- Custom sequence query results projected onto cells
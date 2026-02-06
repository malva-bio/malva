#!/bin/bash
# =============================================================================
# Step 1: Download data, build index, and quantify gene expression
# =============================================================================
# This script prepares a Malva index from the 10x Genomics 1k PBMC dataset
# (v3 chemistry, 3' gene expression) and performs gene expression quantification.
#
# Runtime: ~5-10 minutes depending on your internet connection and hardware
# =============================================================================

# Create directory structure
mkdir -p {barcodes,reads,references,bins,indices/pbmc_1k_v3,quant/pbmc_1k_v3}

# -----------------------------------------------------------------------------
# Download required files
# -----------------------------------------------------------------------------

# Malva binary (available upon request for academic users)
# ...

# 10x Genomics cell barcode whitelist (v3 chemistry)
wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/3M-february-2018.txt -O barcodes/3M-february-2018.txt

# Human transcriptome reference (cDNA + ncRNA, repeat-masked)
wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/human_cdna_ncrna_masked.fa.gz -O references/human_cdna_ncrna_masked.fa.gz

# PBMC 1k v3 sequencing reads (R1: cell barcode + UMI, R2: cDNA)
wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/pbmc_1k_v3_S1_R1_001.fastq.gz -O reads/pbmc_1k_v3_S1_R1_001.fastq.gz
wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/pbmc_1k_v3_S1_R2_001.fastq.gz -O reads/pbmc_1k_v3_S1_R2_001.fastq.gz

# Add Malva to PATH for this session
export PATH=$PATH:$(pwd)/bins/

# -----------------------------------------------------------------------------
# Build the Malva index
# -----------------------------------------------------------------------------
# This step processes raw FASTQ reads and builds a searchable k-mer index.
# The index preserves cell barcode information for single-cell resolution.

malva index \
    --reads-in reads/pbmc_1k_v3_S1_R1_001.fastq.gz reads/pbmc_1k_v3_S1_R2_001.fastq.gz \
    --flavor sc_10x_v3 \
    --index-out indices/pbmc_1k_v3 \
    --spatial-bc-in barcodes/3M-february-2018.txt \
    --kmer-length 24 \
    --chunksize 100000000 \
    --merge-chunks

# -----------------------------------------------------------------------------
# Quantify gene expression
# -----------------------------------------------------------------------------
# Pseudoquantify expression by matching index k-mers against reference sequences.
# Output is an h5ad file compatible with scanpy for downstream analysis.

malva quant \
    --index-in indices/pbmc_1k_v3 \
    --reference references/human_cdna_ncrna_masked.fa.gz \
    --folder-out quant/pbmc_1k_v3 \
    --h5ad \
    --pct-threshold 0.99 \
    --kmer-min 0 \
    --kmer-max 1000 \
    --sliding-size 90
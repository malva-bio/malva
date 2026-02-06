Examples
========

This section provides hands-on examples for analyzing single-cell RNA-seq data with Malva Tools.

We use the **1k Human PBMCs** dataset from 10x Genomics (v3 chemistry) as a standard benchmark dataset containing approximately 1,000 peripheral blood mononuclear cells.

Prerequisites
-------------

- Malva Tools installed via Python wheel (see :doc:`/installation`)
- scanpy, matplotlib, pandas, numpy, dnaio packages
- ~20 GB disk space

Tutorial Structure
------------------

Follow these examples in order:

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - Step
     - Section
     - Description
   * - 1
     - :ref:`example-build-index`
     - Download data, build the index, and quantify gene expression
   * - 2
     - :doc:`2_simple_analysis`
     - Load quantification results and perform standard single-cell analysis
   * - 3
     - :doc:`3_sequence_search`
     - Query arbitrary sequences against the index and visualize results

After completing these examples, you will have:

- A Malva index of the PBMC dataset
- Gene expression quantification in h5ad format
- UMAP visualizations with cell type clusters
- Custom sequence query results projected onto cells

.. _example-build-index:

Step 1: Build Index and Quantify
--------------------------------

First, create a working directory and download the required files:

.. code-block:: bash

   # Create directory structure
   mkdir -p malva_example/{barcodes,reads,references,indices,quant}
   cd malva_example

   # Download 10x cell barcode whitelist (v3 chemistry)
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/3M-february-2018.txt \
       -O barcodes/3M-february-2018.txt

   # Download human transcriptome reference (cDNA + ncRNA, repeat-masked)
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/human_cdna_ncrna_masked.fa.gz \
       -O references/human_cdna_ncrna_masked.fa.gz

   # Download PBMC 1k v3 sequencing reads
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/pbmc_1k_v3_S1_R1_001.fastq.gz \
       -O reads/pbmc_1k_v3_S1_R1_001.fastq.gz
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/pbmc_1k_v3_S1_R2_001.fastq.gz \
       -O reads/pbmc_1k_v3_S1_R2_001.fastq.gz

Build the Malva index from raw FASTQ reads:

.. code-block:: bash

   malva index \
       --reads-in reads/pbmc_1k_v3_S1_R1_001.fastq.gz reads/pbmc_1k_v3_S1_R2_001.fastq.gz \
       --flavor sc_10x_v3 \
       --spatial-bc-in barcodes/3M-february-2018.txt \
       --index-out indices/pbmc_1k_v3 \
       --kmer-length 24 \
       --chunksize 100000000 \
       --merge-chunks

Quantify gene expression by matching k-mers against the reference transcriptome:

.. code-block:: bash

   malva quant \
       --index-in indices/pbmc_1k_v3 \
       --reference references/human_cdna_ncrna_masked.fa.gz \
       --folder-out quant/pbmc_1k_v3 \
       --h5ad \
       --pct-threshold 0.99 \
       --kmer-min 0 \
       --kmer-max 1000 \
       --sliding-size 90

This produces ``quant/pbmc_1k_v3/pseudoquant.h5ad``, a scanpy-compatible file.

Now proceed to the notebooks below for downstream analysis.

.. toctree::
   :maxdepth: 1
   :caption: Analysis Notebooks

   2_simple_analysis
   3_sequence_search

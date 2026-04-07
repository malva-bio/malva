Examples
========

Hands-on tutorials for analyzing single-cell RNA-seq data with Malva.

We use the **1k Human PBMCs** dataset from 10x Genomics (v3 chemistry) as a standard benchmark containing approximately 1,000 peripheral blood mononuclear cells.

----

Tutorial Overview
-----------------

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: :octicon:`terminal;1.5em` Step 1: Build Index
        :link: example-build-index
        :link-type: ref
        :class-card: sd-border-0 sd-shadow-sm

        Download data, build the k-mer index, and quantify gene expression.

    .. grid-item-card:: :octicon:`graph;1.5em` Step 2: Analyze
        :link: 2_simple_analysis
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm

        Load results, cluster cells, and identify marker genes.

    .. grid-item-card:: :octicon:`search;1.5em` Step 3: Query
        :link: 3_sequence_search
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm

        Search for custom sequences and visualize results.

----

Prerequisites
-------------

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: :octicon:`package;1em` Malva
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Installed via wheel

        :doc:`/installation`

    .. grid-item-card:: :octicon:`code;1em` Python Packages
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        scanpy, matplotlib, pandas, numpy, dnaio

    .. grid-item-card:: :octicon:`database;1em` Disk Space
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        ~20 GB for example data

----

What You'll Learn
-----------------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: :octicon:`check-circle;1em` Build a Malva index
        :class-card: sd-border-0 sd-shadow-sm

    .. grid-item-card:: :octicon:`check-circle;1em` Quantify gene expression
        :class-card: sd-border-0 sd-shadow-sm

    .. grid-item-card:: :octicon:`check-circle;1em` Cluster cells and find markers
        :class-card: sd-border-0 sd-shadow-sm

    .. grid-item-card:: :octicon:`check-circle;1em` Query custom sequences
        :class-card: sd-border-0 sd-shadow-sm

----

.. _example-build-index:

Step 1: Build Index and Quantify
--------------------------------

**Download the example data**

.. code-block:: bash

   # Create directory structure
   mkdir -p malva_example/{barcodes,reads,references,indices,quant}
   cd malva_example

   # Download 10x cell barcode whitelist (v3 chemistry)
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/3M-february-2018.txt \
       -O barcodes/3M-february-2018.txt

   # Download human transcriptome reference
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/human_cdna_ncrna_masked.fa.gz \
       -O references/human_cdna_ncrna_masked.fa.gz

   # Download PBMC 1k v3 sequencing reads
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/pbmc_1k_v3_S1_R1_001.fastq.gz \
       -O reads/pbmc_1k_v3_S1_R1_001.fastq.gz
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/pbmc_1k_v3_S1_R2_001.fastq.gz \
       -O reads/pbmc_1k_v3_S1_R2_001.fastq.gz

**Build the index**

.. code-block:: bash

   malva index \
       --reads-in reads/pbmc_1k_v3_S1_R1_001.fastq.gz reads/pbmc_1k_v3_S1_R2_001.fastq.gz \
       --flavor sc_10x_v3 \
       --spatial-bc-in barcodes/3M-february-2018.txt \
       --index-out indices/pbmc_1k_v3 \
       --kmer-length 24 \
       --chunksize 100000000 \
       --merge-chunks

**Quantify gene expression**

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

.. tip::

   Output: ``quant/pbmc_1k_v3/pseudoquant.h5ad`` - a scanpy-compatible file ready for analysis.

----

Analysis Notebooks
------------------

Continue with the Jupyter notebooks below:

.. toctree::
   :maxdepth: 1

   2_simple_analysis
   3_sequence_search

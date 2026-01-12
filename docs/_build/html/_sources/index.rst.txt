Malva Tools
=========================

Build and query Malva indices on your own single-cell and spatial transcriptomics data, locally.

Availability
------------

Malva Tools are provided free of charge for academic non-profit research.

To request access, contact the Malva team.

Overview
--------

Malva Tools allow you to process your private datasets locally using the same algorithm that powers the public Malva Index. Build searchable indices from raw sequencing reads and perform sequence-level queries without uploading data to external servers.

**Core Commands**

- ``malva index``: Build a searchable k-mer index from FASTQ files
- ``malva quant``: Pseudoquantify gene expression against a reference
- ``malva show``: Generate spatial visualizations of query sequences
- ``malva serve``: Launch an interactive web interface for exploration


Quick Example
-------------

.. code-block:: bash

   # Build an index from 10x Genomics data
   malva index \
       --reads-in R1.fastq.gz R2.fastq.gz \
       --flavor sc_10x_v3 \
       --spatial-bc-in barcodes.txt \
       --index-out my_index

   # Quantify gene expression
   malva quant \
       --index-in my_index \
       --reference human_cdna.fa.gz \
       --folder-out output \
       --h5ad

Features
--------

- Process 100M reads in under 2 minutes
- Reference-free analysis: query any sequence without realignment
- Single-cell and spatial transcriptomics support
- Output compatible with scanpy and other standard tools
- Low memory footprint, suitable for laptop or HPC

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   about
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Subcommands & API

   cmd_index
   cmd_combine
   cmd_quant
   cmd_cellxmer
   cmd_show
   cmd_serve
   api/modules

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: Links

   Malva Platform <https://malva.bio>
   Malva Client <https://malva-bio.github.io/malva/client>
   GitHub Repository <https://github.com/malva-bio/malva>
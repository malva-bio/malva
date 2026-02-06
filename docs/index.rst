Malva Tools
===========

**Build and query sequence indices on your own single-cell and spatial transcriptomics data, locally.**

Malva Tools bring the power of the `Malva Platform <https://malva.bio>`_ to your private datasets. Process raw sequencing reads into searchable k-mer indices and perform sequence-level queries without uploading data to external servers.

.. note::

   Malva Tools are provided **free of charge** for academic non-profit research.
   See :doc:`installation` for download instructions.

----

Getting Started
---------------

.. grid:: 2

    .. grid-item-card:: Installation
        :link: installation
        :link-type: doc

        Download and set up Malva Tools on your system.

    .. grid-item-card:: Quick Start: Single-Cell
        :link: quickstart
        :link-type: doc

        Index and analyze 10x Genomics scRNA-seq data.

    .. grid-item-card:: Quick Start: Spatial
        :link: quickstart_spatial
        :link-type: doc

        Work with Open-ST, Visium, and other spatial platforms.

    .. grid-item-card:: Examples
        :link: examples/index
        :link-type: doc

        Jupyter notebooks with complete analysis workflows.

----

Core Commands
-------------

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - ``malva index``
     - Build a searchable k-mer index from FASTQ files
   * - ``malva quant``
     - Pseudoquantify gene expression against a reference transcriptome
   * - ``malva show``
     - Generate spatial visualizations of query sequences
   * - ``malva serve``
     - Launch an interactive web interface for exploration

----

Key Features
------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - **Fast**
     - Process 100M reads in under 2 minutes
   * - **Reference-free**
     - Query any sequence without realignment
   * - **Flexible**
     - Single-cell and spatial transcriptomics support
   * - **Compatible**
     - Output works with scanpy and standard tools
   * - **Lightweight**
     - Low memory footprint, runs on laptops or HPC

----

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

----

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Guides

   about
   installation
   quickstart
   quickstart_spatial

.. toctree::
   :maxdepth: 1
   :caption: Command Reference

   cmd_index
   cmd_quant
   cmd_show
   cmd_serve
   cmd_combine
   cmd_cellxmer

.. toctree::
   :maxdepth: 1
   :caption: Examples & API

   examples/index
   api/modules

----

Links
-----

- `Malva Platform <https://malva.bio>`_ - Search the public Malva Index
- `Malva Client <https://github.com/malva-bio/malva_client>`_ - Python client for the Malva API
- `GitHub Repository <https://github.com/malva-bio/malva>`_ - Source code and issues

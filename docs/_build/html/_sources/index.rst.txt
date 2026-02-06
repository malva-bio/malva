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
    :gutter: 3

    .. grid-item-card:: :octicon:`download;1em` Installation
        :link: installation
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm

        Download and set up Malva Tools on your system via Python wheel or Apptainer container.

    .. grid-item-card:: :octicon:`rocket;1em` Quick Start: Single-Cell
        :link: quickstart
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm

        Build your first index from 10x Genomics scRNA-seq data and quantify gene expression.

    .. grid-item-card:: :octicon:`image;1em` Quick Start: Spatial
        :link: quickstart_spatial
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm

        Work with Open-ST, Visium, Stereo-seq, and other spatial transcriptomics platforms.

    .. grid-item-card:: :octicon:`code;1em` Examples
        :link: examples/index
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm

        Jupyter notebooks with complete analysis workflows including clustering and visualization.

----

Core Commands
-------------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: :octicon:`database;1em` malva index
        :class-card: sd-border-0 sd-shadow-sm

        Build a searchable k-mer index from FASTQ files

    .. grid-item-card:: :octicon:`graph;1em` malva quant
        :class-card: sd-border-0 sd-shadow-sm

        Pseudoquantify gene expression against a reference transcriptome

    .. grid-item-card:: :octicon:`paintbrush;1em` malva show
        :class-card: sd-border-0 sd-shadow-sm

        Generate spatial visualizations of query sequences

    .. grid-item-card:: :octicon:`browser;1em` malva serve
        :class-card: sd-border-0 sd-shadow-sm

        Launch an interactive web interface for exploration

----

Key Features
------------

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: :octicon:`zap;1em` Fast
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Process 100M reads in under 2 minutes

    .. grid-item-card:: :octicon:`search;1em` Reference-free
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Query any sequence without realignment

    .. grid-item-card:: :octicon:`git-branch;1em` Flexible
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Single-cell and spatial transcriptomics

    .. grid-item-card:: :octicon:`package;1em` Compatible
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Works with scanpy and standard tools

    .. grid-item-card:: :octicon:`cpu;1em` Lightweight
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Runs on laptops or HPC clusters

    .. grid-item-card:: :octicon:`lock;1em` Private
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Your data stays on your machine

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

.. toctree::
   :maxdepth: 1
   :caption: Links

   Malva Platform <https://malva.bio>
   GitHub <https://github.com/malva-bio/malva>
   Malva Client <https://github.com/malva-bio/malva_client>

About Malva
=================

Malva is a tool to build sequence indices from single-cell omics data, enabling sequence search at single-cell resolution.

While the public `Malva Index <https://malva.bio>`_ provides access to a large corpus of pre-indexed data, Malva lets you apply the same technology to your own experiments without uploading data to external servers.

----

What Malva Can Do
-----------------------

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: :octicon:`database;1.5em` Build Indices
        :class-card: sd-border-0 sd-shadow-sm

        Create searchable k-mer indices from raw FASTQ files. Indices preserve cell barcode or spatial coordinate information for single-cell resolution queries.

    .. grid-item-card:: :octicon:`graph;1.5em` Quantify Expression
        :class-card: sd-border-0 sd-shadow-sm

        Pseudoquantify gene expression by matching index k-mers against reference sequences. Output is compatible with scanpy and standard single-cell workflows.

    .. grid-item-card:: :octicon:`search;1.5em` Query Sequences
        :class-card: sd-border-0 sd-shadow-sm

        Search for any nucleotide sequence across your indexed data. Find transcript isoforms, viral sequences, circular RNAs, splice junctions, or mutations.

    .. grid-item-card:: :octicon:`paintbrush;1.5em` Visualize Results
        :class-card: sd-border-0 sd-shadow-sm

        Generate spatial visualizations showing where query sequences are expressed. Supports both static images and interactive exploration.

----

Use Cases
---------

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: :octicon:`lock;1em` Private Analysis
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Analyze proprietary or unpublished datasets locally

    .. grid-item-card:: :octicon:`bug;1em` Novel Detection
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Detect sequences not in standard references (pathogens, vectors)

    .. grid-item-card:: :octicon:`verified;1em` Validation
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Validate findings from the public Malva Index in your data

    .. grid-item-card:: :octicon:`image;1em` Spatial Analysis
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Process spatial data with coordinate-level resolution

    .. grid-item-card:: :octicon:`tools;1em` Custom Indices
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Build indices for specific experimental designs

    .. grid-item-card:: :octicon:`beaker;1em` Discovery
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Find novel isoforms, circular RNAs, and splice junctions

----

Supported Platforms
-------------------

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: 10x Genomics
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Chromium (v1, v2, v3) and Visium

    .. grid-item-card:: Spatial Technologies
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Open-ST, Stereo-seq, Slide-seq

    .. grid-item-card:: Other Platforms
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Any barcode-based technology

----

Performance
-----------

.. grid:: 4
    :gutter: 2

    .. grid-item-card:: :octicon:`zap;1.5em`
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        **< 2 min**

        Index 100M reads

    .. grid-item-card:: :octicon:`stopwatch;1.5em`
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        **Seconds**

        Query millions of cells

    .. grid-item-card:: :octicon:`cpu;1.5em`
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        **Low Memory**

        Runs on workstations

    .. grid-item-card:: :octicon:`server;1.5em`
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        **Scalable**

        Works on HPC clusters

----

Availability
------------

.. note::

   Malva is provided **free of charge** for academic non-profit research.
   Any academic user with an ORCID account has free and unlimited access (within hardware constraints).

**To get started:**

1. Visit the `Malva Platform <https://malva.bio>`_ and sign in with your ORCID account
2. See the :doc:`quickstart` guide for your first index and query

----

Citation
--------

If you use Malva in your research, please cite `our paper in Nature <https://www.nature.com/articles/s41586-026-10975-w>`:

.. code-block:: text

   @article{LenPerin2026,
    title = {Ultrafast and reference-free sequence discovery in single-cell data},
    ISSN = {1476-4687},
    url = {http://dx.doi.org/10.1038/s41586-026-10975-w},
    DOI = {10.1038/s41586-026-10975-w},
    journal = {Nature},
    publisher = {Springer Science and Business Media LLC},
    author = {León-Periñán,  Daniel and Karaiskos,  Nikos and Rajewsky,  Nikolaus},
    year = {2026},
    month = Aug 
    }

About Malva Tools
=================

Overview
--------

Malva Tools are utilities for building and querying sequence indices from single-cell and spatial transcriptomics data. They enable reference-free analysis at nucleotide resolution on your own private datasets.

While the public Malva Index provides access to a large corpus of pre-indexed data, Malva Tools let you apply the same technology to your own experiments without uploading data to external servers.

What Malva Tools Can Do
-----------------------

**Build Indices**
    Create searchable k-mer indices from raw FASTQ files. Indices preserve cell barcode or spatial coordinate information for single-cell resolution queries.

**Quantify Expression**
    Pseudoquantify gene expression by matching index k-mers against reference sequences. Output is compatible with scanpy and standard single-cell workflows.

**Query Sequences**
    Search for any nucleotide sequence across your indexed data. Find transcript isoforms, viral sequences, circular RNAs, splice junctions, or mutations.

**Visualize Results**
    Generate spatial visualizations showing where query sequences are expressed. Supports both static images and interactive exploration.

Use Cases
---------

- Analyze proprietary or unpublished datasets locally
- Detect sequences not in standard references (novel isoforms, pathogens, vectors)
- Validate findings from the public Malva Index in your own data
- Process spatial transcriptomics data with coordinate-level resolution
- Build custom indices for specific experimental designs

Supported Data Types
--------------------

Malva Tools support common single-cell and spatial transcriptomics platforms:

- 10x Genomics Chromium (v1, v2, v3)
- 10x Genomics Visium
- Slide-seq / Curio Seeker
- Stereo-seq
- Open-ST
- Other technologies with barcode-based cell identification

Performance
-----------

Malva Tools are designed for efficiency:

- Index 100 million reads in under 2 minutes
- Query sequences across millions of cells in seconds
- Low memory footprint suitable for standard workstations
- Scales to large datasets on HPC systems

Availability
------------

Malva Tools are provided free of charge for academic non-profit research.

**To request access:**

1. Send an email to daniel.leonperinan@mdc-berlin.de
2. Include your name, institution, and intended use case
3. You will receive download instructions and a license agreement

Citation
--------

If you use Malva Tools in your research, please cite:

    [Citation to be added upon publication]
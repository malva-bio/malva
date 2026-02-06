Quick Start
===========

This guide walks through a complete workflow: building an index, quantifying expression, and querying sequences. We use the 10x Genomics 1k PBMC dataset (v3 chemistry) as an example.

The full example with notebooks is available in the `examples/malva_tools <https://github.com/malva-bio/malva/tree/main/examples/malva_tools>`_ folder.

Prerequisites
-------------

- Malva Tools installed via Python wheel or Apptainer (see :doc:`installation`)
- ~20 GB disk space for this example

Step 1: Prepare the Data
------------------------

Create a working directory and download the example files:

.. code-block:: bash

   # Create directory structure
   mkdir -p malva_example/{barcodes,reads,references,indices,quant}
   cd malva_example

   # Download cell barcode whitelist (10x v3 chemistry)
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/3M-february-2018.txt \
       -O barcodes/3M-february-2018.txt

   # Download reference transcriptome
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/human_cdna_ncrna_masked.fa.gz \
       -O references/human_cdna_ncrna_masked.fa.gz

   # Download sequencing reads
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/pbmc_1k_v3_S1_R1_001.fastq.gz \
       -O reads/pbmc_1k_v3_S1_R1_001.fastq.gz
   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/examples/malva_tools/pbmc_1k_v3_S1_R2_001.fastq.gz \
       -O reads/pbmc_1k_v3_S1_R2_001.fastq.gz

Step 2: Build the Index
-----------------------

Create a searchable k-mer index from the raw reads:

.. code-block:: bash

   malva index \
       --reads-in reads/pbmc_1k_v3_S1_R1_001.fastq.gz reads/pbmc_1k_v3_S1_R2_001.fastq.gz \
       --flavor sc_10x_v3 \
       --spatial-bc-in barcodes/3M-february-2018.txt \
       --index-out indices/pbmc_1k_v3 \
       --kmer-length 24 \
       --chunksize 100000000 \
       --merge-chunks

**Parameters explained:**

- ``--reads-in``: Input FASTQ files (R1 contains barcodes, R2 contains cDNA)
- ``--flavor``: Library chemistry (sc_10x_v3 for 10x Chromium v3)
- ``--spatial-bc-in``: Whitelist of valid cell barcodes
- ``--index-out``: Output directory for the index
- ``--kmer-length``: K-mer size (24 is recommended)
- ``--chunksize``: Reads per chunk (reduce if memory limited)
- ``--merge-chunks``: Combine chunks into final index

Runtime: 2-5 minutes for this dataset (on 1 CPU core).

Step 3: Quantify Gene Expression
--------------------------------

Pseudoquantify expression by matching k-mers against a reference:

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

**Parameters explained:**

- ``--index-in``: Path to the index from step 2
- ``--reference``: Reference sequences (transcriptome FASTA)
- ``--folder-out``: Output directory
- ``--h5ad``: Output in h5ad format (scanpy compatible)
- ``--pct-threshold``: Fraction of k-mers required for a match
- ``--sliding-size``: Window size for k-mer matching

Runtime: 10-60 minutes for this dataset (on 1 CPU core, depending on IO throughput of your machine).

Output: ``quant/pbmc_1k_v3/pseudoquant.h5ad``


Step 4: Query Custom Sequences
------------------------------

Search for any sequence in your indexed data using the Python API.

**Basic sequence search:**

.. code-block:: python

   from malva.index import MalvaIndex
   import pandas as pd

   # Open the index
   mindex = MalvaIndex("indices/pbmc_1k_v3")
   mindex.open()

   # Query a sequence directly (e.g., a transcript isoform or viral sequence)
   sequence = "ATGCAGTCGGGCACTCACTGGAGAGTTCTGGGCCTCTGCCTCTTATCAG..."

   # Search returns: cell indices, pseudocounts, and additional info
   locations, intensities, info = mindex.where(
       sequence,
       sliding_size=64,        # window size for k-mer matching
       pct_threshold=0.65,     # minimum fraction of matching k-mers
       count_at_most=100000,   # upper count threshold
       count_at_least=0,       # lower count threshold
   )

   mindex.close()

   # Convert to dataframe
   df_results = pd.DataFrame({
       "cell": locations,
       "expression": intensities
   })

**Query multiple sequences from a FASTA file:**

.. code-block:: python

   import dnaio

   mindex = MalvaIndex("indices/pbmc_1k_v3")
   mindex.open()

   # Load sequences from FASTA
   with dnaio.open("sequences.fa") as fasta:
       for record in fasta:
           locations, intensities, _ = mindex.where(
               record.sequence,
               sliding_size=64,
               pct_threshold=0.65,
           )
           print(f"{record.name}: found in {len(locations)} cells")

   mindex.close()

This returns the cells containing k-mers from your query sequence, along with pseudocount values.

Troubleshooting
---------------

**Apptainer cannot find files on HPC**

If you get ``FileNotFoundError`` for files that exist when using the Apptainer container, it may not have access to the required paths. Edit the ``malva`` wrapper script to bind additional paths:

.. code-block:: bash

   #!/bin/bash
   DIST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   exec apptainer exec \
       --bind "$(pwd):$(pwd)" \
       --bind "/data:/data" \
       --bind "$DIST:$DIST" \
       --env PYTHONPATH="$DIST/site-packages" \
       --pwd "$(pwd)" \
       "$DIST/python.sif" \
       python -m malva "$@"

Replace ``/data`` with the path to your data directory (e.g., ``/scratch``, ``/home``, or your institution's storage path).

**Symlinks not resolved**

Apptainer may not follow symlinks outside bound paths. Use the actual file paths (``readlink -f <symlink>``) or ensure the symlink target is also bound.

Next Steps
----------

- See :doc:`quickstart_spatial` for spatial transcriptomics workflows
- See the `examples folder <https://github.com/malva-bio/malva/tree/main/examples/malva_tools>`_ for complete Jupyter notebooks
- Check the command reference for all available options
- Try querying viral sequences, circular RNAs, or custom transcripts
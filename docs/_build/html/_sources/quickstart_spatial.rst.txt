Quick Start: Spatial Data
=========================

This guide covers indexing and analyzing spatial transcriptomics data with Malva Tools, using Open-ST as an example. The workflow applies to other spatial platforms (Visium, Stereo-seq, Slide-seq) with minor modifications.

Prerequisites
-------------

- Malva Tools installed (see :doc:`installation`)
- Spatial transcriptomics data with stitched coordinates
- An AnnData file (h5ad) containing barcodes and spatial coordinates

Step 1: Prepare Spatial Barcodes
--------------------------------

Malva requires a tab-separated barcode file with spatial coordinates. The file must have three columns:

1. **Barcode**: The nucleotide sequence (e.g., ``ACGTACGTACGTACGT``)
2. **X coordinate**: The x position (stitched/global coordinates)
3. **Y coordinate**: The y position (stitched/global coordinates)

**Expected file format:**

.. code-block:: text

   BC	X	Y
   AACAACCCATGCAACT	1234.56	5678.90
   AACAAGAAGATGTCGG	2345.67	6789.01
   AACAATCAGGTTCCTA	3456.78	7890.12
   ...

**Important notes:**

- Use stitched/global coordinates, not per-tile coordinates
- If your barcodes have suffixes (e.g., ``ACGT...:tile_1`` or ``ACGT..._L001``), extract only the sequence portion
- Coordinates should be from ``obsm['spatial']`` if using AnnData, not from ``obs['xcoord']``/``obs['ycoord']``

**Example Python script to generate the barcode file:**

.. code-block:: python

   import scanpy as sc
   import pandas as pd
   import numpy as np

   # Load your AnnData file with spatial coordinates
   adata = sc.read_h5ad("your_spatial_data.h5ad")

   # Extract barcodes - remove any suffixes after ':' or '_'
   # Adjust the split character based on your data format
   obs_names = list(map(str, adata.obs_names))
   barcodes = [name.split(":")[0] for name in obs_names]  # or split("_")[0]

   # Get spatial coordinates from obsm (stitched coordinates)
   if "spatial" not in adata.obsm_keys():
       raise KeyError("adata.obsm does not contain 'spatial' key")

   spatial = np.asarray(adata.obsm["spatial"])
   x_coords = spatial[:, 0]
   y_coords = spatial[:, 1]

   # Create and save the barcode file
   df = pd.DataFrame({
       "BC": barcodes,
       "X": x_coords,
       "Y": y_coords
   })
   df.to_csv("all_pucks.tsv", sep="\t", index=False)

Step 2: Build the Index
-----------------------

Index your spatial data using the ``openst`` flavor:

.. code-block:: bash

   malva index \
       --reads-in R1.fastq.gz R2.fastq.gz \
       --flavor openst \
       --spatial-bc-in all_pucks.tsv \
       --index-out my_spatial_index \
       --kmer-length 24 \
       --chunksize 100000000 \
       --merge-chunks

**Parameters explained:**

- ``--reads-in``: Input FASTQ files (R1 and R2)
- ``--flavor openst``: Library chemistry for Open-ST data
- ``--spatial-bc-in``: The barcode file with spatial coordinates from Step 1
- ``--index-out``: Output directory for the index
- ``--kmer-length 24``: K-mer size (24 is recommended)
- ``--chunksize``: Reads per chunk (reduce if memory limited)
- ``--merge-chunks``: Combine chunks into final index

Other available flavors include: ``visium``, ``stereoseq``, ``slideseq``.

Step 3: Query Sequences (Python API)
------------------------------------

Search for sequences of interest (e.g., bacterial 16S rRNA) using the Python API.

**Basic sequence search:**

.. code-block:: python

   from malva.index import MalvaIndex
   import pandas as pd

   # Open the spatial index
   mindex = MalvaIndex("my_spatial_index")
   mindex.open()

   # Query a sequence directly
   sequence = "ATGCAGTCGGGCACTCACTGGAGAGTTCTGGGCCTCTGCCTCTTATCAG..."

   # Search returns: spatial locations, pseudocounts, and additional info
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
       "location": locations,
       "expression": intensities
   })

**Query multiple sequences from a FASTA file:**

.. code-block:: python

   import dnaio

   mindex = MalvaIndex("my_spatial_index")
   mindex.open()

   results = {}
   with dnaio.open("bacteria_16S.fa") as fasta:
       for record in fasta:
           locations, intensities, _ = mindex.where(
               record.sequence,
               sliding_size=64,
               pct_threshold=0.65,
           )
           results[record.name] = (locations, intensities)
           print(f"{record.name}: found in {len(locations)} spatial locations")

   mindex.close()

Step 4: Visualize Spatial Distribution
--------------------------------------

**Command-line visualization:**

Use ``malva show`` to generate spatial plots directly:

.. code-block:: bash

   malva show \
       --index-in my_spatial_index \
       --query bacteria_16S.fa \
       --image-out output_images/

This generates TIFF images for each query sequence in the output folder.

**Python API visualization with MalvaPlot:**

For programmatic control over visualization, use the ``MalvaPlot`` class:

.. code-block:: python

   from malva.index import MalvaIndex
   from malva.show import MalvaPlot
   import matplotlib.pyplot as plt

   # Open the index and create a plotter
   mindex = MalvaIndex("my_spatial_index")
   plotter = MalvaPlot(mindex)

   # Query a sequence
   mindex.open()
   locations, intensities, _ = mindex.where(
       "ATGCAGTCGGGCACTCACTGGAGAGTTCTGGGCCTCTGCCTCTTATCAG...",
       sliding_size=64,
       pct_threshold=0.65,
   )
   mindex.close()

   # Generate a spatial image
   image = plotter.image(
       locations,
       intensities,
       render_scale=1,           # scale factor for output resolution
       render_smoothing=1.5,     # Gaussian smoothing sigma
       normalize=True            # normalize to 0-255 for display
   )

   # Display with matplotlib
   plt.figure(figsize=(10, 10))
   plt.imshow(image, cmap="viridis")
   plt.axis("off")
   plt.title("Spatial distribution of query sequence")
   plt.colorbar(label="Expression")
   plt.savefig("spatial_plot.png", dpi=150, bbox_inches="tight")
   plt.show()

**Save as TIFF for downstream analysis:**

.. code-block:: python

   import tifffile

   # Generate image without normalization for quantitative analysis
   image = plotter.image(locations, intensities, normalize=False)

   # Save as ImageJ-compatible TIFF
   tifffile.imwrite(
       "spatial_expression.tif",
       image,
       metadata={"axes": "YX"},
       imagej=True
   )

**Interactive exploration with malva serve:**

.. code-block:: bash

   malva serve \
       --index-in my_spatial_index \
       --port 5000

Then open http://localhost:5000 in your browser to explore spatial distributions interactively.

Troubleshooting
---------------

**Apptainer cannot find files on HPC**

If you get ``FileNotFoundError`` for files that exist, Apptainer may not have access to the required paths. Edit the ``malva`` wrapper script to bind additional paths:

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

Apptainer may not follow symlinks outside bound paths. Use the actual file paths or ensure the symlink target is also bound.

Next Steps
----------

- See :doc:`cmd_show` for all visualization options
- See :doc:`cmd_serve` for the interactive web interface
- Check the command reference for advanced options

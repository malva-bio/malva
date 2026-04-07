malva.barcodes module
=====================

.. note::

   This is a Cython extension module. API documentation is only available when
   Malva is built from source with Cython extensions compiled.

.. automodule:: malva.barcodes
   :members:
   :show-inheritance:
   :undoc-members:

The ``malva.barcodes`` module provides compiled barcode handling for spatial and
single-cell data.

Key components:

- **SpatialIndex** -- Spatial index mapping barcodes to coordinates.
- **create_spatial_index(barcode_file)** -- Create a spatial index from a tab-separated barcode file with (barcode, x, y) columns.
- **create_singlecell_index(barcode_file)** -- Create a barcode-only index for single-cell data from a whitelist file.
- **BackgroundModel** -- Background k-mer frequency model for statistical filtering.

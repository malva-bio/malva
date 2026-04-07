malva.indexes module
====================

.. note::

   This is a Cython extension module. API documentation is only available when
   Malva is built from source with Cython extensions compiled.

.. automodule:: malva.indexes
   :members:
   :show-inheritance:
   :undoc-members:

The ``malva.indexes`` module provides the core compiled indexing and query engine.

Key components:

- **PrefixIndex** -- Low-level prefix-bucketed k-mer index for disk-backed queries.
- **process_fastq_reads(...)** -- Stream FASTQ reads and extract (k-mer, cell-id) pairs into sorted chunks.
- **build_from_sorted_chunks(...)** -- Merge sorted chunk files into the final prefix-bucketed index (pi.bin, suffixes.bin, data.bin).
- **merge_prefix_indices(...)** -- Merge multiple per-sample prefix indices into a single combined index.
- **quantify_where(...)** -- Window-based sequence query scoring against a prefix index.

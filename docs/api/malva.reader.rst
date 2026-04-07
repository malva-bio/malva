malva.reader module
===================

.. note::

   This is a Cython extension module. API documentation is only available when
   Malva is built from source with Cython extensions compiled.

.. automodule:: malva.reader
   :members:
   :show-inheritance:
   :undoc-members:

The ``malva.reader`` module provides compiled FASTA/FASTQ file readers.

Key components:

- **iterate_fasta(filepath)** -- Iterate over records in a FASTA file, yielding (name, sequence) pairs.

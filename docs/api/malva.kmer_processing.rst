malva.kmer\_processing module
=============================

.. note::

   This is a Cython extension module. API documentation is only available when
   Malva is built from source with Cython extensions compiled.

.. automodule:: malva.kmer_processing
   :members:
   :show-inheritance:
   :undoc-members:

The ``malva.kmer_processing`` module provides compiled k-mer encoding and
decoding routines.

Key components:

- **decode_kmer(encoded, k)** -- Decode a 64-bit integer back to a nucleotide string of length k.
- **encode_kmer(sequence)** -- Encode a nucleotide string as a 64-bit integer using 2-bit encoding (A=00, C=01, G=10, T=11).

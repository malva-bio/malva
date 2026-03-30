# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

# cython: language_level=3

from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t

from libcpp.string cimport string

from libcpp.unordered_set cimport unordered_set
import numpy as np
cimport numpy as np
np.import_array()

cdef class FastKmerProcessor:
    cdef:
        readonly int kmer_size
        readonly bint overlapping
        vector[string] stored_sequences
        unordered_set[uint64_t] unique_kmers
        int min_valid_sequence_size

    cdef int process_sequence_chunk(self, const unsigned char* seq_ptr, Py_ssize_t length) except -1 nogil
    cdef np.ndarray process_sequences(self, sequences)

cdef class KmerFastqParser:
    cdef:
        Py_ssize_t buffer_size
        char *buffer
        Py_ssize_t bytes_in_buffer
        bint extra_newline
        bint eof
        object file
        char *record_start
        int kmer_size
        bint overlapping
        int jump_amount
    cdef readonly Py_ssize_t number_of_records

    cdef _read_into_buffer(self)
    cdef vector[uint64_t] next(self)

cdef class SequenceFastqParser:
    cdef:
        Py_ssize_t buffer_size
        char *buffer
        Py_ssize_t bytes_in_buffer
        bint extra_newline
        bint eof
        object file
        char *record_start
        int trim_start
        int trim_end
    cdef readonly Py_ssize_t number_of_records

    cdef _read_into_buffer(self)
    cdef uint64_t next(self)
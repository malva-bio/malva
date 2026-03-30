# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.string cimport string as cpp_string
from libc.stdint cimport uint16_t, uint32_t, int32_t, uint64_t

cdef class FastKmerExtractor:
    cdef:
        int kmer_size
        bint remove_noncomplex
        vector[uint64_t] kmers
        vector[cpp_string] sliding_seqs
        
    cdef void get_sliding_sequences(self, const char* sequence, int sliding_size) noexcept nogil
    cdef void process_sequence(self, const char* seq, int sliding_size) nogil
    cdef vector[uint64_t] process_sequence_group(self, list sequences, int sliding_size) except *
# cython: language_level=3

from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t

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
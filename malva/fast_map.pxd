# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint64_t, uint32_t

cdef extern from "ankerl/unordered_dense.h" namespace "ankerl::unordered_dense":
    cdef cppclass map[K, V]:
        ctypedef K key_type
        ctypedef V mapped_type
        
        map() except +
        V& operator[](const K&) except +
        bint empty() const
        size_t size() const
        void clear()
        void erase(const K&) except +
        
        cppclass iterator:
            pair[const K, V]& operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        
        iterator begin()
        iterator end()
        iterator find(const K&)

cdef class FastMapOfMap:
    cdef map[uint64_t, vector[uint32_t]] c_map
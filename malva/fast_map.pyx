# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t
from cython.operator cimport dereference as deref, preincrement as inc

cdef class FastMapOfMap:
    def __setitem__(self, uint64_t key, uint32_t value):
        self.add(key, value)

    def __getitem__(self, uint64_t key):
        cdef map[uint64_t, vector[uint32_t]].iterator it = self.c_map.find(key)
        if it == self.c_map.end():
            raise KeyError(key)
        return [x for x in deref(it).second]

    def __len__(self):
        return self.c_map.size()

    def __contains__(self, uint64_t key):
        return self.c_map.find(key) != self.c_map.end()

    def clear(self):
        self.c_map.clear()

    def items(self):
        cdef map[uint64_t, vector[uint32_t]].iterator it = self.c_map.begin()
        cdef map[uint64_t, vector[uint32_t]].iterator end = self.c_map.end()
        result = []
        while it != end:
            key = <uint64_t>deref(it).first
            value = [<uint32_t>x for x in deref(it).second]
            result.append((key, value))
            inc(it)
        return result

    def add(self, uint64_t key, uint32_t value):
        cdef map[uint64_t, vector[uint32_t]].iterator it = self.c_map.find(key)
        if it == self.c_map.end():
            self.c_map[key] = vector[uint32_t](1, value)
        else:
            deref(it).second.push_back(value)

    def extend(self, uint64_t key, values):
        for value in values:
            self.add(key, <uint32_t>value)
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.stdint cimport uint64_t
cimport numpy as np

cdef struct CachePage:
    void* data
    uint64_t page_number
    bint dirty

cdef class PageCache:
    cdef:
        CachePage* pages
        size_t n_pages
        size_t page_size
        object file  # Using object instead of FILE* for better Python integration
        str filename

    cdef void flush(self) except *
    cdef void _write_page(self, size_t cache_idx) except *
    cdef void* get_page(self, uint64_t page_number, bint for_writing=*) except *

cdef class PageAlignedArray:
    cdef:
        public Py_ssize_t _size
        public object _dtype
        Py_ssize_t _dtype_size
        np.dtype _np_dtype
        bint _mmap_mode
        object _mmap
        PageCache _cache
        str _filename
        Py_ssize_t _buffer_size  # Added for optimized writes

    cdef void _init_mmap(self, str filename, Py_ssize_t size) except *
    cdef void _get_slice_data(self, Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step, void* dest) except *
    cdef void _set_slice_data(self, Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step, void* src) except *
    cdef void _direct_write(self, void* src, Py_ssize_t start, Py_ssize_t size) except *

cdef class ChunkedStore:
    cdef:
        public dict _datasets
        public dict _attrs
        str _filename_base
        bint _mmap_mode
        Py_ssize_t _chunk_size

    cdef void _load_metadata(self) except *
    cdef PageAlignedArray _create_dataset(self, str name, tuple shape, object dtype) except *
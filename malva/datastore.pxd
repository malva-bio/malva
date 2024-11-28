# distutils: language = c++
from libc.stdint cimport uint64_t, uintptr_t
cimport numpy as np

cdef struct CachePage:
    void* data
    uint64_t page_number
    bint dirty

cdef struct PageCacheData:
    CachePage* pages
    size_t n_pages
    size_t page_size
    size_t dtype_size

cdef struct ArrayShape:
    Py_ssize_t* dims
    int ndim
    Py_ssize_t total_size

cdef class PageCache:
    cdef:
        PageCacheData* _data
        object file
        str filename

    cdef void flush(self) except *
    cdef void _write_page(self, size_t cache_idx) except *
    cdef void* get_page(self, uint64_t page_number, bint for_writing=*) except *
    cdef void _init_pages_nogil(self, CachePage* pages, size_t n_pages, size_t page_size) nogil
    cdef void _cleanup_pages_nogil(self, CachePage* pages, size_t n_pages) nogil
    cdef void _init_file(self, str filename, size_t page_size) except *

cdef class PageAlignedArray:
    cdef:
        ArrayShape _shape
        public object _dtype
        Py_ssize_t _dtype_size
        np.dtype _np_dtype
        bint _mmap_mode
        object _mmap
        PageCache _cache
        str _filename
        Py_ssize_t _read_buffer_size
        Py_ssize_t _write_buffer_size

    cdef void _init_mmap(self, str filename, Py_ssize_t size) except *
    cdef void _get_slice_data(self, Py_ssize_t* indices, Py_ssize_t size, void* dest) except *
    cdef void _set_slice_data(self, Py_ssize_t* indices, Py_ssize_t size, void* src) except *
    cdef void _get_strided_data(self, tuple key, void* dest) except *
    cdef void _set_strided_data(self, tuple key, np.ndarray value) except *
    cdef void _direct_write(self, void* src, Py_ssize_t* indices, Py_ssize_t size) except *
    cdef Py_ssize_t _compute_flat_index(self, Py_ssize_t* indices) nogil
    cdef void _init_shape(self, tuple shape) except *
    cdef tuple _get_shape(self)
    cpdef tuple get_shape(self)

cdef class ChunkedStore:
    cdef:
        public dict _datasets
        public dict _attrs
        str _filename_base
        bint _mmap_mode
        Py_ssize_t _chunk_size

    cdef void _load_metadata(self) except *
    cdef PageAlignedArray _create_dataset(self, str name, tuple shape, object dtype) except *
    cpdef bint contains(self, str key) except *
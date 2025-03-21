# distutils: language = c++
from libc.stdint cimport uint32_t, uint64_t, uintptr_t
from libcpp.unordered_map cimport unordered_map
cimport numpy as np

cdef struct CachePage:
    void* data
    uint64_t page_number
    bint dirty
    uint64_t last_access

cdef struct ArrayShape:
    Py_ssize_t* dims
    int ndim
    Py_ssize_t total_size

cdef class PageCache:
    cdef:
        CachePage* pages
        size_t n_pages
        size_t page_size
        uint64_t access_counter
        object file
        str filename
        unordered_map[uint64_t, size_t] page_map

    cdef void flush(self) except *
    cdef void _write_page(self, size_t cache_idx) except *
    cdef inline void* get_page(self, uint64_t page_number, bint for_writing=False) except *
    cdef void _init_pages_nogil(self, size_t n_pages, size_t page_size) nogil
    cdef void _cleanup_pages_nogil(self, size_t n_pages) nogil
    cdef void _init_file(self, str filename, size_t page_size) except *

cdef class PageAlignedArray:
    cdef:
        ArrayShape _shape
        public object _dtype
        Py_ssize_t _dtype_size
        object _np_dtype
        bint _mmap_mode
        object _mmap
        PageCache _cache
        EliasPageCache _elias_cache
        str _filename
        bint _compression_enabled
        Py_ssize_t _read_buffer_size
        Py_ssize_t _write_buffer_size

    cdef void _init_mmap(self, str filename, Py_ssize_t size) except *
    cdef void _get_slice_data(self, Py_ssize_t* indices, Py_ssize_t size, void* dest) nogil except *
    cdef void _set_slice_data(self, Py_ssize_t* indices, Py_ssize_t size, void* src) except *
    cdef void _set_contiguous_slice(self, tuple key, np.ndarray value) except *
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
    cdef PageAlignedArray _create_dataset(self, str name, tuple shape, object dtype, bint compression_enabled=*) except *
    cpdef bint contains(self, str key) except *


cdef struct CompressedBlock:
    uint64_t original_value
    uint32_t compressed_size

cdef struct EliasCachePage:
    void* data                  # Raw page data
    void* compressed_data       # Compressed page data
    uint64_t page_number       # Current page number
    size_t compressed_size     # Size of compressed data
    bint dirty                 # Whether page needs writing
    uint64_t last_access      # For LRU tracking
    bint is_compressed        # Compression state flag
    uint64_t first_value      # First value in page
    uint64_t last_value       # Last value in page

cdef class EliasPageCache:
    cdef:
        EliasCachePage* pages
        size_t n_pages
        size_t page_size
        uint64_t access_counter
        object file
        str filename
        unordered_map[uint64_t, size_t] page_map
        dict page_metadata
        bint compression_enabled
        
    cdef void _init_pages_nogil(self, size_t n_pages, size_t page_size) nogil
    cdef void _init_file(self, str filename, size_t page_size) except *
    cdef void _cleanup_all(self) nogil
    cdef void* get_compressed_page(self, uint64_t page_number, bint for_writing=*) except *
    cdef void _write_compressed_page(self, size_t cache_idx) except *
    cdef void flush(self) except *
    cdef inline uint32_t encode_value(self, uint64_t value, char* dest) nogil
    cdef inline uint64_t decode_value(self, const char* src, uint32_t* bytes_read) nogil
    cdef size_t compress_page(self, void* src, void* dest, size_t size, uint64_t prev_value) nogil
    cdef size_t decompress_page(self, void* src, void* dest, size_t compressed_size, uint64_t prev_value) nogil
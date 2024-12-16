# distutils: language = c++
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport cython
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t, uint64_t
import mmap as py_mmap
import os
import logging

np.import_array()

DEF PAGE_SIZE = 65536#4096
DEF DEFAULT_CHUNK_SIZE = 16384
DEF CACHE_SIZE = 1024
DEF WRITE_BUFFER_SIZE = 1024 * 1024 * 4
DEF FREAD_BUFFER_SIZE = 4096
DEF MAX_DIMS = 32

cdef extern from "numpy/arrayobject.h":
    void* PyArray_DATA(np.ndarray arr) nogil
    np.ndarray PyArray_SimpleNew(int nd, np.npy_intp* dims, int typenum) nogil

cdef extern from *:
    """
    #include <x86intrin.h>
    static inline int count_leading_zeros(unsigned long long x) {
        return __builtin_clzll(x);
    }
    """
    int count_leading_zeros(unsigned long long x) nogil

cdef class PageCache:
    def __cinit__(self, str filename, size_t page_size=PAGE_SIZE, size_t n_pages=CACHE_SIZE):
        self.page_size = page_size
        self.n_pages = n_pages
        self.access_counter = 0
        self.filename = filename

        self.page_map = unordered_map[uint64_t, size_t]()

        self.pages = <CachePage*>malloc(n_pages * sizeof(CachePage))
        if not self.pages:
            raise MemoryError()
        
        self._init_pages_nogil(n_pages, page_size)
        self._init_file(filename, page_size)

    def __dealloc__(self):
        if self.pages:
            self._cleanup_pages_nogil(self.n_pages)
            free(self.pages)
        
        if self.file is not None:
            self.file.flush()
            self.file.close()

    cdef void _cleanup_pages_nogil(self, size_t n_pages) nogil:
        cdef size_t i
        for i in range(n_pages):
            if self.pages[i].data != NULL:
                free(self.pages[i].data)

    cdef void _init_file(self, str filename, size_t page_size) except *:
        if os.path.exists(filename):
            self.file = open(filename, 'rb+')
            data = self.file.read(page_size)
            if data:
                memcpy(self.pages[0].data, <char*>data, len(data))
        else:
            self.file = open(filename, 'wb+')
            self.file.write(b'\0' * page_size)
            self.file.flush()

    cdef void flush(self) except *:
        cdef size_t i
        if self.file is not None:
            for i in range(self.n_pages):
                if self.pages[i].dirty:
                    self._write_page(i)
                    self.pages[i].dirty = False
            self.file.flush()

    cdef void _write_page(self, size_t cache_idx) except *:
        cdef:
            size_t offset = self.pages[cache_idx].page_number * self.page_size
            bytes data = bytes((<char[:self.page_size]>self.pages[cache_idx].data)[:self.page_size])
        
        self.file.seek(offset)
        self.file.write(data)

    cdef void _init_pages_nogil(self, size_t n_pages, size_t page_size) nogil:
        cdef size_t i
        for i in range(n_pages):
            self.pages[i].data = malloc(page_size)
            if self.pages[i].data == NULL:
                with gil:
                    raise MemoryError()
            memset(self.pages[i].data, 0, page_size)
            self.pages[i].page_number = 0
            self.pages[i].dirty = False
            self.pages[i].last_access = 0

    cdef inline void* get_page(self, uint64_t page_number, bint for_writing=False) except *:
        cdef:
            size_t i, lru_idx
            uint64_t oldest_access
            bytes data
            
        # Fast lookup with nogil where possible
        with nogil:
            # Use map lookup first
            if self.page_map.count(page_number):
                lru_idx = self.page_map[page_number]
                if for_writing:
                    self.pages[lru_idx].dirty = True
                self.pages[lru_idx].last_access = self.access_counter
                self.access_counter += 1
                return self.pages[lru_idx].data

            # Cache miss - find LRU
            oldest_access = self.access_counter
            for i in range(self.n_pages):
                if self.pages[i].last_access < oldest_access:
                    oldest_access = self.pages[i].last_access
                    lru_idx = i

            # Remove old page from map
            if self.page_map.count(self.pages[lru_idx].page_number):
                self.page_map.erase(self.pages[lru_idx].page_number)

        # Handle file I/O (requires GIL)
        if self.pages[lru_idx].dirty:
            self._write_page(lru_idx)
        
        self.pages[lru_idx].page_number = page_number
        self.pages[lru_idx].dirty = for_writing
        self.pages[lru_idx].last_access = self.access_counter
        self.access_counter += 1
        self.page_map[page_number] = lru_idx
        
        # Use buffered read for better I/O performance
        cdef size_t bytes_read
        with nogil:
            memset(self.pages[lru_idx].data, 0, self.page_size)
            
        self.file.seek(page_number * self.page_size)
        data = self.file.read(self.page_size)
        if data:
            memcpy(self.pages[lru_idx].data, <char*>data, len(data))
            
        return self.pages[lru_idx].data

cdef class PageAlignedArray:
    def __cinit__(self, tuple shape, dtype='uint64', bint mmap_mode=False, str filename=None):
        self._dtype = np.dtype(dtype)
        self._np_dtype = self._dtype
        self._dtype_size = self._dtype.itemsize
        self._mmap_mode = mmap_mode
        self._filename = filename
        self._read_buffer_size = FREAD_BUFFER_SIZE
        self._write_buffer_size = WRITE_BUFFER_SIZE
        
        if filename is None:
            raise ValueError("Filename required for both mmap and non-mmap modes")
        
        self._init_shape(shape)
        aligned_size = (self._shape.total_size * self._dtype_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)
        
        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                f.seek(aligned_size - 1)
                f.write(b'\0')
        
        if mmap_mode:
            self._init_mmap(filename, aligned_size)
        else:
            self._cache = PageCache(filename, PAGE_SIZE)

    cdef void _init_mmap(self, str filename, Py_ssize_t size) except *:
        with open(filename, 'r+b') as f:
            self._mmap = py_mmap.mmap(f.fileno(), size, access=py_mmap.ACCESS_WRITE)

    cdef Py_ssize_t _compute_flat_index(self, Py_ssize_t* indices) nogil:
        cdef:
            Py_ssize_t flat_idx = 0
            Py_ssize_t stride = 1
            int i
        
        for i in range(self._shape.ndim - 1, -1, -1):
            flat_idx += indices[i] * stride
            stride *= self._shape.dims[i]
        
        return flat_idx

    cdef void _init_shape(self, tuple shape) except *:
        cdef int i
        self._shape.ndim = len(shape)
        self._shape.dims = <Py_ssize_t*>malloc(self._shape.ndim * sizeof(Py_ssize_t))
        if not self._shape.dims:
            raise MemoryError()
        
        self._shape.total_size = 1
        for i in range(self._shape.ndim):
            self._shape.dims[i] = shape[i]
            self._shape.total_size *= shape[i]

    cdef tuple _get_shape(self):
        """
        Get the shape of the array as a tuple.
        """
        return tuple([self._shape.dims[i] for i in range(self._shape.ndim)])

    cpdef tuple get_shape(self):
        """
        Public method to get the shape
        """
        return self._get_shape()

    @property 
    def shape(self):
        """
        Public property to access array shape
        """
        return self._get_shape()

    cdef void _direct_write(self, void* src, Py_ssize_t* indices, Py_ssize_t size) except *:
        cdef:
            Py_ssize_t flat_idx = self._compute_flat_index(indices)
            Py_ssize_t offset = flat_idx * self._dtype_size
            size_t bytes_to_write = size * self._dtype_size
            
        if self._mmap_mode:
            memcpy(<char*>(<uintptr_t>self._mmap.buf) + offset, src, bytes_to_write)
            self._mmap.flush()
        else:
            data = bytes((<char[:bytes_to_write]>src)[:bytes_to_write])
            with open(self._filename, 'rb+') as f:
                f.seek(offset)
                f.write(data)

    cdef void _get_slice_data(self, Py_ssize_t* indices, Py_ssize_t size, void* dest) nogil except *:
        cdef:
            Py_ssize_t flat_idx = self._compute_flat_index(indices)
            Py_ssize_t offset = flat_idx * self._dtype_size
            size_t bytes_to_read = size * self._dtype_size
            uint64_t page_number
            size_t offset_in_page
            size_t bytes_remaining
            size_t bytes_this_page
            void* page_data
            char* dest_ptr = <char*>dest
            
        if self._mmap_mode:
            # TODO: we don't support mmap yet
            return

        # For large reads, use direct file access
        if bytes_to_read >= self._read_buffer_size:
            with gil:
                self._cache.file.seek(offset)
                data = self._cache.file.read(bytes_to_read)
                if data:
                    memcpy(dest, <char*>data, len(data))
            return

        # Optimized page-based reading
        page_number = offset // PAGE_SIZE
        offset_in_page = offset % PAGE_SIZE
        bytes_remaining = bytes_to_read
        
        while bytes_remaining > 0:
            with gil:
                page_data = self._cache.get_page(page_number, False)
            
            bytes_this_page = min(PAGE_SIZE - offset_in_page, bytes_remaining)
            memcpy(dest_ptr, <char*>page_data + offset_in_page, bytes_this_page)
            
            bytes_remaining -= bytes_this_page
            dest_ptr += bytes_this_page
            page_number += 1
            offset_in_page = 0

    cdef void _set_slice_data(self, Py_ssize_t* indices, Py_ssize_t size, void* src) except *:
        cdef:
            Py_ssize_t flat_idx = self._compute_flat_index(indices)
            Py_ssize_t offset = flat_idx * self._dtype_size
            size_t bytes_to_write = size * self._dtype_size
            
        # Fast path for large contiguous writes
        if bytes_to_write >= self._write_buffer_size:
            if self._mmap_mode:
                # TODO: we don't support mmap
                return
            else:
                data = bytes((<char[:bytes_to_write]>src)[:bytes_to_write])
                with open(self._filename, 'rb+') as f:
                    f.seek(offset)
                    f.write(data)
            return

        # Page-based writing for smaller or non-contiguous access
        cdef:
            uint64_t page_number = offset // PAGE_SIZE
            size_t offset_in_page = offset % PAGE_SIZE
            size_t bytes_remaining = bytes_to_write
            size_t bytes_this_page
            void* page_data
            
        while bytes_remaining > 0:
            page_data = self._cache.get_page(page_number, True)
            bytes_this_page = min(PAGE_SIZE - offset_in_page, bytes_remaining)
            
            memcpy(
                <char*>page_data + offset_in_page,
                <char*>src + (bytes_to_write - bytes_remaining),
                bytes_this_page
            )
            
            bytes_remaining -= bytes_this_page
            page_number += 1
            offset_in_page = 0

    def flush(self):
        """Ensure all changes are written to disk"""
        if self._mmap_mode:
            self._mmap.flush()
        else:
            self._cache.flush()

    def __getitem__(self, key):
        cdef:
            Py_ssize_t[MAX_DIMS] indices
            np.ndarray result
            np.npy_intp dims[MAX_DIMS]
            void* result_data
            int i
            Py_ssize_t total_size = 1
            bint is_contiguous = True
            
        # Handle different key types
        if isinstance(key, (int, slice)):
            # Convert single index/slice to tuple for consistent handling
            key = (key,)
            
            # For 1D access of multi-dimensional array, reshape the access
            if self._shape.ndim > 1:
                # Create a flattened view
                total_elements = self._shape.total_size
                if isinstance(key[0], int):
                    if key[0] >= total_elements:
                        raise IndexError("Index out of bounds")
                    indices[0] = key[0]
                    total_size = 1
                else:
                    start = key[0].start if key[0].start is not None else 0
                    stop = key[0].stop if key[0].stop is not None else total_elements
                    step = key[0].step if key[0].step is not None else 1
                    if start >= total_elements or (stop is not None and stop > total_elements):
                        raise IndexError("Index out of bounds")
                    indices[0] = start
                    total_size = (stop - start) // step
                    if step != 1:
                        is_contiguous = False
                
                # Create 1D result array
                dims[0] = 1 if isinstance(key[0], int) else total_size
                ndim = 1
                
            else:
                # Normal 1D array handling
                if isinstance(key[0], int):
                    if key[0] >= self._shape.dims[0]:
                        raise IndexError("Index out of bounds")
                    indices[0] = key[0]
                    total_size = 1
                else:
                    start = key[0].start if key[0].start is not None else 0
                    stop = key[0].stop if key[0].stop is not None else self._shape.dims[0]
                    step = key[0].step if key[0].step is not None else 1
                    if start >= self._shape.dims[0] or (stop is not None and stop > self._shape.dims[0]):
                        raise IndexError("Index out of bounds")
                    indices[0] = start
                    total_size = (stop - start) // step
                    if step != 1:
                        is_contiguous = False
                
                dims[0] = 1 if isinstance(key[0], int) else total_size
                ndim = 1
                
        elif isinstance(key, tuple):
            # Multi-dimensional indexing
            if len(key) > self._shape.ndim:
                raise IndexError(f"Too many indices: got {len(key)}, maximum allowed is {self._shape.ndim}")
                
            ndim = len(key)
            stride = 1
            for i in range(ndim):
                if isinstance(key[i], slice):
                    start = key[i].start if key[i].start is not None else 0
                    stop = key[i].stop if key[i].stop is not None else self._shape.dims[i]
                    step = key[i].step if key[i].step is not None else 1
                    if start >= self._shape.dims[i] or (stop is not None and stop > self._shape.dims[i]):
                        raise IndexError("Index out of bounds")
                    dims[i] = (stop - start) // step
                    indices[i] = start
                    total_size *= dims[i]
                    if step != 1:
                        is_contiguous = False
                else:
                    if key[i] >= self._shape.dims[i]:
                        raise IndexError("Index out of bounds")
                    indices[i] = key[i]
                    dims[i] = 1
                    
            # Fill remaining dimensions with full slices if any
            for i in range(ndim, self._shape.ndim):
                dims[i] = self._shape.dims[i]
                indices[i] = 0
                total_size *= dims[i]
        else:
            raise IndexError("Invalid index type")
            
        # Create the result array
        result = PyArray_SimpleNew(ndim, dims, self._np_dtype.num)
        result_data = PyArray_DATA(result)
        
        if is_contiguous:
            self._get_slice_data(indices, total_size, result_data)
        else:
            self._get_strided_data(key, result_data)
        
        return np.squeeze(result)

    def __setitem__(self, key, value):
        cdef:
            Py_ssize_t[MAX_DIMS] indices
            np.ndarray arr
            void* arr_data
            int i, dim
            Py_ssize_t total_size = 1
            bint is_contiguous = True
            
        # Convert single index/slice to tuple
        if not isinstance(key, tuple):
            key = (key,)
            
        # Convert value to numpy array if it isn't already
        arr = np.ascontiguousarray(value, dtype=self._dtype)
        
        # Check dimensions
        if len(key) > self._shape.ndim:
            raise IndexError(f"Too many indices: got {len(key)}, maximum allowed is {self._shape.ndim}")
            
        # Calculate total size and validate slices
        cdef list slice_dims = []
        for i, k in enumerate(key):
            if isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else self._shape.dims[i]
                step = k.step if k.step is not None else 1
                
                # Handle negative indices
                if start < 0:
                    start += self._shape.dims[i]
                if stop < 0:
                    stop += self._shape.dims[i]
                
                # Validate bounds
                if start < 0 or start >= self._shape.dims[i]:
                    raise IndexError(f"Start index {start} out of bounds for axis {i}")
                if stop < 0 or stop > self._shape.dims[i]:
                    raise IndexError(f"Stop index {stop} out of bounds for axis {i}")
                
                dim_size = (stop - start + step - 1) // step
                slice_dims.append(dim_size)
                total_size *= dim_size
                
                if step != 1:
                    is_contiguous = False
            else:
                # Handle integer index
                idx = k if k >= 0 else k + self._shape.dims[i]
                if idx < 0 or idx >= self._shape.dims[i]:
                    raise IndexError(f"Index {k} out of bounds for axis {i}")
                slice_dims.append(1)
        
        # Add remaining dimensions
        for i in range(len(key), self._shape.ndim):
            slice_dims.append(self._shape.dims[i])
            total_size *= self._shape.dims[i]
        
        # Validate input array shape
        expected_shape = tuple(d for d in slice_dims if d != 1)
        if arr.size != total_size:
            raise ValueError(f"Cannot assign array of size {arr.size} to slice of size {total_size}")
        
        # Handle multi-dimensional assignment
        if is_contiguous:
            # Fast path for contiguous data
            self._set_contiguous_slice(key, arr)
        else:
            # Fallback for non-contiguous slices
            self._set_strided_slice(key, arr)

    cdef void _set_contiguous_slice(self, tuple key, np.ndarray value) except *:
        cdef:
            Py_ssize_t start, stop, step
            Py_ssize_t offset = 0
            void* src_data = PyArray_DATA(value)
            size_t bytes_to_write = value.size * self._dtype_size
            size_t stride = 1
            int i
            uint64_t byte_offset
            uint64_t page_number
            size_t page_offset
            size_t bytes_this_page
            void* page_data
            size_t bytes_written = 0
            
        # Calculate flat offset for the start of the slice
        for i in range(len(key) - 1, -1, -1):
            if isinstance(key[i], slice):
                start = key[i].start if key[i].start is not None else 0
                if start < 0:
                    start += self._shape.dims[i]
                offset += start * stride
            else:
                idx = key[i] if key[i] >= 0 else key[i] + self._shape.dims[i]
                offset += idx * stride
            stride *= self._shape.dims[i]
        
        byte_offset = offset * self._dtype_size
        
        # Sanity check
        if byte_offset + bytes_to_write > self._shape.total_size * self._dtype_size:
            raise IndexError("Write operation would exceed array bounds")
            
        # For large contiguous writes, use direct file I/O
        if bytes_to_write >= self._write_buffer_size:
            # Flush any existing pages that overlap with our write range
            start_page = byte_offset // PAGE_SIZE
            end_page = (byte_offset + bytes_to_write - 1) // PAGE_SIZE + 1
            
            for page_num in range(start_page, end_page):
                if self._cache.page_map.count(page_num):
                    idx = self._cache.page_map[page_num]
                    if self._cache.pages[idx].dirty:
                        self._cache._write_page(idx)
                    self._cache.page_map.erase(page_num)
            
            # Write directly to file
            data = bytes((<char[:bytes_to_write]>src_data)[:bytes_to_write])
            with open(self._filename, 'rb+') as f:
                f.seek(byte_offset)
                f.write(data)
                
        else:
            # Use page cache for smaller writes
            while bytes_written < bytes_to_write:
                page_number = (byte_offset + bytes_written) // PAGE_SIZE
                page_offset = (byte_offset + bytes_written) % PAGE_SIZE
                
                bytes_this_page = min(
                    PAGE_SIZE - page_offset,
                    bytes_to_write - bytes_written
                )
                
                page_data = self._cache.get_page(page_number, True)
                
                with nogil:
                    memcpy(
                        <char*>page_data + page_offset,
                        <char*>src_data + bytes_written,
                        bytes_this_page
                    )
                
                bytes_written += bytes_this_page

    cdef void _get_strided_data(self, tuple key, void* dest) except *:
        cdef:
            Py_ssize_t[MAX_DIMS] starts, stops, steps, position
            int i
            np.ndarray temp_arr
            void* temp_data
            Py_ssize_t flat_idx, total_elements = 1
            void* page_data
            Py_ssize_t dest_idx = 0
            
        # Parse slices
        for i in range(self._shape.ndim):
            if isinstance(key[i], slice):
                starts[i] = key[i].start if key[i].start is not None else 0
                stops[i] = key[i].stop if key[i].stop is not None else self._shape.dims[i]
                steps[i] = key[i].step if key[i].step is not None else 1
                total_elements *= (stops[i] - starts[i]) // steps[i]
            else:
                starts[i] = stops[i] = key[i]
                steps[i] = 1
        
        # Initialize position array
        for i in range(self._shape.ndim):
            position[i] = starts[i]
        
        # Create temporary array for the result
        temp_arr = np.zeros(total_elements, dtype=self._np_dtype)
        temp_data = PyArray_DATA(temp_arr)
        
        dest_idx = 0
        while True:
            # Read single element
            flat_idx = self._compute_flat_index(position)
            if self._mmap_mode:
                memcpy(
                    <char*>dest + dest_idx * self._dtype_size,
                    <char*>(<uintptr_t>self._mmap.buf) + flat_idx * self._dtype_size,
                    self._dtype_size
                )
            else:
                page_data = self._cache.get_page(
                    (flat_idx * self._dtype_size) // PAGE_SIZE, False
                )
                memcpy(
                    <char*>dest + dest_idx * self._dtype_size,
                    <char*>page_data + (flat_idx * self._dtype_size) % PAGE_SIZE,
                    self._dtype_size
                )
            
            dest_idx += 1
            
            # Update position
            for i in range(self._shape.ndim - 1, -1, -1):
                position[i] += steps[i]
                if position[i] < stops[i]:
                    break
                if i > 0:
                    position[i] = starts[i]
            else:
                break

    cdef void _set_strided_data(self, tuple key, np.ndarray value) except *:
        cdef:
            Py_ssize_t[MAX_DIMS] starts, stops, steps, position
            int i, ndim
            void* src_data = PyArray_DATA(value)
            Py_ssize_t flat_idx, total_elements = 1
            Py_ssize_t src_idx = 0
            void* page_data
            size_t page_number, offset
            
        # Initialize arrays
        for i in range(MAX_DIMS):
            starts[i] = 0
            stops[i] = 0
            steps[i] = 1
            position[i] = 0
            
        ndim = min(len(key), self._shape.ndim)
        
        # Parse slices with bounds checking
        for i in range(ndim):
            if isinstance(key[i], slice):
                starts[i] = key[i].start if key[i].start is not None else 0
                stops[i] = key[i].stop if key[i].stop is not None else self._shape.dims[i]
                steps[i] = key[i].step if key[i].step is not None else 1
                
                # Normalize negative indices
                if starts[i] < 0:
                    starts[i] += self._shape.dims[i]
                if stops[i] < 0:
                    stops[i] += self._shape.dims[i]
                    
                # Validate bounds
                if starts[i] < 0 or starts[i] >= self._shape.dims[i]:
                    raise IndexError(f"Start index out of bounds for dimension {i}")
                if stops[i] < 0 or stops[i] > self._shape.dims[i]:
                    raise IndexError(f"Stop index out of bounds for dimension {i}")
                    
                total_elements *= (stops[i] - starts[i] + steps[i] - 1) // steps[i]
            else:
                # Handle integer index
                idx = key[i]
                if idx < 0:
                    idx += self._shape.dims[i]
                if idx < 0 or idx >= self._shape.dims[i]:
                    raise IndexError(f"Index out of bounds for dimension {i}")
                    
                starts[i] = stops[i] = idx
                steps[i] = 1
                
        # Fill remaining dimensions
        for i in range(ndim, self._shape.ndim):
            starts[i] = 0
            stops[i] = self._shape.dims[i]
            steps[i] = 1
            total_elements *= self._shape.dims[i]
            
        # Initialize position
        for i in range(self._shape.ndim):
            position[i] = starts[i]
            
        # Write data with bounds checking
        while True:
            flat_idx = self._compute_flat_index(position)
            
            if self._mmap_mode:
                if flat_idx * self._dtype_size >= self._mmap.size():
                    raise IndexError("Array index out of bounds")
                memcpy(
                    <char*>(<uintptr_t>self._mmap.buf) + flat_idx * self._dtype_size,
                    <char*>src_data + src_idx * self._dtype_size,
                    self._dtype_size
                )
            else:
                page_number = (flat_idx * self._dtype_size) // PAGE_SIZE
                offset = (flat_idx * self._dtype_size) % PAGE_SIZE
                
                # Ensure we don't write past the end of a page
                if offset + self._dtype_size > PAGE_SIZE:
                    raise IndexError("Invalid page access")
                    
                page_data = self._cache.get_page(page_number, True)
                
                with nogil:
                    memcpy(
                        <char*>page_data + offset,
                        <char*>src_data + src_idx * self._dtype_size,
                        self._dtype_size
                    )
            
            src_idx += 1
            if src_idx >= value.size:
                break
                
            # Update position with bounds checking
            for i in range(self._shape.ndim - 1, -1, -1):
                position[i] += steps[i]
                if position[i] < stops[i]:
                    break
                if i > 0:  # Don't reset position[0] if we're done
                    position[i] = starts[i]
            else:
                break  # We've processed all elements
        
        if self._mmap_mode:
            self._mmap.flush()

    def __len__(self):
        """Return the total size of the array"""
        return self._shape.total_size

cdef class ChunkedStore:
    def __cinit__(self, str filename_base, str mode='r', bint mmap_mode=False, Py_ssize_t chunk_size=DEFAULT_CHUNK_SIZE):
        self._filename_base = filename_base
        self._mmap_mode = mmap_mode
        self._chunk_size = chunk_size
        self._datasets = {}
        self._attrs = {}
        
        if mode == 'r' and os.path.exists(f"{filename_base}.meta"):
            self._load_metadata()

    cdef void _load_metadata(self) except *:
        import pickle
        with open(f"{self._filename_base}.meta", 'rb') as f:
            metadata = pickle.load(f)
            self._attrs = metadata['attrs']
            
            for name, info in metadata['datasets'].items():
                self._datasets[name] = PageAlignedArray(
                    info['shape'],  # Now using shape instead of size
                    info['dtype'],
                    self._mmap_mode,
                    f"{self._filename_base}_{name}.dat"
                )
                
    cdef PageAlignedArray _create_dataset(self, str name, tuple shape, object dtype) except *:
        cdef PageAlignedArray arr
        
        # Validate shape
        if not shape:
            raise ValueError("Shape cannot be empty")
        for dim in shape:
            if dim <= 0:
                raise ValueError("All dimensions must be positive")

        arr = PageAlignedArray(
            shape,  # Pass the full shape tuple
            dtype,
            self._mmap_mode,
            f"{self._filename_base}_{name}.dat"
        )
        self._datasets[name] = arr
        return arr

    def create_dataset(self, str name, tuple shape, dtype='float64', **kwargs):
        """
        Create a new dataset with the specified shape and type.
        
        Parameters:
        -----------
        name : str
            Name of the dataset
        shape : tuple
            Shape of the dataset (e.g., (100, 200) for 2D array)
        dtype : str or numpy.dtype
            Data type of the dataset
        **kwargs : dict
            Additional arguments (for compatibility)
            
        Returns:
        --------
        PageAlignedArray
            The newly created dataset
        """
        return self._create_dataset(name, shape, dtype)
        
    def __getitem__(self, name):
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found")
        return self._datasets[name]

    def __setitem__(self, name, value):
        if isinstance(value, np.ndarray):
            if name not in self._datasets:
                self.create_dataset(name, value.shape, value.dtype)
            self._datasets[name][:] = value
        else:
            raise TypeError("Value must be a numpy array")

    def __iter__(self):
        """Make ChunkedStore iterable."""
        return iter(self._datasets)
        
    def __len__(self):
        """Return number of datasets."""
        return len(self._datasets)

    def keys(self):
        """Return dataset keys."""
        return self._datasets.keys()
        
    def values(self):
        """Return dataset values."""
        return self._datasets.values()

    def items(self):
        """Return dataset items."""
        return self._datasets.items()

    def __contains__(self, key):
        """Enable 'in' operator."""
        return self.contains(key)
        
    cpdef bint contains(self, str key) except *:
        """Check if dataset exists."""
        return key in self._datasets
            
    @property
    def attrs(self):
        return self._attrs
        
    def flush(self):
        """
        Flush all changes to disk and save metadata
        """
        import pickle
        
        # First flush all datasets
        for dataset in self._datasets.values():
            dataset.flush()
        
        # Prepare and save metadata
        metadata = {
            'attrs': self._attrs,
            'datasets': {
                name: {
                    'shape': arr.get_shape(),
                    'dtype': str(arr._dtype)
                }
                for name, arr in self._datasets.items()
            }
        }
        
        with open(f"{self._filename_base}.meta", 'wb') as f:
            pickle.dump(metadata, f)

    def close(self):
        self.flush()
        self._datasets.clear()

    def get_chunk_indices(self, dataset_name):
        """
        Generate chunk indices for efficient iteration over a dataset
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
            
        Yields:
        -------
        tuple
            Slice objects for each dimension
        """
        if dataset_name not in self._datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found")
            
        dataset = self._datasets[dataset_name]
        shape = dataset._get_shape()
        ndim = len(shape)
        
        # Calculate chunks for each dimension
        chunk_sizes = []
        for dim_size in shape:
            chunk_size = min(self._chunk_size, dim_size)
            chunk_sizes.append(chunk_size)
        
        # Generate all combinations of chunks
        current_indices = [0] * ndim
        while True:
            # Create slices for current chunk
            slices = []
            for dim in range(ndim):
                start = current_indices[dim]
                stop = min(start + chunk_sizes[dim], shape[dim])
                slices.append(slice(start, stop))
            
            yield tuple(slices)
            
            # Update indices
            for dim in range(ndim - 1, -1, -1):
                current_indices[dim] += chunk_sizes[dim]
                if current_indices[dim] < shape[dim]:
                    break
                current_indices[dim] = 0
                if dim == 0:
                    return

    def copy_dataset(self, source_name, dest_name):
        """
        Copy a dataset within the store
        """
        if source_name not in self._datasets:
            raise KeyError(f"Source dataset '{source_name}' not found")
            
        source = self._datasets[source_name]
        dest = self.create_dataset(dest_name, source._get_shape(), source._dtype)
        
        # Copy data in chunks
        for slices in self.get_chunk_indices(source_name):
            dest[slices] = source[slices]

    def resize_dataset(self, name, new_shape):
        """
        Resize a dataset to a new shape
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found")
            
        old_dataset = self._datasets[name]
        old_shape = old_dataset._get_shape()
        
        # Create new dataset with new shape
        temp_name = f"{name}_temp"
        new_dataset = self.create_dataset(temp_name, new_shape, old_dataset._dtype)
        
        # Copy data from old dataset to new dataset
        # Calculate common shape (minimum dimensions)
        common_shape = tuple(min(old, new) for old, new in zip(old_shape, new_shape))
        slices = tuple(slice(0, dim) for dim in common_shape)
        
        # Copy data in chunks
        chunk_slices = [slice(None)] * len(new_shape)
        for chunk_start in range(0, common_shape[0], self._chunk_size):
            chunk_end = min(chunk_start + self._chunk_size, common_shape[0])
            chunk_slices[0] = slice(chunk_start, chunk_end)
            new_dataset[tuple(chunk_slices)] = old_dataset[tuple(chunk_slices)]
        
        # Replace old dataset with new dataset
        del self._datasets[name]
        os.remove(f"{self._filename_base}_{name}.dat")
        os.rename(f"{self._filename_base}_{temp_name}.dat", f"{self._filename_base}_{name}.dat")
        self._datasets[name] = new_dataset
        del self._datasets[temp_name]

cdef class EliasPageCache:
    def __cinit__(self, str filename, size_t page_size=PAGE_SIZE, size_t n_pages=CACHE_SIZE, bint compression_enabled=False):
        self.page_size = page_size
        self.n_pages = n_pages
        self.access_counter = 0
        self.filename = filename
        self.compression_enabled = compression_enabled
        self.page_map = unordered_map[uint64_t, size_t]()
        
        # Allocate pages with extra space for compressed data
        self.pages = <EliasCachePage*>malloc(n_pages * sizeof(EliasCachePage))
        if not self.pages:
            raise MemoryError()
            
        for i in range(n_pages):
            self.pages[i].data = malloc(page_size)
            self.pages[i].compressed_data = malloc(page_size * 2)  # Worst case
            if not self.pages[i].data or not self.pages[i].compressed_data:
                raise MemoryError()
            memset(self.pages[i].data, 0, page_size)
            self.pages[i].page_number = 0
            self.pages[i].dirty = False
            self.pages[i].last_access = 0
            self.pages[i].is_compressed = False
            self.pages[i].compressed_size = 0

    cdef inline uint32_t encode_value(self, uint64_t value, char* dest) nogil:
        cdef:
            uint32_t bits_required = 64 - count_leading_zeros(value | 1)
            uint32_t len_bits = 64 - count_leading_zeros(bits_required | 1)
            uint32_t total_bits = len_bits * 2 + bits_required - 1
            uint32_t bytes_needed = (total_bits + 7) >> 3
            uint64_t encoded = value & ((1ULL << bits_required) - 1)
            
        encoded = (encoded << len_bits) | bits_required
        memcpy(dest, &encoded, bytes_needed)
        return bytes_needed

    cdef inline uint64_t decode_value(self, const char* src, uint32_t* bytes_read_ptr) nogil:
        cdef:
            uint64_t temp
            uint32_t len_bits, value_bits
            
        memcpy(&temp, src, 8)
        len_bits = temp & 0x3F
        value_bits = (temp >> len_bits) & ((1ULL << len_bits) - 1)
        
        bytes_read_ptr[0] = (len_bits * 2 + value_bits - 1 + 7) >> 3
        return value_bits

    cdef size_t compress_page(self, void* src, void* dest, size_t size) nogil:
        cdef:
            size_t i, out_pos = 0
            uint64_t* src_ptr = <uint64_t*>src
            char* dest_ptr = <char*>dest
            uint64_t prev_value = 0
            uint64_t delta
            uint32_t encoded_size
            
        for i in range(size // sizeof(uint64_t)):
            delta = src_ptr[i]
            if i > 0:
                delta -= prev_value
            encoded_size = self.encode_value(delta, dest_ptr + out_pos)
            out_pos += encoded_size
            prev_value = src_ptr[i]
            
        return out_pos

    cdef size_t decompress_page(self, void* src, void* dest, size_t compressed_size) nogil:
        cdef:
            size_t in_pos = 0, out_pos = 0
            char* src_ptr = <char*>src
            uint64_t* dest_ptr = <uint64_t*>dest
            uint64_t prev_value = 0
            uint32_t bytes_read
            
        while in_pos < compressed_size:
            dest_ptr[out_pos // sizeof(uint64_t)] = self.decode_value(
                src_ptr + in_pos, &bytes_read) + prev_value
            prev_value = dest_ptr[out_pos // sizeof(uint64_t)]
            in_pos += bytes_read
            out_pos += sizeof(uint64_t)
            
        return out_pos

    cdef void* get_compressed_page(self, uint64_t page_number, bint for_writing=False) except *:
        cdef:
            size_t i, lru_idx
            uint64_t oldest_access
            bytes data
            
        with nogil:
            if self.page_map.count(page_number):
                lru_idx = self.page_map[page_number]
                if for_writing:
                    self.pages[lru_idx].dirty = True
                self.pages[lru_idx].last_access = self.access_counter
                self.access_counter += 1
                
                if self.pages[lru_idx].is_compressed and for_writing:
                    # Decompress for writing
                    self.decompress_page(
                        self.pages[lru_idx].compressed_data,
                        self.pages[lru_idx].data,
                        self.pages[lru_idx].compressed_size
                    )
                    self.pages[lru_idx].is_compressed = False
                    
                return self.pages[lru_idx].data

            # Cache miss - find LRU
            oldest_access = self.access_counter
            for i in range(self.n_pages):
                if self.pages[i].last_access < oldest_access:
                    oldest_access = self.pages[i].last_access
                    lru_idx = i

            # Remove old page from map
            if self.page_map.count(self.pages[lru_idx].page_number):
                self.page_map.erase(self.pages[lru_idx].page_number)

        # Handle file I/O
        if self.pages[lru_idx].dirty:
            if self.compression_enabled and not self.pages[lru_idx].is_compressed:
                self.pages[lru_idx].compressed_size = self.compress_page(
                    self.pages[lru_idx].data,
                    self.pages[lru_idx].compressed_data,
                    self.page_size
                )
                self._write_compressed_page(lru_idx)
            else:
                self._write_page(lru_idx)
                
        self.pages[lru_idx].page_number = page_number
        self.pages[lru_idx].dirty = for_writing
        self.pages[lru_idx].last_access = self.access_counter
        self.access_counter += 1
        self.page_map[page_number] = lru_idx
        
        # Read page
        with nogil:
            memset(self.pages[lru_idx].data, 0, self.page_size)
            
        self.file.seek(page_number * self.page_size)
        data = self.file.read(self.page_size)
        
        if data and self.compression_enabled:
            self.pages[lru_idx].compressed_size = len(data)
            memcpy(self.pages[lru_idx].compressed_data, <char*>data, len(data))
            self.decompress_page(
                self.pages[lru_idx].compressed_data,
                self.pages[lru_idx].data,
                len(data)
            )
            self.pages[lru_idx].is_compressed = False
        elif data:
            memcpy(self.pages[lru_idx].data, <char*>data, len(data))
            
        return self.pages[lru_idx].data

def convert_h5_to_chunked(h5_filename, output_base, chunk_size=10000000):
    """
    Convert HDF5 file to chunked store format with memory-efficient processing.
    """
    import h5py
    from rich.progress import track
    import logging
    
    logging.info(f"Converting {h5_filename} to chunked store format")
    
    with h5py.File(h5_filename, 'r', driver='split') as h5f:
        store = ChunkedStore(output_base, 'w', chunk_size=chunk_size)
        
        # Copy attributes
        for key, value in h5f.attrs.items():
            store.attrs[key] = value
            
        # Copy datasets with chunked processing
        for name, dataset in h5f.items():
            logging.info(f"Processing dataset {name}")
            store.create_dataset(name, dataset.shape, dataset.dtype)
            
            # Calculate optimal chunk size for first dimension
            first_dim_chunk_size = min(chunk_size, dataset.shape[0])
            n_chunks = (dataset.shape[0] + first_dim_chunk_size - 1) // first_dim_chunk_size
            
            if len(dataset.shape) == 1:
                # Handle 1D arrays as before
                for start_idx in track(range(0, dataset.shape[0], first_dim_chunk_size),
                                    description=f"Converting {name}",
                                    total=n_chunks):
                    end_idx = min(start_idx + first_dim_chunk_size, dataset.shape[0])
                    chunk_data = dataset[start_idx:end_idx]
                    store[name][start_idx:end_idx] = chunk_data
            else:
                # Handle multi-dimensional arrays
                # Create a complete slice list with ':' for all dimensions except first
                slice_list = [slice(None)] * len(dataset.shape)
                
                for start_idx in track(range(0, dataset.shape[0], first_dim_chunk_size),
                                    description=f"Converting {name}",
                                    total=n_chunks):
                    end_idx = min(start_idx + first_dim_chunk_size, dataset.shape[0])
                    # Update only the first dimension slice
                    slice_list[0] = slice(start_idx, end_idx)
                    chunk_data = dataset[tuple(slice_list)]
                    store[name][tuple(slice_list)] = chunk_data
                
        store.close()
        
    logging.info("Conversion complete")
    return store
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

DEF PAGE_SIZE = 4096
DEF DEFAULT_CHUNK_SIZE = 16384
DEF CACHE_SIZE = 16384  # Number of pages to cache
DEF WRITE_BUFFER_SIZE = 1024 * 1024  # 1MB write buffer

cdef extern from "numpy/arrayobject.h":
    void* PyArray_DATA(np.ndarray arr) nogil
    np.ndarray PyArray_SimpleNew(int nd, np.npy_intp* dims, int typenum) nogil

cdef class PageCache:
    def __cinit__(self, str filename, size_t page_size=PAGE_SIZE, size_t n_pages=CACHE_SIZE, size_t dtype_size=8):
        self.filename = filename
        self.page_size = page_size
        self.n_pages = n_pages
        self.dtype_size = dtype_size  # Store dtype_size
        
        # Ensure page_size is aligned with dtype_size
        if self.page_size % dtype_size != 0:
            self.page_size = ((self.page_size + dtype_size - 1) // dtype_size) * dtype_size
        
        self.pages = <CachePage*>malloc(n_pages * sizeof(CachePage))
        if not self.pages:
            raise MemoryError()
        
        for i in range(n_pages):
            # Allocate aligned memory for each page
            self.pages[i].data = malloc(self.page_size)
            if not self.pages[i].data:
                raise MemoryError()
            self.pages[i].page_number = 0
            self.pages[i].dirty = False
        
        self.file = open(filename, 'rb+' if os.path.exists(filename) else 'wb+')

    def __dealloc__(self):
        self.flush()
        if self.file is not None:
            self.file.close()
        if self.pages:
            for i in range(self.n_pages):
                if self.pages[i].data:
                    free(self.pages[i].data)
            free(self.pages)

    cdef void flush(self) except *:
        cdef size_t i
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

    cdef void* get_page(self, uint64_t page_number, bint for_writing=False) except *:
        cdef:
            size_t i, lru_idx = 0
            uint64_t oldest_access = 0xFFFFFFFFFFFFFFFF
            bytes data
            size_t bytes_read
            char* data_ptr

        # Look for page in cache
        for i in range(self.n_pages):
            if self.pages[i].page_number == page_number:
                if for_writing:
                    self.pages[i].dirty = True
                return self.pages[i].data

        # Not found - find LRU page to evict
        for i in range(self.n_pages):
            if self.pages[i].page_number < oldest_access:
                oldest_access = self.pages[i].page_number
                lru_idx = i

        # Write back dirty page if needed
        if self.pages[lru_idx].dirty:
            self._write_page(lru_idx)

        # Load new page
        self.pages[lru_idx].page_number = page_number
        self.pages[lru_idx].dirty = for_writing
        
        self.file.seek(page_number * self.page_size)
        data = self.file.read(self.page_size)
        if data:  # If we read some data
            bytes_read = len(data)
            memcpy(self.pages[lru_idx].data, <char*>data, bytes_read)
            if bytes_read < self.page_size:  # Zero-fill the rest if we read less than a page
                memset(<char*>self.pages[lru_idx].data + bytes_read, 0, self.page_size - bytes_read)
        else:  # If we read nothing (EOF), zero the page
            memset(self.pages[lru_idx].data, 0, self.page_size)
        
        return self.pages[lru_idx].data

cdef class PageAlignedArray:
    def __cinit__(self, Py_ssize_t size, dtype='uint64', bint mmap_mode=False, str filename=None):
        self._dtype = np.dtype(dtype)
        self._np_dtype = self._dtype
        self._dtype_size = self._dtype.itemsize
        self._size = size
        self._mmap_mode = mmap_mode
        self._filename = filename
        self._buffer_size = WRITE_BUFFER_SIZE
        
        if filename is None:
            raise ValueError("Filename required for both mmap and non-mmap modes")
            
        # Ensure size is properly aligned based on dtype
        cdef Py_ssize_t aligned_size = (size * self._dtype_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)
        
        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                f.seek(aligned_size - 1)
                f.write(b'\0')
        
        if mmap_mode:
            self._init_mmap(filename, aligned_size)
        else:
            # Pass dtype_size to PageCache
            self._cache = PageCache(filename, PAGE_SIZE, CACHE_SIZE, self._dtype_size)

    def __dealloc__(self):
        if self._mmap_mode and self._mmap is not None:
            self._mmap.close()

    cdef void _direct_write(self, void* src, Py_ssize_t start, Py_ssize_t size) except *:
        """Direct write to file for large chunks."""
        cdef:
            Py_ssize_t offset = start * self._dtype_size
            size_t bytes_to_write = size * self._dtype_size
            bytes data = bytes((<char[:bytes_to_write]>src)[:bytes_to_write])
            
        if self._mmap_mode:
            memcpy(<char*>(<uintptr_t>self._mmap.buf) + offset, src, bytes_to_write)
            self._mmap.flush()
        else:
            with open(self._filename, 'rb+') as f:
                f.seek(offset)
                f.write(data)
        
    cdef void _init_mmap(self, str filename, Py_ssize_t size) except *:
        with open(filename, 'r+b') as f:
            self._mmap = py_mmap.mmap(f.fileno(), size, access=py_mmap.ACCESS_WRITE)
            
    cdef void _get_slice_data(self, Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step, void* dest) except *:
        cdef:
            Py_ssize_t i = start, idx = 0
            uint64_t page_number
            size_t offset_in_page
            void* page_data
            size_t bytes_to_copy
            size_t item_size = self._dtype_size  # Cache dtype size
            
        while i < stop:
            # Fix page number calculation to account for item size
            page_number = (i * item_size) // PAGE_SIZE
            offset_in_page = (i * item_size) % PAGE_SIZE
            
            if self._mmap_mode:
                memcpy(
                    <char*>dest + (idx * item_size),  # Fix stride
                    <char*>(<uintptr_t>self._mmap.buf) + (i * item_size),
                    item_size
                )
            else:
                page_data = self._cache.get_page(page_number)
                bytes_to_copy = min(PAGE_SIZE - offset_in_page, item_size)
                
                # Main copy
                memcpy(
                    <char*>dest + (idx * item_size),
                    <char*>page_data + offset_in_page,
                    bytes_to_copy
                )
                
                # Handle page boundary crossing
                if bytes_to_copy < item_size:
                    page_data = self._cache.get_page(page_number + 1)
                    memcpy(
                        <char*>dest + (idx * item_size) + bytes_to_copy,
                        <char*>page_data,
                        item_size - bytes_to_copy
                    )
            
            idx += 1
            i += step

    cdef void _set_slice_data(self, Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step, void* src) except *:
        cdef:
            Py_ssize_t chunk_size = stop - start
            Py_ssize_t i = start, idx = 0
            uint64_t page_number
            size_t offset_in_page
            void* page_data
            size_t bytes_to_copy
            size_t item_size = self._dtype_size  # Cache dtype size

        # Use direct write for large sequential writes
        if step == 1 and chunk_size * item_size >= self._buffer_size:
            self._direct_write(src, start, chunk_size)
            return

        while i < stop:
            # Fix page number calculation to account for item size
            page_number = (i * item_size) // PAGE_SIZE
            offset_in_page = (i * item_size) % PAGE_SIZE
            
            if self._mmap_mode:
                memcpy(
                    <char*>(<uintptr_t>self._mmap.buf) + (i * item_size),
                    <char*>src + (idx * item_size),
                    item_size
                )
            else:
                page_data = self._cache.get_page(page_number, True)
                bytes_to_copy = min(PAGE_SIZE - offset_in_page, item_size)
                
                # Main copy
                memcpy(
                    <char*>page_data + offset_in_page,
                    <char*>src + (idx * item_size),
                    bytes_to_copy
                )
                
                # Handle page boundary crossing
                if bytes_to_copy < item_size:
                    page_data = self._cache.get_page(page_number + 1, True)
                    memcpy(
                        <char*>page_data,
                        <char*>src + (idx * item_size) + bytes_to_copy,
                        item_size - bytes_to_copy
                    )
            
            idx += 1
            i += step

        if self._mmap_mode:
            self._mmap.flush()

    def flush(self):
        """Ensure all changes are written to disk"""
        if self._mmap_mode:
            self._mmap.flush()
        else:
            self._cache.flush()

    def __getitem__(self, key):
        cdef:
            Py_ssize_t start, stop, step
            np.ndarray result
            np.npy_intp dims[1]
            void* result_data
            Py_ssize_t idx
        
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self._size
            step = key.step if key.step is not None else 1
            
            if start < 0 or stop > self._size:
                raise IndexError("Index out of bounds")
                
            dims[0] = (stop - start) // step
            result = PyArray_SimpleNew(1, dims, self._np_dtype.num)
            result_data = PyArray_DATA(result)
            self._get_slice_data(start, stop, step, result_data)
            return result
        else:
            idx = <Py_ssize_t>key
            if idx < 0 or idx >= self._size:
                raise IndexError("Index out of bounds")
                
            dims[0] = 1
            result = PyArray_SimpleNew(1, dims, self._np_dtype.num)
            result_data = PyArray_DATA(result)
            self._get_slice_data(idx, idx + 1, 1, result_data)
            return result[0]

    def __setitem__(self, key, value):
        cdef:
            Py_ssize_t start, stop, step, idx
            np.ndarray arr
            void* arr_data
            
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self._size
            step = key.step if key.step is not None else 1
            
            if start < 0 or stop > self._size:
                raise IndexError("Index out of bounds")
                
            arr = np.ascontiguousarray(value, dtype=self._np_dtype)
            if (stop - start) // step != len(arr):
                raise ValueError("Value length does not match slice")
            
            arr_data = PyArray_DATA(arr)
            self._set_slice_data(start, stop, step, arr_data)
        else:
            idx = <Py_ssize_t>key
            if idx < 0 or idx >= self._size:
                raise IndexError("Index out of bounds")
                
            arr = np.ascontiguousarray([value], dtype=self._np_dtype)
            arr_data = PyArray_DATA(arr)
            self._set_slice_data(idx, idx + 1, 1, arr_data)

    def __len__(self):
        return self._size

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
                    info['size'],
                    info['dtype'],
                    self._mmap_mode,
                    f"{self._filename_base}_{name}.dat"
                )
                
    cdef PageAlignedArray _create_dataset(self, str name, tuple shape, object dtype) except *:
        cdef PageAlignedArray arr
        size = int(np.prod(shape))
        arr = PageAlignedArray(
            size, 
            dtype,
            self._mmap_mode,
            f"{self._filename_base}_{name}.dat"
        )
        self._datasets[name] = arr
        return arr

    def create_dataset(self, str name, tuple shape, dtype='uint64', **kwargs):
        return self._create_dataset(name, shape, dtype)
        
    def __getitem__(self, name):
        return self._datasets[name]
        
    def __setitem__(self, name, value):
        if isinstance(value, np.ndarray):
            if name not in self._datasets:
                self.create_dataset(name, value.shape, value.dtype)
            self._datasets[name][:] = value
            
    @property
    def attrs(self):
        return self._attrs
        
    def flush(self):
        import pickle
        metadata = {
            'attrs': self._attrs,
            'datasets': {
                name: {
                    'size': arr._size,
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

def convert_h5_to_chunked(h5_filename, output_base, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Convert HDF5 file to chunked store format with memory-efficient processing.
    """
    import h5py
    from rich.progress import track
    import logging
    
    logging.info(f"Converting {h5_filename} to chunked store format")
    
    with h5py.File(h5_filename, 'r') as h5f:
        store = ChunkedStore(output_base, 'w', chunk_size=chunk_size)
        
        # Copy attributes
        for key, value in h5f.attrs.items():
            store.attrs[key] = value
            
        # Copy datasets with chunked processing
        for name, dataset in h5f.items():
            logging.info(f"Processing dataset {name}")
            store.create_dataset(name, dataset.shape, dataset.dtype)
            
            # Calculate optimal chunk size
            chunk_size = min(chunk_size, dataset.shape[0])
            n_chunks = (dataset.shape[0] + chunk_size - 1) // chunk_size
            
            for start_idx in track(range(0, dataset.shape[0], chunk_size),
                                 description=f"Converting {name}",
                                 total=n_chunks):
                end_idx = min(start_idx + chunk_size, dataset.shape[0])
                chunk_data = dataset[start_idx:end_idx]
                store[name][start_idx:end_idx] = chunk_data
                
        store.close()
        
    logging.info("Conversion complete")
    return store
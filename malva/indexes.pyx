# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport cython
cimport numpy as np

from libc.stdint cimport uint32_t, int32_t, uint64_t
from libc.math cimport floor
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdio cimport FILE, fopen, fwrite, fread, fclose, getline
from libcpp.algorithm cimport lower_bound
from libcpp.vector cimport vector
from libcpp.utility cimport pair, move
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref, predecrement as dec
from cpython.exc cimport PyErr_CheckSignals

import h5py
import logging
import os
import io
import time
import numpy as np
import pandas as pd

from rich.progress import track

from malva.fast_map cimport map
from malva.fastq_processing cimport SequenceFastqParser, KmerFastqParser
from malva.kmer_processing import encode_kmer, get_kmers_numeric
from malva.utils import check_cell_string, convert_to_bytes
from malva.xopen import xopen

cdef int BUFFER_SIZE = max(io.DEFAULT_BUFFER_SIZE, 128 * 1024)

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)

cdef extern from "<cstdio>" nogil:
    double atof(const char* nptr)

cdef int compare_indexed_value(const pair[uint64_t, uint32_t]& a, const pair[uint64_t, uint32_t]& b) nogil:
    return a.first < b.first

cdef extern from *:
    """
    struct CompareFirst {
        bool operator()(const std::pair<uint64_t, std::pair<uint64_t, uint64_t>>& lhs, const uint64_t& rhs) const {
            return lhs.first < rhs;
        }
    };
    """
    struct CompareFirst:
        pass

cdef pair[uint64_t, pair[uint32_t, uint32_t]] binary_search(vector[pair[uint64_t, pair[uint32_t, uint32_t]]] vec, uint64_t target):
    cdef vector[pair[uint64_t, pair[uint32_t, uint32_t]]].iterator result

    result = lower_bound(vec.begin(), vec.end(), target, CompareFirst())
    
    if result == vec.end():
        return vec.back()
    elif deref(result).first != target and result != vec.begin():
        dec(result)
    
    return deref(result)

# TODO: docstring for this function
cdef int backed_binary_search_int(object arr, uint64_t low, uint64_t high, uint64_t x):
    if high >= low:
        mid = (high + low) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return backed_binary_search_int(arr, low, mid - 1, x)
        else:
            return backed_binary_search_int(arr, mid + 1, high, x)
 
    else:
        return -1

np.import_array()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ctypedef uint64_t kmer_t

cdef struct SpatialCoord:
    uint32_t x
    uint32_t y

cdef class MalvaIndex:
    cdef:
        public str index_dir
        public str index_file
        public int kmer_size
        public bint verbose
        public object index
        public tuple coord_lims
        public int n_chunks
        public list data_lengths
        public object spatial_coord
        public int n_spatial
        public int _n_kmers_processed
        public vector[pair[uint64_t, uint32_t]] _iter_seqs
        map[uint64_t, pair[uint64_t, uint64_t]] _index_backed
        vector[pair[uint64_t, pair[uint32_t, uint32_t]]] _cindex
        SpatialIndex spatial_index

    def __cinit__(self, str index_dir, bint rewrite=False, int kmer_size_initialize=24, bint verbose=False):
        self.index_dir = index_dir
        self.index = None
        self.index_file = os.path.join(self.index_dir, 'malva_index.h5')
        self.kmer_size = kmer_size_initialize
        self.n_chunks = 0
        self._n_kmers_processed = 0
        self.verbose = verbose
        self.spatial_index = SpatialIndex()

        self._iter_seqs = vector[pair[uint64_t, uint32_t]]()
        self._index_backed = map[uint64_t, pair[uint64_t, uint64_t]]()
        self._cindex = vector[pair[uint64_t, pair[uint32_t, uint32_t]]]()

        if rewrite:
            if os.path.exists(self.index_dir):
                import shutil
                shutil.rmtree(self.index_dir)
            os.mkdir(self.index_dir)
        elif self.index_exists(self):
            logging.info("The index exists. Will load now.")
            self.open()
        else:
            logging.info(f"Will create malva index at `{self.index_file}` with {kmer_size_initialize}-mers")
            self.initialize(kmer_size=kmer_size_initialize)

    @staticmethod
    def index_exists(self):
        # because we are using the driver 'split', to avoid corruption issues
        # this is a very weird issue that took me too long to debug
        # .. and I didn't find a solution :))))
        return os.path.exists(f'{self.index_file}-m.h5') and os.path.exists(f'{self.index_file}-r.h5')

    def initialize(self, int kmer_size=8):
        if kmer_size < 8 or kmer_size > 32:
            raise ValueError("`kmer_size` must be between 8 and 32 (inclusive)")
        if kmer_size < 24:
            logging.warn(f"For accuracy and speed, we recommend using > 24-mers (currently using {kmer_size}-mers)")

        self.kmer_size = kmer_size
        self.initialize_kmer_index()

    def initialize_kmer_index(self):
        self.open(mode='w')
        self.index.attrs['kmer_size'] = self.kmer_size
        self.index.attrs['n_chunks'] = self.n_chunks
        self.close()

    # TODO: rename to BarcodeIndex
    def set_spatial_index(self, SpatialIndex sindex):
        self.spatial_index = sindex
        self.set_spatial_coords(sindex.get_coords())

    def set_spatial_coords(self, coords: np.ndarray):
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    
        self.coord_lims = (xmin, xmax, ymin, ymax)
        self.n_spatial = len(coords)

        # TODO: this will be a context manager
        self.open(mode='r+')
        if 'spatial_coord' in self.index:
            del self.index['spatial_coord']

        self.index['spatial_coord'] = coords
        self.index.attrs['coord_lims'] = self.coord_lims
        self.index.attrs['n_spatial'] = self.n_spatial
        self.close()

    def open(self, str mode='r'):
        self.index = h5py.File(self.index_file, mode, driver="split")
        if 'kmer_size' in self.index.attrs:
            self.kmer_size = self.index.attrs['kmer_size']

        if 'coord_lims' in self.index.attrs:
            self.coord_lims = tuple(self.index.attrs['coord_lims'])

        if 'n_spatial' in self.index.attrs:
            self.n_spatial = self.index.attrs['n_spatial']

        if 'n_chunks' in self.index.attrs:
            self.n_chunks = self.index.attrs['n_chunks']
            self.data_lengths = [len(self.index[f'index_{chunk}_data']) for chunk in range(self.n_chunks)]

        if 'spatial_coord' in self.index:
            self.spatial_coord = self.index['spatial_coord']

    def close(self):
        self.index.flush()
        self.index.close()

    cdef void process_kmer(self, vector[pair[uint64_t, uint32_t]] kmer_coords):
        cdef:
            int _chunk
            vector[uint64_t] k_unique
            vector[uint64_t] k_change
            vector[uint32_t] k_data
            size_t i
            uint64_t current_kmer

        if self._n_kmers_processed == 0:
            logging.warn("There are no kmers to process!")
            return

        _chunk = self.n_chunks

        current_kmer = kmer_coords[0].first
        k_unique.push_back(current_kmer)
        k_change.push_back(<uint64_t>0)
        k_data.push_back(kmer_coords[0].second)

        with nogil:
            for i in range(1, self._n_kmers_processed):
                if kmer_coords[i].first != current_kmer:
                    k_unique.push_back(kmer_coords[i].first)
                    k_change.push_back(<uint64_t>i)
                    current_kmer = kmer_coords[i].first
                k_data.push_back(kmer_coords[i].second)

        _index = h5py.File(self.index_file, 'a', driver="split")

        try:
            temp_indices = _index.create_dataset(f"index_{_chunk}_indices", shape=(k_unique.size(),), dtype=np.uint64)
            temp_indices[:] = np.asarray(<np.uint64_t[:k_unique.size()]>&k_unique[0])
            temp_indices = _index.create_dataset(f"index_{_chunk}_indptr", shape=(k_change.size(),), dtype=np.uint64)
            temp_indices[:] = np.asarray(<np.uint64_t[:k_change.size()]>&k_change[0])
            temp_indices = _index.create_dataset(f"index_{_chunk}_data", shape=(k_data.size(),), dtype=np.uint32, chunks=True, compression=None)
            temp_indices[:] = np.asarray(<np.uint32_t[:k_data.size()]>&k_data[0])

            self.n_chunks += 1
            _index.attrs['n_chunks'] = self.n_chunks
        finally:
            _index.flush()
            _index.close()
        

    cdef int add_kmers(self, vector[uint64_t] kmers, uint64_t cell_bc) nogil:
        cdef:
            uint32_t coord
            size_t n_kmers, i
            pair[uint64_t, uint32_t] item

        coord = self.spatial_index.get_key(cell_bc)
        if coord == 0:
            return 0

        n_kmers = kmers.size()

        for i in range(n_kmers):
            item = pair[uint64_t, uint32_t](kmers[i], coord)
            self._iter_seqs.push_back(item)

        self._n_kmers_processed += n_kmers
        return 1

    cdef void _add_reads(self, list reads_in, str bam_tags, str read_group, int[] trim_limit, int n_report, int chunksize, int threads):
        cdef int num_reads = len(reads_in)
        cdef int _n_sequences = 0
        cdef SequenceFastqParser iter_r1
        cdef KmerFastqParser iter_r2
        cdef uint64_t r1
        cdef vector[uint64_t] r2

        if read_group != 'r1':
            raise NotImplementedError("Only 'r1' is implemented")

        _t0 = time.time()

        if num_reads == 2:
            iter_r1 = SequenceFastqParser(xopen(reads_in[0], "rb", threads=max(threads//2, 1)), BUFFER_SIZE, trim_start = trim_limit[0], trim_end = trim_limit[1])
            iter_r2 = KmerFastqParser(xopen(reads_in[1], "rb", threads=max(threads//2, 1)), BUFFER_SIZE, kmer_size = self.kmer_size)
            
            while True:
                try:
                    r2 = iter_r2.next()
                    r1 = iter_r1.next()
                except StopIteration: # reached eof
                    break

                _n_sequences += 1
                self.add_kmers(r2, r1)
            
                if (_n_sequences) % n_report == 0:
                    _t1 = time.time()
                    _elapsed = _t1 - _t0
                    logging.info(f"Processed {_n_sequences:,} sequences in {round(_elapsed, 2)} s ({round(n_report/_elapsed, 2):,} reads/s)")
                    _t0 = time.time()
                if (_n_sequences) % chunksize == 0:
                    self.write()
            
            # write last time the remaining reads
            self.write()
        else:
            raise ValueError("`--reads-in` must point to paired FASTQ files")

    def add_reads(self, list reads_in, str bam_tags='CB:{cell}', str cell='r1[2:27]', n_report: int=10_000_000, chunksize: int=100_000_000, threads: int = 1):
        self._iter_seqs.clear()
        read_group, start, end = check_cell_string(cell)
        self._add_reads(reads_in, bam_tags, read_group, [start, end], n_report, chunksize, threads)

    def merge_chunks(self, file_out):
        self._merge_chunks(file_out)
    
    @cython.wraparound(True)
    cdef void _merge_chunks(self, str file_out):
        cdef:
            uint32_t chunksize
            list srt_pointer = [0]*self.n_chunks
            list end_pointer = [0]*self.n_chunks
            list srt_pointer_to_data = [0]*self.n_chunks
            list end_pointer_to_data = [0]*self.n_chunks
            uint64_t min_value
            uint64_t curr_i_value
            int i
            size_t current_i_data_len
            np.ndarray[uint64_t, ndim=1] k_unique
            np.ndarray[uint64_t, ndim=1] k_change
            np.ndarray[uint32_t, ndim=1] k_data
            uint64_t current_kmer
            size_t total_processed
            size_t total_processed_data

        self.open()
        # this is chosen like this so the memory usage is ~ the same as when building the data
        # assuming that we run on the same cumputer
        chunksize = len(self.index['index_0_indices']) // (self.n_chunks * 2)
        logging.debug(f"Will use chunksize={chunksize}")

        # Initialize pointers
        for i in range(self.n_chunks):
            srt_pointer[i] = 0
            end_pointer[i] = min(chunksize, len(self.index[f'index_{i}_indices']))
            srt_pointer_to_data[i] = 0
            end_pointer_to_data[i] = 0

        with h5py.File(file_out, 'w', driver="split") as f_out:
            for key, value in self.index.attrs.items():
                f_out.attrs[key] = value
                
            f_out.attrs['n_chunks'] = 1
            
            max_size = sum(len(self.index[f'index_{i}_indices']) for i in range(self.n_chunks))
            max_size_data = sum(len(self.index[f'index_{i}_data']) for i in range(self.n_chunks))
            indices_out = f_out.create_dataset('index_0_indices', shape=(0,), maxshape=(max_size,), dtype='uint64')
            indptr_out = f_out.create_dataset('index_0_indptr', shape=(0,), maxshape=(max_size,), dtype='uint64')
            data_out = f_out.create_dataset('index_0_data', shape=(max_size_data,), dtype='uint32')

            if 'spatial_coord' in self.index:
                f_out.create_dataset('spatial_coord', data=self.index['spatial_coord'], dtype=np.float32)

            total_processed = 0
            total_processed_data = 0
            indptr_out.resize(1, axis=0)
            indptr_out[0] = 0

            while True:
                if all(srt_pointer[i] >= len(self.index[f'index_{i}_indices']) for i in range(self.n_chunks)):
                    break

                logging.info(f"Processed {total_processed_data:,}/{max_size_data:,} indexed locations")

                # find the smallest value across all chunks
                # because we avoid overflowing to much more than assigned chunksize
                min_value = 0xFFFFFFFFFFFFFFFF
                for i in range(self.n_chunks):
                    if end_pointer[i] < len(self.index[f'index_{i}_indices']):
                        curr_i_value = self.index[f'index_{i}_indices'][end_pointer[i]]
                        if curr_i_value < min_value:
                            min_value = curr_i_value

                k_unique = np.array([], dtype=np.uint64)
                k_change = np.array([], dtype=np.uint64)
                k_data = np.array([], dtype=np.uint32)

                for i in range(self.n_chunks):
                    if srt_pointer[i] == end_pointer[i]:
                        continue

                    current_i_data_len = len(self.index[f'index_{i}_data'])
                    end_pointer[i] = np.searchsorted(self.index[f'index_{i}_indices'][srt_pointer[i]:end_pointer[i]], min_value, side='right') + srt_pointer[i]
                    
                    srt_pointer_to_data[i] = self.index[f'index_{i}_indptr'][srt_pointer[i]]
                    # TODO: memoryview
                    chunk_indices = self.index[f'index_{i}_indices'][srt_pointer[i]:end_pointer[i]].astype(np.uint64)
                    chunk_indptr = self.index[f'index_{i}_indptr'][srt_pointer[i]:end_pointer[i]].astype(np.uint64)
    
                    if (end_pointer[i] + 1) >= len(self.index[f'index_{i}_indptr']):
                        end_pointer_to_data[i] = current_i_data_len
                    else:
                        end_pointer_to_data[i] = self.index[f'index_{i}_indptr'][end_pointer[i]+1]

                    chunk_indptr = np.diff(chunk_indptr, append=end_pointer_to_data[i])
    
                    k_unique = np.append(k_unique, np.repeat(chunk_indices, chunk_indptr.astype(np.int64)).astype(np.uint64))
                    k_data = np.append(k_data, self.index[f'index_{i}_data'][srt_pointer_to_data[i]:end_pointer_to_data[i]])

                # sort indices and reorder data accordingly
                # TODO: can be implemented more efficiently (reduce CPU time by 50%)
                sort_idx = np.argsort(k_unique)
                k_unique = k_unique[sort_idx]
                k_data = k_data[sort_idx]
                
                # Find unique elements and their indices
                unique_mask = np.ones(len(k_unique), dtype=bool)
                unique_mask[1:] = np.diff(k_unique) != 0

                # Extract unique k-mers and their change points
                k_unique = k_unique[unique_mask]
                k_change = np.where(unique_mask)[0].astype(np.uint64)

                new_size = total_processed_data + len(k_data)
                data_out[total_processed_data:new_size] = k_data

                last_indptr_out = indptr_out[-1]
                new_size = total_processed + len(k_unique)
                indices_out.resize(new_size, axis=0)
                indptr_out.resize(new_size, axis=0)
                indices_out[total_processed:] = k_unique
                indptr_out[total_processed:] = k_change + last_indptr_out + 1

                total_processed += len(k_unique)
                total_processed_data += len(k_data)

                # update index pointers
                for i in range(self.n_chunks):
                    srt_pointer[i] = end_pointer[i] + 1
                    end_pointer[i] = min(srt_pointer[i] + chunksize, len(self.index[f'index_{i}_indices']))

        self.close()

    def write(self):
        cdef vector[pair[uint64_t, uint32_t]].iterator first = self._iter_seqs.begin()
        cdef vector[pair[uint64_t, uint32_t]].iterator last = self._iter_seqs.end()
        sort(first, last, &compare_indexed_value)
        
        self.process_kmer(self._iter_seqs)

        self._n_kmers_processed = 0
        self._iter_seqs.clear()

    cdef unordered_map[uint64_t, unordered_set[uint32_t]] _find_kmer(self, np.ndarray kmers, uint32_t count_at_most=10_000, uint32_t count_at_least=10, uint32_t chunk_id=0):
        cdef:
            unordered_map[uint64_t, unordered_set[uint32_t]] vals = unordered_map[uint64_t, unordered_set[uint32_t]]()
            pair[uint64_t, uint64_t] _indptr
            uint64_t kmer
            np.ndarray _res
            uint32_t _res_item
            unordered_set[uint32_t] _set

        # TODO: move _data outside of here?
        _data = self.index[f'index_{chunk_id}_data']

        # TODO: move iterator out of here (at find_kmer)
        if self.verbose:
            iterator = track(kmers, description=f'Counting kmers at chunk {chunk_id}')
        else:
            iterator = kmers

        for kmer in iterator:
            _indptr = self._index_backed[kmer]
            
            if ((_indptr.second - _indptr.first) >= count_at_most) or ((_indptr.second - _indptr.first) <= count_at_least):
                continue

            _res = _data[_indptr.first:_indptr.second]
            _set = unordered_set[uint32_t]()
            for _res_item in _res:
                _set.insert(_res_item)

            vals[kmer] = _set

        return vals
    
    cdef unordered_map[uint64_t, unordered_set[uint32_t]] _find_kmer_constrained_memory(self, np.ndarray kmers, uint32_t count_at_most=10_000, uint32_t count_at_least=10, uint32_t chunk_id=0):
        # TODO: does not work! something is off here - what is returned by the binary_search function?
        cdef:
            unordered_map[uint64_t, unordered_set[uint32_t]] vals = unordered_map[uint64_t, unordered_set[uint32_t]]()
            pair[uint64_t, pair[uint32_t, uint32_t]] _cindex_res
            pair[uint64_t, uint64_t] _indptr
            uint64_t kmer, _start, _end, _indptr_first, _indptr_second
            int chunk_len
            np.ndarray _res
            uint32_t _res_item
            unordered_set[uint32_t] _set

        _data = self.index[f'index_{chunk_id}_data']
        _index_chunk = self.index[f'index_{chunk_id}_indices']
        chunk_len = len(self.index[f'index_{chunk_id}_indices'])

        if self.verbose:
            iterator = track(kmers, description=f'Counting kmers at chunk {chunk_id}')
        else:
            iterator = kmers

        for kmer in iterator:
            # find the approximate location using cindex (on memory)
            _cindex_res = binary_search(self._cindex, kmer)
            _start, _end = _cindex_res.second.first, _cindex_res.second.second

            # find the exact location in the index file (on disk)
            # we need _start and _end to define a _high and _low
            _index_idx = backed_binary_search_int(_index_chunk, _start, _end, kmer)

            # when binary search does not succeed
            if _index_idx == -1:
                continue
            
            # we get the indptrs (on disk) using the location
            _indptr_first = self.index[f'index_{chunk_id}_indptr'][_index_idx]
            _indptr_second = self.index[f'index_{chunk_id}_indptr'][_index_idx+1] if _index_idx < chunk_len else chunk_len
            
            # this is the same as for self._find_kmer(...), output datastructure should be compatible!
            if ((_indptr_second - _indptr_first) >= count_at_most) or ((_indptr_second - _indptr_first) <= count_at_least):
                continue

            _res = _data[_indptr_first:_indptr_second]
            _set = unordered_set[uint32_t]()
            for _res_item in _res:
                _set.insert(_res_item)

            vals[kmer] = _set

        return vals
    
    cdef unordered_map[uint64_t, unordered_set[uint32_t]] find_kmer(self, np.ndarray kmers, uint32_t count_at_most=10_000, uint32_t count_at_least=10, uint32_t chunk_id=0):
        if not self._index_backed.empty():
            return self._find_kmer(kmers, count_at_most, count_at_least, chunk_id)
        elif not self._cindex.empty():
            return self._find_kmer_constrained_memory(kmers, count_at_most, count_at_least, chunk_id)
        else:
            raise Exception("ERROR: index not found in memory.")

    cdef void _load_index_to_memory(self, int chunk_id = 0, size_t chunk_size=1_000_000):
        cdef:
            np.ndarray _indices_chunk, _indptr_chunk
            size_t i = 0, start = 0, end = 0
            size_t total_length, chunk_end

        if self.n_chunks > 1:
            logging.warn(f"Cannot process data split into more than 1 chunk. Processing chunk 0 out of {self.n_chunks}")

        total_length = len(self.index[f'index_{chunk_id}_indices'])

        while start < total_length - 1:
            end = min(start + chunk_size, total_length)
            chunk_end = min(end + 1, total_length)

            if chunk_end == end:
                end = end - 1

            _indices_chunk = self.index[f'index_{chunk_id}_indices'][start:end]
            _indptr_chunk = self.index[f'index_{chunk_id}_indptr'][start:chunk_end]

            for i in range(len(_indices_chunk)):
                self._index_backed[_indices_chunk[i]] = pair[uint64_t, uint64_t](_indptr_chunk[i], _indptr_chunk[i+1])

            start = end

        if end == total_length:
            last_index = total_length - 1
            self._index_backed[self.index[f'index_{chunk_id}_indices'][last_index]] = pair[uint64_t, uint64_t](
                self.index[f'index_{chunk_id}_indptr'][last_index],
                total_length
            )

    cdef void _load_index_to_constrained_memory(self, int chunk_id = 0, int max_mem_bytes = 0):
        # calculate the size of the constrained index (cindex)
        # each element will at least 3*64bit integers, plus some data-structure overhead
        cdef:
            int OVERHEAD = 2
            int cindex_size = max_mem_bytes//(24*OVERHEAD)
            int chunk_len, chunk_each
            size_t i = 0
            np.ndarray _cindex_indices, _cindex_indptr
            uint64_t _last_cindex_indices

        chunk_len = len(self.index[f'index_{chunk_id}_indices']) - 1
        
        chunk_each = chunk_len//cindex_size
        if chunk_each == 1:
            logging.debug("Maximum memory compatible with chunk length - falling back to loading entire index (no cindex)")
            self._load_index_to_memory(chunk_id)
            return

        _cindex_indices = self.index[f'index_{chunk_id}_indices'][::chunk_each]
        _cindex_indptr = np.arange(0, chunk_len, chunk_each)

        # TODO: remove this in runtime
        assert len(_cindex_indices) == len(_cindex_indptr)

        for i in range(len(_cindex_indices)-1):
            self._cindex.push_back(pair[uint64_t, pair[uint32_t, uint32_t]](_cindex_indices[i], pair[uint32_t, uint32_t](_cindex_indptr[i], _cindex_indptr[i+1])))

        # we add the last position
        _last_cindex_indices = self.index[f'index_{chunk_id}_indices'][chunk_len]
        self._cindex.push_back(pair[uint64_t, pair[uint32_t, uint32_t]](_last_cindex_indices, pair[uint32_t, uint32_t](_cindex_indptr[i+1], chunk_len)))

    def load_index_to_memory(self, chunk_id: int = 0, chunk_size: int = 1_000_000, max_mem: str = None, force: bool = False):
        max_mem_bytes = convert_to_bytes(max_mem) if max_mem is not None else 0

        # TODO: double check this
        if (not self._index_backed.empty() or not self._cindex.empty()) and not force:
            return
        
        # we make sure to clear both backed and constrained index
        # in case we load different modes at different times
        self._index_backed.clear()
        self._cindex.clear()

        if max_mem_bytes <= 0:
            self._load_index_to_memory(chunk_id, chunk_size)
        else:
            self._load_index_to_constrained_memory(chunk_id, max_mem_bytes)

    def where(self, sequence: Union[str, List[str]], sliding_size: int=128, pct_threshold: float=0.65, count_at_most: int=10_000, count_at_least: int=10, chunk_id: int = 0, single_count: bool = False, max_mem: str = None, force_reload: bool = False, *args, **kwargs):
        # TODO: reimplement seq_matches again, supporting various sequences...
        cdef:
            unordered_map[uint64_t, unordered_set[uint32_t]] current_kmers
            unordered_map[uint32_t, pair[uint32_t, uint32_t]] primary_map = unordered_map[uint32_t, pair[uint32_t, uint32_t]]()
            unordered_map[uint32_t, uint32_t] secondary_map = unordered_map[uint32_t, uint32_t]()
            np.ndarray kmer_locations = np.array([0]), kmer_count = np.array([0])
            float CONST_THRESHOLD = 0
            uint32_t idx_kmer
            int idx = 0
            pair[uint32_t, uint32_t] item
            pair[uint32_t, pair[uint32_t, uint32_t]] item_primary
            list whole_sliding_sequences = []
            list seq_matches = [[0, 1]]

        if pct_threshold < 0 or pct_threshold > 1:
            raise ValueError("`pct_threshold` must be a valid value between 0 and 1")

        if isinstance(sequence, str):
            sequence = [sequence]

        def get_whole_sliding_sequence(string, k):
            return [string[i:] for i in range(k)]
        
        for seq in sequence:
            if len(seq) < self.kmer_size:
                raise ValueError(f"Query sequence of length {len(seq)} cannot be smaller than kmer size {self.kmer_size}!")
            whole_sliding_sequences.extend(get_whole_sliding_sequence(seq, self.kmer_size))

        all_kmer_list = []
        for subseq in whole_sliding_sequences:
            all_kmer_list += [get_kmers_numeric(subseq, self.kmer_size, remove_noncomplex=True)]

        all_kmer_list = np.unique(np.concatenate(all_kmer_list))
        all_kmer_list = all_kmer_list[all_kmer_list != 0]

        if len(all_kmer_list) == 0:
            return (kmer_locations, kmer_count, seq_matches)

        self.load_index_to_memory(chunk_id=chunk_id, max_mem=max_mem, force=force_reload)

        CONST_THRESHOLD = (sliding_size//self.kmer_size) * pct_threshold

        current_kmers = self.find_kmer(all_kmer_list, count_at_most=count_at_most, count_at_least=count_at_least, chunk_id=chunk_id)

        if self.verbose:
            iterator = track(whole_sliding_sequences, description='Counting occurrences at kmers')
        else:
            iterator = whole_sliding_sequences
    
        for subseq in iterator:
            all_kmer_list = get_kmers_numeric(subseq, self.kmer_size, remove_noncomplex=True)

            for idx_kmer, kmer in enumerate(all_kmer_list):
                if current_kmers.find(kmer) == current_kmers.end():
                    continue
                
                # we do not add occurrence to low complexity kmers
                values = current_kmers[kmer] if kmer != 0 else []
                for value in values:
                    if primary_map.find(value) == primary_map.end():
                        primary_map[value].first = 1
                    else:
                        primary_map[value].first += 1
                    
                    primary_map[value].second = idx_kmer
                    # when the value is updated, we check for a max bound, so the comparison to CONST_THRESHOLD makes sense
                    primary_map[value].first = min(primary_map[value].first, <uint32_t>(sliding_size//self.kmer_size))

                # accumulate counts during first sliding_size - but process last iter
                # note to my future self: this makes sense
                if (idx_kmer + 1) < (sliding_size//self.kmer_size) and (idx_kmer + 1) < len(all_kmer_list):
                    continue

                for item_primary in primary_map:
                    value = item_primary.first
                    if secondary_map.find(value) == secondary_map.end() and primary_map[value].first > CONST_THRESHOLD:
                        secondary_map[value] = 1
                    elif primary_map[value].first > CONST_THRESHOLD and not single_count:
                        secondary_map[value] += 1
                    # TODO: this is faulty (messing up quantification, underestimating values...)
                    # when the value is not updated, .second != idx_kmer, thus we need to subtract
                    if primary_map[value].second - idx_kmer > 0:
                        primary_map[value].first = max(<uint32_t>0, (<int32_t>(primary_map[value].first) - 1))
            
            primary_map.clear()
 
        kmer_locations = np.empty(secondary_map.size(), dtype=np.uint32)
        kmer_count = np.empty(secondary_map.size(), dtype=np.uint32)

        for item in secondary_map:
            kmer_locations[idx] = item.first
            kmer_count[idx] = item.second
            idx += 1

        return (kmer_locations, kmer_count, seq_matches)

cdef struct LineData:
    float x
    float y
    uint64_t cell_bc

# TODO: rename and reorganize to BarcodeIndex
# TODO: then there's a SpatialIndex class inherited from BarcodeIndex
cdef class SpatialIndex:
    cdef:
        map[uint64_t, uint32_t] index
        vector[pair[float, float]] coords

    cdef void add(self, uint64_t cell_bc, uint32_t i) nogil:
        self.index[cell_bc] = i

    def get_coords(self):
        cdef np.ndarray[np.float32_t, ndim=2] arr = np.empty((self.coords.size(), 2), dtype=np.float32)
        cdef size_t i
        for i in range(self.coords.size()):
            arr[i, 0] = self.coords[i].first
            arr[i, 1] = self.coords[i].second
        return arr

    cdef uint32_t get_key(self, uint64_t key) noexcept nogil:
        return self.index[key]

    def save_binary(self, str filename):
        cdef FILE* file = fopen(filename.encode('ascii'), "wb")
        if file == NULL:
            raise IOError(f"Cannot open file {filename} for writing")

        # Write size of the map
        cdef size_t size = self.index.size()
        fwrite(&size, sizeof(size_t), 1, file)

        # Write contents: key, x, y
        cdef uint64_t key
        cdef float x, y
        cdef map[uint64_t, uint32_t].iterator it = self.index.begin()
        cdef map[uint64_t, uint32_t].iterator end = self.index.end()
        while it != end:
            key = move(deref(it).first)
            x = self.coords[deref(it).second].first
            y = self.coords[deref(it).second].second
            fwrite(&key, sizeof(uint64_t), 1, file)
            fwrite(&x, sizeof(float), 1, file)
            fwrite(&y, sizeof(float), 1, file)
            it += 1  # Correctly increment the iterator

        fclose(file)

    def load_binary(self, str filename):
        cdef FILE* file = fopen(filename.encode('ascii'), "rb")
        if file == NULL:
            raise IOError(f"Cannot open file {filename} for reading")

        # Read size of the map
        cdef size_t size
        fread(&size, sizeof(size_t), 1, file)

        # Clear existing structures and read new contents
        self.index.clear()
        self.coords.clear()

        cdef uint64_t key
        cdef float x, y
        cdef uint32_t i
        for i in range(size):
            fread(&key, sizeof(uint64_t), 1, file)
            fread(&x, sizeof(float), 1, file)
            fread(&y, sizeof(float), 1, file)
            self.index[key] = i
            self.coords.push_back(pair[float, float](x, y))

        fclose(file)

cdef void process_line(const char* line, int line_length, LineData* data, bint encode = True) noexcept nogil:
    cdef int field = 0
    cdef int start = 0
    cdef int i
    cdef char c
    cdef char[64] kmer
    for i in range(line_length):
        c = line[i]
        if c == b',' or c == b'\t' or c == b'\n':
            if field == 0:  # cell_bc
                memcpy(kmer, &line[start], i - start)
                kmer[i - start] = b'\0'
                if encode:
                    with gil:
                        data.cell_bc = encode_kmer(kmer.decode("ascii")[:i])
            elif field == 1:  # x coordinate
                data.x = atof(&line[start])
            elif field == 2:  # y coordinate
                data.y = atof(&line[start])
                break
            field += 1
            start = i + 1

def create_spatial_index(str spatial_barcode_file):
    cdef:
        SpatialIndex sindex = SpatialIndex()
        FILE* file
        char* line = NULL
        size_t len = 0
        ssize_t read
        LineData data
        vector[pair[float, float]] coords
        uint32_t line_count = 0 # cannot index more than 4 billion locations/cells
        Py_ssize_t report_interval = 10000000

    logging.info("Starting spatial index creation...")

    file = fopen(spatial_barcode_file.encode('ascii'), "r")
    if file == NULL:
        raise IOError(f"Cannot open file {spatial_barcode_file} for reading")

    getline(&line, &len, file)

    while True:
        read = getline(&line, &len, file)
        if read == -1:
            break
        process_line(line, read, &data)
        
        coords.push_back(pair[float, float](data.x, data.y))
        sindex.add(data.cell_bc, line_count)

        line_count += 1
        if line_count % report_interval == 0:
            PyErr_CheckSignals()
            logging.info(f"Processed {line_count:,} spatial barcodes.")

    # we also set the actual coordinate values
    sindex.coords = coords

    free(line)
    fclose(file)

    logging.info(f"Spatial index creation completed.")

    return sindex
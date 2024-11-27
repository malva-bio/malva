# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport cython
cimport numpy as np

from libc.stdint cimport uint16_t, uint32_t, int32_t, uint64_t
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
from malva.kmer_processing import encode_kmer, get_kmers_numeric, get_sliding_kmers_numeric
from malva.utils import check_cell_string, convert_to_bytes
from malva.xopen import xopen

cdef int BUFFER_SIZE = max(io.DEFAULT_BUFFER_SIZE, 4096 * 1024)

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)

cdef extern from "<cstdio>" nogil:
    double atof(const char* nptr)

cdef int compare_indexed_value(const pair[uint64_t, uint32_t]& a, const pair[uint64_t, uint32_t]& b) nogil:
    return a.first < b.first

cdef extern from *:
    """
    struct SearchGroup {
        std::vector<std::pair<uint64_t, size_t>> kmers;
        uint64_t start;
        uint64_t end;
        
        // Default constructor
        SearchGroup() : start(0), end(0) {}
        
        // Constructor with parameters
        SearchGroup(uint64_t s, uint64_t e) : start(s), end(e) {}
    };
    """
    struct SearchGroup:
        vector[pair[uint64_t, size_t]] kmers
        uint64_t start
        uint64_t end


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
        SpatialIndex spatial_index
        BackgroundModel background_model

        # for the hierarchical index
        bint using_hierarchical
        vector[size_t] hierarchical_sizes
        int page_size

    def __cinit__(self, str index_dir, bint rewrite=False, int kmer_size_initialize=24, bint verbose=False):
        self.index_dir = index_dir
        self.index = None
        self.index_file = os.path.join(self.index_dir, 'malva_index.h5')
        self.kmer_size = kmer_size_initialize
        self.n_chunks = 0
        self._n_kmers_processed = 0
        self.verbose = verbose
        self.spatial_index = SpatialIndex()
        self.background_model = BackgroundModel(self.kmer_size)

        self._iter_seqs = vector[pair[uint64_t, uint32_t]]()
        self._index_backed = map[uint64_t, pair[uint64_t, uint64_t]]()

        # for the hierarchical index
        self.using_hierarchical = False
        self.hierarchical_sizes = vector[size_t]()
        self.page_size = 1024

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
            logging.warning(f"For accuracy and speed, we recommend using > 24-mers (currently using {kmer_size}-mers)")

        self.kmer_size = kmer_size
        self.initialize_kmer_index()

    def initialize_kmer_index(self):
        self.open(mode='w')
        self.index.attrs['kmer_size'] = self.kmer_size
        self.index.attrs['n_chunks'] = self.n_chunks
        self.close()

    cdef void _create_hierarchical_index(self, int chunk_id):
        """Creates hierarchical index structure in HDF5 file."""
        if self.verbose:
            logging.info(f"Creating hierarchical index for chunk {chunk_id}")
            
        cdef:
            size_t base_size = len(self.index[f'index_{chunk_id}_indices'])
            size_t current_size = (base_size + self.page_size - 1) // self.page_size
            vector[size_t] level_sizes
            np.ndarray source_data
            str level_name
            size_t i, level

        # Calculate sizes for each level
        while current_size > self.page_size:
            level_sizes.push_back(current_size)
            if self.verbose:
                logging.info(f"Level size: {current_size}")
            current_size = (current_size + self.page_size - 1) // self.page_size

        level_sizes.push_back(current_size)
        if self.verbose:
            logging.info(f"Final level size: {current_size}")
            
        self.hierarchical_sizes = level_sizes

        # Ensure we're in write mode
        self.index.flush()
        if not self.index.mode == 'r+':
            logging.error(f"HDF5 file not in write mode! Current mode: {self.index.mode}")
            return

        # Store sizes
        level_sizes_name = f"hierarchical_{chunk_id}_sizes"
        if level_sizes_name in self.index:
            del self.index[level_sizes_name]
        
        sizes_array = np.array([level_sizes[i] for i in range(level_sizes.size())])
        self.index.create_dataset(level_sizes_name, data=sizes_array)
        
        # Create datasets for each level
        for level in range(len(level_sizes)):
            level_name = f"hierarchical_{chunk_id}_level_{level}"
            if self.verbose:
                logging.info(f"Creating level {level} dataset: {level_name}")
                
            if level_name in self.index:
                del self.index[level_name]
            
            if level == 0:
                source_data = self.index[f'index_{chunk_id}_indices'][::self.page_size]
            else:
                source_data = self.index[f"hierarchical_{chunk_id}_level_{level-1}"][::self.page_size]

            dset = self.index.create_dataset(
                level_name, 
                shape=(level_sizes[level],), 
                dtype=np.uint64,
                chunks=(min(self.page_size, level_sizes[level]),)
            )
            dset[:len(source_data)] = source_data[:level_sizes[level]]
            
            if self.verbose:
                logging.info(f"Created dataset {level_name} with size {len(source_data)}")

        self.index.flush()
        if self.verbose:
            logging.info("Completed hierarchical index creation")
            logging.info(f"Available datasets: {list(self.index.keys())}")

    # TODO: rename to BarcodeIndex
    def set_spatial_index(self, SpatialIndex sindex):
        self.spatial_index = sindex
        self.set_spatial_coords(sindex.get_coords())

    # TODO: not n_spatial but n_cells
    def set_barcode_index(self, SpatialIndex sindex):
        self.spatial_index = sindex
        self.n_spatial = sindex.num_items()

        self.open(mode='r+')
        self.index.attrs['n_spatial'] = self.n_spatial
        self.close()

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

    def open(self, str mode='r+'):
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
            size_t i, items = 0
            uint64_t current_kmer
            uint32_t current_data

        if self._n_kmers_processed == 0:
            logging.warning("There are no kmers to process!")
            return

        _chunk = self.n_chunks

        current_kmer = kmer_coords[0].first
        current_data = kmer_coords[0].second
        k_unique.push_back(current_kmer)
        k_change.push_back(<uint64_t>0)
        k_data.push_back(kmer_coords[0].second)
        items += 1

        with nogil:
            for i in range(1, self._n_kmers_processed):
                if kmer_coords[i].first != current_kmer:
                    # New kmer encountered
                    k_unique.push_back(kmer_coords[i].first)
                    k_change.push_back(items)
                    current_kmer = kmer_coords[i].first

                    # Reset current_data for the new kmer
                    current_data = kmer_coords[i].second
                    k_data.push_back(current_data)
                    items += 1
                    
                elif kmer_coords[i].second != current_data:
                    # Same kmer, but new data
                    current_data = kmer_coords[i].second
                    k_data.push_back(current_data)
                    items += 1

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
            int n_chunks = self.n_chunks
            list srt_pointer = [0]*n_chunks
            list end_pointer = [0]*n_chunks
            list srt_pointer_to_data = [0]*n_chunks
            list end_pointer_to_data = [0]*n_chunks
            uint64_t min_value
            uint64_t curr_i_value
            uint64_t len_per_k_data_chunk
            uint64_t total_len_per_k_data_chunk
            uint64_t total_data
            int i
            uint64_t current_i_data_len
            np.ndarray[uint64_t, ndim=1] k_unique
            np.ndarray[uint64_t, ndim=1] k_change
            np.ndarray[uint64_t, ndim=1] _k_change_cumsum
            np.ndarray[uint32_t, ndim=1] k_data
            np.ndarray[uint32_t, ndim=1] k_data_sorted
            np.ndarray[np.uint64_t, ndim=1] dest_indices
            uint64_t current_kmer
            uint64_t last_indptr_out_next = 0
            size_t total_processed
            size_t total_processed_data

            np.uint64_t start, end, dest, length

        self.open()
        # this is chosen like this so the memory usage is ~ the same as when building the data
        # assuming that we run on the same cumputer
        chunksize = len(self.index['index_0_indices']) // (n_chunks * 2)
        logging.debug(f"Will use chunksize={chunksize}")

        # Initialize pointers
        for i in range(n_chunks):
            srt_pointer[i] = 0
            end_pointer[i] = min(chunksize, len(self.index[f'index_{i}_indices']))
            srt_pointer_to_data[i] = 0
            end_pointer_to_data[i] = 0

        file_out_tmp = file_out + ".tmp"

        with h5py.File(file_out_tmp, 'w', driver="split") as f_out:
            for key, value in self.index.attrs.items():
                f_out.attrs[key] = value
                
            f_out.attrs['n_chunks'] = 1
            
            max_size = sum(len(self.index[f'index_{i}_indices']) for i in range(n_chunks))
            max_size_data = sum(len(self.index[f'index_{i}_data']) for i in range(n_chunks))
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
                if all(srt_pointer[i] >= len(self.index[f'index_{i}_indices']) for i in range(n_chunks)):
                    break

                if self.verbose:
                    logging.info(f"Processed {total_processed_data:,}/{max_size_data:,} indexed locations")

                # find the smallest value across all chunks
                # because we avoid overflowing to much more than assigned chunksize
                min_value = 0xFFFFFFFFFFFFFFFF
                for i in range(n_chunks):
                    if end_pointer[i] < len(self.index[f'index_{i}_indices']):
                        curr_i_value = self.index[f'index_{i}_indices'][end_pointer[i]]
                        if curr_i_value < min_value:
                            min_value = curr_i_value

                for i in range(n_chunks):
                    if srt_pointer[i] == end_pointer[i]:
                        continue

                    current_i_data_len = len(self.index[f'index_{i}_data'])
                    end_pointer[i] = np.searchsorted(self.index[f'index_{i}_indices'][srt_pointer[i]:end_pointer[i]], min_value, side='right') + srt_pointer[i]
                    srt_pointer_to_data[i] = self.index[f'index_{i}_indptr'][srt_pointer[i]]

                    if (end_pointer[i] + 1) >= len(self.index[f'index_{i}_indptr']):
                        end_pointer_to_data[i] = current_i_data_len
                    else:
                        end_pointer_to_data[i] = self.index[f'index_{i}_indptr'][end_pointer[i]+1]

                max_data_size = int(sum(end_pointer_to_data[i] - srt_pointer_to_data[i] for i in range(n_chunks)))
                k_unique = np.array([], dtype=np.uint64)
                k_indptr_start = np.array([], dtype=np.uint64)
                k_indptr_end = np.array([], dtype=np.uint64)
                k_data = np.zeros(max_data_size, dtype=np.uint32)

                total_data = 0

                for i in range(n_chunks):
                    chunk_indices = self.index[f'index_{i}_indices'][srt_pointer[i]:end_pointer[i]].astype(np.uint64)
                    chunk_indptr = self.index[f'index_{i}_indptr'][srt_pointer[i]:end_pointer[i]].astype(np.uint64)
                    chunk_data_size = end_pointer_to_data[i] - srt_pointer_to_data[i]
                    k_unique = np.append(k_unique, chunk_indices)

                    # we need to 'recenter' k_indptr, because it is in local coordinates (for k_data)
                    k_indptr_start = np.append(k_indptr_start, chunk_indptr + total_data - chunk_indptr[0])
                    k_indptr_end = np.append(k_indptr_end, np.append(chunk_indptr[1:], end_pointer_to_data[i]) + total_data - chunk_indptr[0])
                    k_data[int(total_data):int(total_data + chunk_data_size)] = self.index[f'index_{i}_data'][srt_pointer_to_data[i]:end_pointer_to_data[i]]

                    total_data += chunk_data_size

                # TODO: we can do at maximum performance by using scipy csr_matrix (optionally transposing)
                # sort indices and reorder data accordingly
                sort_idx = np.argsort(k_unique)
                k_unique = k_unique[sort_idx]
                # TODO: higher performance by not using unique but iterating over it
                k_unique_unique = np.unique(k_unique)
                k_indptr_start = k_indptr_start[sort_idx].astype(np.uint64)
                k_indptr_end = k_indptr_end[sort_idx].astype(np.uint64)
                
                _k_change_cumsum = np.append(np.array([0], dtype=np.uint64), np.cumsum(k_indptr_end - k_indptr_start).astype(np.uint64))
                _idx_change = np.append(np.array([1], dtype=np.uint64), (np.diff(k_unique) != 0).astype(np.uint64)).astype(bool)
                k_change = _k_change_cumsum[:-1][_idx_change]
                
                # reorder k_data based on the sorted indices
                k_data_sorted = np.zeros_like(k_data)
                dest_indices = np.cumsum(k_indptr_end - k_indptr_start)

                for start, end, dest in zip(k_indptr_start, k_indptr_end, dest_indices):
                    k_data_sorted[dest - (end - start):dest] = k_data[start:end]

                # move data to h5 object
                new_size = total_processed_data + len(k_data_sorted)
                data_out[total_processed_data:new_size] = k_data_sorted

                # last_indptr_out takes care that we store the pointers
                # respect to the correct coordinates
                last_indptr_out = indptr_out[-1]
                new_size = total_processed + len(k_unique_unique)
                indices_out.resize(new_size, axis=0)
                indptr_out.resize(new_size, axis=0)
                indices_out[total_processed:] = k_unique_unique
                indptr_out[total_processed:] = k_change + last_indptr_out + last_indptr_out_next

                total_processed += len(k_unique_unique)
                total_processed_data += len(k_data_sorted)
                last_indptr_out_next = <uint64_t>_k_change_cumsum[-1] - <uint64_t>k_change[-1]

                # update index pointers
                for i in range(n_chunks):
                    srt_pointer[i] = end_pointer[i] + 1
                    end_pointer[i] = min(srt_pointer[i] + chunksize, len(self.index[f'index_{i}_indices']))

        self.close()
        
        if self.verbose:
            logging.info("Reorganizing the h5 file for performance (might take a while...)")

        with h5py.File(file_out_tmp, "r", driver="split") as f_in, h5py.File(file_out, "w", driver="split") as f_out:
                for key, value in f_in.attrs.items():
                    f_out.attrs[key] = value
                
                f_out.attrs['n_chunks'] = 1

                f_out.create_dataset('index_0_indices', data=f_in['index_0_indices'], dtype='uint64')
                f_out.create_dataset('index_0_indptr', data=f_in['index_0_indptr'], dtype='uint64')
                f_out.create_dataset('index_0_data', data=f_in['index_0_data'], dtype='uint32')

                if 'spatial_coord' in f_in:
                    f_out.create_dataset('spatial_coord', data=f_in['spatial_coord'], dtype='float32')

        # remove the temporary file (which has the wrong file structure). 
        # TODO: another strategy? we need to have plenty of storage for this! (anyway we have it)
        os.remove(f'{file_out_tmp}-r.h5')
        os.remove(f'{file_out_tmp}-m.h5')

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

        # TODO: move _data outside of here? (for maybe some performance gain when calling _find_kmer repeatedly?)
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
    
    cdef vector[pair[uint64_t, pair[uint64_t, uint64_t]]] _batch_find_in_hierarchy(self, vector[uint64_t]& kmers, int chunk_id):
        """Batch binary search in hierarchy for multiple kmers."""
        cdef:
            vector[pair[uint64_t, pair[uint64_t, uint64_t]]] results
            vector[SearchGroup] current_groups
            vector[SearchGroup] next_groups
            np.ndarray level_data
            uint64_t kmer
            size_t i, j
            int current_level = len(self.hierarchical_sizes) - 1
            SearchGroup initial_group
            SearchGroup new_group
            
        # Initialize first group with all kmers at highest level
        initial_group.start = 0
        initial_group.end = self.hierarchical_sizes[current_level]
        for i in range(kmers.size()):
            initial_group.kmers.push_back(pair[uint64_t, size_t](kmers[i], i))
        current_groups.push_back(initial_group)
        
        while current_level >= 0:
            level_data = self.index[f"hierarchical_{chunk_id}_level_{current_level}"][:]
            next_groups.clear()
            
            # Process each group at this level
            for i in range(current_groups.size()):
                if current_groups[i].kmers.empty():
                    continue
                
                # Binary search for each kmer in the group's range
                for j in range(current_groups[i].kmers.size()):
                    kmer = current_groups[i].kmers[j].first
                    
                    # Find position in current level's range
                    start_pos = current_groups[i].start
                    end_pos = current_groups[i].end
                    
                    # Binary search in this range
                    while start_pos < end_pos:
                        mid = (start_pos + end_pos) // 2
                        if level_data[mid] < kmer:
                            start_pos = mid + 1
                        else:
                            end_pos = mid
                    
                    if current_level == 0:
                        # At leaf level, calculate final range
                        range_start = max(0, start_pos - 1) * self.page_size
                        range_end = min((start_pos + 1) * self.page_size, 
                                    len(self.index[f'index_{chunk_id}_indices']))
                        results.push_back(pair[uint64_t, pair[uint64_t, uint64_t]](
                            kmer,
                            pair[uint64_t, uint64_t](range_start, range_end)
                        ))
                    else:
                        # Create new group for next level
                        new_group.start = max(0, start_pos - 1) * self.page_size
                        new_group.end = min((start_pos + 1) * self.page_size,
                                        self.hierarchical_sizes[current_level - 1])
                        new_group.kmers.clear()
                        new_group.kmers.push_back(current_groups[i].kmers[j])
                        next_groups.push_back(new_group)
            
            current_groups = next_groups
            current_level -= 1
        
        return results

    cdef unordered_map[uint64_t, unordered_set[uint32_t]] _find_kmer_constrained_memory(self, np.ndarray kmers, uint32_t count_at_most=10_000, uint32_t count_at_least=10, uint32_t chunk_id=0):
        """Find kmers using hierarchical index structure with batch processing."""
        cdef:
            unordered_map[uint64_t, unordered_set[uint32_t]] vals = unordered_map[uint64_t, unordered_set[uint32_t]]()
            vector[uint64_t] kmer_vec
            vector[pair[uint64_t, pair[uint64_t, uint64_t]]] valid_ranges
            vector[pair[uint64_t, pair[uint64_t, uint64_t]]] valid_data_ranges
            uint64_t kmer, start_idx, end_idx, data_start, data_end
            unordered_set[uint32_t] _set
            np.ndarray indices_chunk, indptr_chunk, data_chunk
            int exact_idx
            double t0, t1
            size_t i
            
        _data = self.index[f'index_{chunk_id}_data']
        _indices = self.index[f'index_{chunk_id}_indices']
        _indptr = self.index[f'index_{chunk_id}_indptr']

        if self.verbose:
            iterator = track(kmers, description=f'Processing kmers at chunk {chunk_id}')
        else:
            iterator = kmers

        # Phase 1: Find all index ranges for kmers using batch search
        t0 = time.time()
        # Convert kmers to vector
        for kmer in iterator:
            kmer_vec.push_back(kmer)
        
        # Perform batch hierarchical search
        valid_ranges = self._batch_find_in_hierarchy(kmer_vec, chunk_id)

        t1 = time.time()
        if self.verbose:
            print(f"Time finding hierarchical ranges: {t1-t0:.2f}s")

        # Phase 2: Batch process indices and indptr lookups
        t0 = time.time()
        for i in range(valid_ranges.size()):
            kmer = valid_ranges[i].first
            start_idx = valid_ranges[i].second.first
            end_idx = valid_ranges[i].second.second
            
            indices_chunk = _indices[start_idx:end_idx]
            exact_idx = backed_binary_search_int(indices_chunk, 0, len(indices_chunk)-1, kmer)
            
            if exact_idx == -1:
                continue
                
            exact_idx += start_idx
            indptr_chunk = _indptr[exact_idx:exact_idx+2]
            data_start = indptr_chunk[0]
            data_end = indptr_chunk[1]
            
            if ((data_end - data_start) >= count_at_most) or ((data_end - data_start) <= count_at_least):
                continue
                
            valid_data_ranges.push_back(pair[uint64_t, pair[uint64_t, uint64_t]](
                kmer,
                pair[uint64_t, uint64_t](data_start, data_end)
            ))
        t1 = time.time()
        if self.verbose:
            print(f"Time processing indices and indptr: {t1-t0:.2f}s")

        # Phase 3: Batch process data lookups
        t0 = time.time()
        for i in range(valid_data_ranges.size()):
            kmer = valid_data_ranges[i].first
            data_start = valid_data_ranges[i].second.first
            data_end = valid_data_ranges[i].second.second
            
            _res = _data[data_start:data_end]
            _set = unordered_set[uint32_t]()
            for _res_item in _res:
                _set.insert(_res_item)
            vals[kmer] = _set
        t1 = time.time()
        if self.verbose:
            print(f"Time processing data: {t1-t0:.2f}s")

        return vals
    
    cdef unordered_map[uint64_t, unordered_set[uint32_t]] find_kmer(self, np.ndarray kmers, uint32_t count_at_most=10_000, uint32_t count_at_least=10, uint32_t chunk_id=0):
        """Enhanced find_kmer to support both standard and hierarchical approaches."""
        if not self.using_hierarchical:
            return self._find_kmer(kmers, count_at_most, count_at_least, chunk_id)
        else:
            return self._find_kmer_constrained_memory(kmers, count_at_most, count_at_least, chunk_id)

    cdef void _load_index_to_memory(self, int chunk_id = 0, size_t chunk_size=50_000_000, uint32_t count_at_most=10_000, uint32_t count_at_least=10):
        cdef:
            np.ndarray _indices_chunk, _indptr_chunk
            size_t counts
            size_t i = 0, start = 0, end = 0
            size_t total_length, chunk_end

        if self.n_chunks > 1:
            logging.warning(f"Cannot process data split into more than 1 chunk. Processing chunk 0 out of {self.n_chunks}")

        total_length = len(self.index[f'index_{chunk_id}_indices'])

        while start < total_length - 1:
            end = min(start + chunk_size, total_length)
            chunk_end = min(end + 1, total_length)

            if chunk_end == end:
                end = end - 1

            _indices_chunk = self.index[f'index_{chunk_id}_indices'][start:end]
            _indptr_chunk = self.index[f'index_{chunk_id}_indptr'][start:chunk_end]

            for i in range(len(_indices_chunk)):
                counts = _indptr_chunk[i+1] - _indptr_chunk[i]
                # we will not query them anyway, so we can make the in-memory index more lightweight
                if counts >= count_at_most or counts <= count_at_least:
                    continue
                self._index_backed[_indices_chunk[i]] = pair[uint64_t, uint64_t](_indptr_chunk[i], _indptr_chunk[i+1])

            start = end

        if end == total_length:
            last_index = total_length - 1
            self._index_backed[self.index[f'index_{chunk_id}_indices'][last_index]] = pair[uint64_t, uint64_t](
                self.index[f'index_{chunk_id}_indptr'][last_index],
                total_length
            )

    cdef void _load_index_to_constrained_memory(self, int chunk_id=0, int max_mem_bytes=0):
        """Initialize hierarchical index structure if it doesn't exist."""
        cdef:
            str level_name = f"hierarchical_{chunk_id}_level_0"
            str sizes_name = f"hierarchical_{chunk_id}_sizes"
            np.ndarray sizes_array
            size_t i
        
        if level_name not in self.index:
            if self.verbose:
                logging.info("Creating hierarchical index")
            self._create_hierarchical_index(chunk_id)
        else:
            # Load existing hierarchical sizes
            if self.verbose:
                logging.info("Loading existing hierarchical index")
            sizes_array = self.index[sizes_name][:]
            self.hierarchical_sizes.clear()
            for i in range(len(sizes_array)):
                self.hierarchical_sizes.push_back(sizes_array[i])
        
        self.using_hierarchical = True

    def load_index_to_memory(self, chunk_id: int = 0, chunk_size: int = 50_000_000, max_mem: str = None, force: bool = False, uint32_t count_at_most=10_000, uint32_t count_at_least=10):
        max_mem_bytes = convert_to_bytes(max_mem) if max_mem is not None else 0

        if (not self._index_backed.empty() or self.using_hierarchical) and not force:
            return
        
        # we make sure to clear both backed and constrained index
        # in case we load different modes at different times
        self._index_backed.clear()
        self.using_hierarchical = False

        if max_mem_bytes <= 0:
            self._load_index_to_memory(chunk_id, chunk_size, count_at_most, count_at_least)
        else:
            self._load_index_to_constrained_memory(chunk_id, max_mem_bytes)

    def set_background_model(self, BackgroundModel background_model):
        self.background_model = background_model

    def get_whole_sliding_sequence(self, string, k):
        return [string[i:] for i in range(k) if len(string[i:]) >= k]

    def get_whole_sliding_sequence_chunk(self, string, sliding_size):
        n = len(string)
        all_sliding = []
    
        for i in range(0, n - sliding_size + 1):
            sliding_string = string[i:i+sliding_size]
            all_sliding.append(sliding_string)

        return all_sliding

    def where(self, sequence: Union[str, List[str]], sliding_size: int=128, pct_threshold: float=0.65, count_at_most: int=10_000, count_at_least: int=10, chunk_id: int = 0, single_count: bool = False, max_mem: str = None, force_reload: bool = False, use_background_model: bool = True, *args, **kwargs):
        # TODO: reimplement seq_matches again, supporting various sequences...
        # TODO: when using cDNA, we get less matches than when using UTR. cDNA sequences contain the UTR, does not make sense!!!!!!
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
            list whole_sliding_sequences_idx = []
            int cumulative_seq_len = 0
            list seq_matches = [[0, 1]]

        if pct_threshold < 0 or pct_threshold > 1:
            raise ValueError("`pct_threshold` must be a valid value between 0 and 1")

        if isinstance(sequence, str):
            sequence = [sequence]
        
        for seq in sequence:
            if len(seq) < self.kmer_size:
                raise ValueError(f"Query sequence of length {len(seq)} cannot be smaller than kmer size {self.kmer_size}!")
            # we slide over the k-mers to generate offsets, later we take into account the sliding_size
            _sliding_seq = self.get_whole_sliding_sequence(seq, self.kmer_size)
            whole_sliding_sequences.extend(_sliding_seq)
            whole_sliding_sequences_idx.extend([[_i + cumulative_seq_len for _i in range(s, len(seq), self.kmer_size)] for s in range(len(_sliding_seq))])
            cumulative_seq_len += len(seq)

        all_kmer_list = []
        for subseq in whole_sliding_sequences:
            all_kmer_list += [get_kmers_numeric(subseq, self.kmer_size, remove_noncomplex=True)]

        # TODO: find which kmers are duplicate, and these are used for weighting correctly the overrepresentation score
        all_kmer_list = np.unique(np.concatenate(all_kmer_list))
        all_kmer_list = all_kmer_list[all_kmer_list != 0]

        print("processing kmers", len(all_kmer_list))

        if len(all_kmer_list) == 0:
            return (kmer_locations, kmer_count, seq_matches)

        self.load_index_to_memory(chunk_id=chunk_id, max_mem=max_mem, force=force_reload, count_at_most=count_at_most, count_at_least=count_at_least)

        CONST_THRESHOLD = (sliding_size//self.kmer_size) * pct_threshold
        BACKGROUND_THRESHOLD = 1 # TODO: this can be customizable

        current_kmers = self.find_kmer(all_kmer_list, count_at_most=count_at_most, count_at_least=count_at_least, chunk_id=chunk_id)

        # get unique subsequences
        split_sliding_sequences = set()
        for seq in sequence:
            split_sliding_sequences.update(set(self.get_whole_sliding_sequence_chunk(seq, sliding_size)))

        if self.verbose:
            # iterator = track(zip(whole_sliding_sequences, whole_sliding_sequences_idx), description='Counting occurrences at kmers')
            iterator = track(list(split_sliding_sequences), description='Counting occurrences at kmers')
        else:
            # iterator = zip(whole_sliding_sequences, whole_sliding_sequences_idx)
            iterator = list(split_sliding_sequences)

        # TODO: re-activate seq_matches
        # for subseq, subseq_idx in iterator:
        for subseq in iterator:
            all_kmer_list = get_kmers_numeric(subseq, self.kmer_size, remove_noncomplex=True)

            for idx_kmer, kmer in enumerate(all_kmer_list):
                # the kmer has not been found in the index
                if current_kmers.find(kmer) == current_kmers.end():
                    # seq_matches.extend([[subseq_idx[idx_kmer], 0]])
                    continue
                
                # TODO: we move this outside of the loop, because we can check the k-mers presence when querying them more efficiently
                # those mers above cutoff are not used for counting (i.e., exclude multimappers)
                if use_background_model and self.background_model.is_mer_above_cutoff(kmer, BACKGROUND_THRESHOLD):
                    # seq_matches.extend([[subseq_idx[idx_kmer], 0]])
                    continue
                
                # TODO: here we only count once those sliding sequences that appear more than once
                # we do not add occurrence to low complexity kmers (==0)
                values = current_kmers[kmer] if kmer != 0 else []
                for value in values:
                    if primary_map.find(value) == primary_map.end():
                        primary_map[value].first = 1
                    else:
                        primary_map[value].first += 1
                    
                    primary_map[value].second = idx_kmer
                    # when the value is updated, we check for a max bound, so the comparison to CONST_THRESHOLD makes sense
                    primary_map[value].first = min(primary_map[value].first, <uint32_t>(sliding_size//self.kmer_size))

                # seq_matches.extend([[subseq_idx[idx_kmer], len(values)]])

                # accumulate counts during first sliding_size - but process last iter
                # note to my future self: this makes sense
                if ((idx_kmer + 1) < (sliding_size//self.kmer_size)) and ((idx_kmer + 1) < len(all_kmer_list)):
                    continue

                for item_primary in primary_map:
                    value = item_primary.first
                    if secondary_map.find(value) == secondary_map.end() and primary_map[value].first > CONST_THRESHOLD:
                        secondary_map[value] = 1
                    elif primary_map[value].first > CONST_THRESHOLD and not single_count:
                        # heuristic, avoid counting the same UMI more than once (another large enough sliding window has to occur)
                        primary_map[value].first = 0
                        secondary_map[value] += 1
                    if primary_map[value].second - idx_kmer > 0 and primary_map[value].first > 0:
                        primary_map[value].first = <int32_t>(primary_map[value].first) - 1
            
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
# TODO: this also gets the flavor, and chooses the reader based on that
cdef class SpatialIndex:
    cdef:
        map[uint64_t, uint32_t] index
        vector[pair[float, float]] coords
        vector[pair[uint16_t, uint16_t]] coords_stomics

    def __cinit__(self):
        self.index = map[uint64_t, uint32_t]()
        self.coords = vector[pair[float, float]]()
        self.coords_stomics = vector[pair[uint16_t, uint16_t]]()

    cdef void add(self, uint64_t cell_bc, uint32_t i) nogil:
        self.index[cell_bc] = i

    def get_coords(self):
        cdef np.ndarray[np.float32_t, ndim=2] arr = np.empty((self.coords.size(), 2), dtype=np.float32)
        cdef size_t i
        for i in range(self.coords.size()):
            arr[i, 0] = self.coords[i].first
            arr[i, 1] = self.coords[i].second
        return arr

    def num_items(self):
        return self.index.size()

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

    # the STOmics data provides .bin files that are almost what we expect
    # in our SpatialIndex, but can be stored more efficiently
    # This works for the standard size of 1x1cm, has not been tested for other
    # e.g., support for 13x13cm chips
    def load_binary_stomics(self, str filename, int barcode_length = 25):
        if barcode_length <= 0 or barcode_length > 32:
            raise ValueError("Barcode length must be between 1 and 32 base pairs")

        cdef FILE* file = fopen(filename.encode('ascii'), "rb")
        if file == NULL:
            raise IOError(f"Cannot open file {filename} for reading")

        # Clear existing structures and read new contents
        self.index.clear()
        self.coords.clear()

        cdef uint64_t key, reversed_barcode
        cdef uint32_t x, y
        cdef uint32_t i = 0

        while True:
            if fread(&key, sizeof(uint64_t), 1, file) != 1:
                break
            fread(&x, sizeof(uint32_t), 1, file)
            fread(&y, sizeof(uint32_t), 1, file)

            reversed_barcode = self.reverse_barcode(key, barcode_length)
            self.index[reversed_barcode] = i
            self.coords_stomics.push_back(pair[uint16_t, uint16_t](<uint16_t>x, <uint16_t>y))
            i += 1
            
        fclose(file)

    cdef uint64_t reverse_barcode(self, uint64_t barcode, int barcode_length):
        cdef uint64_t reversed_barcode = 0
        cdef int i

        for i in range(barcode_length):
            reversed_barcode = (reversed_barcode << 2) | (barcode & 0b11)
            barcode >>= 2

        return reversed_barcode

    def get_coords_stomics(self):
        cdef np.ndarray[np.uint16_t, ndim=2] arr = np.empty((self.coords_stomics.size(), 2), dtype=np.uint16)
        cdef size_t i
        for i in range(self.coords_stomics.size()):
            arr[i, 0] = self.coords_stomics[i].first
            arr[i, 1] = self.coords_stomics[i].second
        return arr

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

cdef void process_line_whitelist(const char* line, int line_length, LineData* data) noexcept nogil:
    cdef int start = 0
    cdef int i
    cdef char c
    cdef char[64] kmer

    for i in range(line_length):
        c = line[i]
        if c == b'\n':
            memcpy(kmer, &line[0], i)
            kmer[i] = b'\0'
            with gil:
                data.cell_bc = encode_kmer(kmer.decode("ascii")[:i])

def create_singlecell_index(str whitelist_file):
    cdef:
        SpatialIndex sindex = SpatialIndex()
        FILE* file
        char* line = NULL
        size_t len = 0
        ssize_t read
        LineData data
        uint32_t line_count = 0
        Py_ssize_t report_interval = 10000000

    logging.info("Starting spatial index creation...")

    file = fopen(whitelist_file.encode('ascii'), "r")
    if file == NULL:
        raise IOError(f"Cannot open file {whitelist_file} for reading")

    getline(&line, &len, file)

    while True:
        read = getline(&line, &len, file)
        if read == -1:
            break
        process_line_whitelist(line, read, &data)
        
        sindex.add(data.cell_bc, line_count)

        line_count += 1
        if line_count % report_interval == 0:
            PyErr_CheckSignals()
            logging.info(f"Processed {line_count:,} spatial barcodes.")

    free(line)
    fclose(file)

    logging.info(f"Spatial index creation completed.")

    return sindex

cdef class BackgroundModel:
    cdef:
        map[uint64_t, uint16_t] model
        size_t total_mers
        size_t kmer_size
        bint verbose

    def __cinit__(self, int kmer_size, bint verbose = True):
        self.model = map[uint64_t, uint16_t]()
        self.total_mers = 0
        self.kmer_size = kmer_size
        self.verbose = verbose

    def create_from_reference(self, str filename, bint consecutive_genes = True):
        from malva.reader import iterate_fasta
        from malva.utils import check_file_exists

        cdef:
            map[uint64_t, uint16_t] temp_model
            map[uint64_t, uint16_t].iterator it
            map[uint64_t, uint16_t].iterator end
            str current_gene = ""
            uint64_t key

        check_file_exists(filename, except_when=False)

        if self.verbose:
            iterator = track(iterate_fasta(filename), description=f'Computing background {self.kmer_size}-mer')
        else:
            iterator = iterate_fasta(filename)

        for seq in iterator:
            it_gene_name = seq[0].split(":")[0]
            
            all_kmer_seq = get_sliding_kmers_numeric(seq[1], self.kmer_size, remove_noncomplex=True)
            for kmer in all_kmer_seq:
                temp_model[kmer] = 1
                self.total_mers += 1

            if it_gene_name == current_gene and consecutive_genes:
                continue

            it = temp_model.begin()
            end = temp_model.end()
            while it != end:
                key = move(deref(it).first)
                if self.model.find(key) == self.model.end():
                    self.model[key] = 1
                else:
                    self.model[key] += 1 # we only count once per reference entry (or grouped-consecutive)
                it += 1
            
            current_gene = it_gene_name
            temp_model.clear()
            
        if self.verbose:
            logging.info(f"Processed {self.model.size()} unique {self.kmer_size}-mers across {self.total_mers} occurrences")

    def is_mer_above_cutoff(self, uint64_t kmer, uint16_t cutoff):
        if self.model.find(kmer) == self.model.end():
            return False # assume if it does not exist, is below cutoff
        
        return self.model[kmer] > cutoff

    def save(self, str filename):
        cdef:
            FILE* file = fopen(filename.encode('ascii'), "wb")
            uint64_t key
            uint16_t count
            size_t size = self.model.size()
            map[uint64_t, uint16_t].iterator it = self.model.begin()
            map[uint64_t, uint16_t].iterator end = self.model.end()
    
        if file == NULL:
            raise IOError(f"Cannot open file {filename} for writing")

        fwrite(&size, sizeof(size_t), 1, file)

        while it != end:
            key = move(deref(it).first)
            count = self.model[key]
            fwrite(&key, sizeof(uint64_t), 1, file)
            fwrite(&count, sizeof(uint16_t), 1, file)
            it += 1

        fclose(file)

    def export_fasta(self, str filename):
        raise NotImplementedError("Cannot save as FASTA yet")

    def load(self, str filename):
        cdef:
            FILE* file = fopen(filename.encode('ascii'), "rb")
            size_t size
            uint64_t key
            uint16_t count

        if file == NULL:
            raise IOError(f"Cannot open file {filename} for reading")

        self.model.clear()
        self.total_mers = 0

        fread(&size, sizeof(size_t), 1, file)

        for _ in range(size):
            fread(&key, sizeof(uint64_t), 1, file)
            fread(&count, sizeof(uint16_t), 1, file)
            self.model[key] = count
            self.total_mers += 1

        fclose(file)

    def import_jellyfish_fasta(self, str filename):
        from malva.reader import iterate_fasta
        from malva.kmer_processing import encode_kmer

        cdef:
            uint64_t key
            uint16_t count

        self.model.clear()
        self.total_mers = 0

        if self.verbose:
            iterator = track(iterate_fasta(filename), description=f'Computing background {self.kmer_size}-mer frequency from {filename}')
        else:
            iterator = iterate_fasta(filename)

        for seq in iterator:
            self.model[encode_kmer(seq[1], self.kmer_size)] = <uint16_t>int(seq[0])
            self.total_mers += 1

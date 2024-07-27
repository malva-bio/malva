# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport cython
cimport numpy as np

from libc.stdint cimport uint16_t, uint32_t, int32_t, uint64_t
from libc.stdio cimport FILE, fopen, fwrite, fread, fclose
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref, preincrement as inc

import h5py
import logging
import os
import io
import time
import numpy as np
import pandas as pd

from operator import itemgetter
from rich.progress import track
from sortedcontainers import SortedList
from xopen import xopen

from malva.fast_map cimport map
from malva.kmer_processing import encode_kmer, get_kmers_numeric
from malva.fastq_processing cimport SequenceFastqParser, KmerFastqParser
from malva.utils import check_cell_string
from libcpp.utility cimport move

cdef int BUFFER_SIZE = max(io.DEFAULT_BUFFER_SIZE, 128 * 1024)

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)

cdef int compare_indexed_value(const pair[uint64_t, uint32_t]& a, const pair[uint64_t, uint32_t]& b) nogil:
    return a.first < b.first

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
        public object index
        public tuple coord_lims
        public int n_chunks
        public list data_lengths
        public int n_spatial
        public np.ndarray spatial_coords
        public int _n_kmers_processed
        public vector[pair[uint64_t, uint32_t]] _iter_seqs
        map[uint64_t, pair[uint32_t, uint32_t]] _index_backed
        SpatialIndex spatial_index

    def __cinit__(self, str index_dir, bint rewrite=False, int kmer_size_initialize=8):
        self.index_dir = index_dir
        self.index_file = os.path.join(self.index_dir, 'malva_index.h5')
        self.kmer_size = kmer_size_initialize
        self.n_chunks = 0
        self._n_kmers_processed = 0
        self.spatial_index = SpatialIndex()

        self._iter_seqs = vector[pair[uint64_t, uint32_t]]()
        self._index_backed = map[uint64_t, pair[uint32_t, uint32_t]]()

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
        return os.path.exists(self.index_file)

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

    def append_spatial(self, SpatialIndex sindex):
        self.spatial_index = sindex
        self.coord_lims = (sindex.xmin, sindex.xmax, sindex.ymin, sindex.ymax)
        self.n_spatial = sindex.xmax * sindex.ymax
        
        self.open(mode='r+')
        self.index.attrs['coord_lims'] = self.coord_lims
        self.index.attrs['n_spatial'] = self.n_spatial
        self.close()

    def open(self, str mode='r'):
        self.index = h5py.File(self.index_file, mode)
        if 'kmer_size' in self.index.attrs:
            self.kmer_size = self.index.attrs['kmer_size']

        if 'coord_lims' in self.index.attrs:
            self.coord_lims = tuple(self.index.attrs['coord_lims'])

        if 'n_spatial' in self.index.attrs:
            self.n_spatial = self.index.attrs['n_spatial']

        if 'n_chunks' in self.index.attrs:
            self.n_chunks = self.index.attrs['n_chunks']
            self.data_lengths = [len(self.index[f'index_{chunk}_data']) for chunk in range(self.n_chunks)]

    def close(self):
        self.index.close()

    cdef void process_kmer(self, vector[pair[uint64_t, uint32_t]] kmer_coords):
        cdef:
            np.npy_intp dims[1]
            np.npy_intp dims_b[2]
            int _chunk
            vector[uint64_t] k_unique
            vector[uint32_t] k_change
            size_t i, n = kmer_coords.size()
            uint64_t current_kmer
            pair[uint64_t, uint32_t]* data_ptr = &kmer_coords[0] if n > 0 else NULL

        if self._n_kmers_processed == 0:
            logging.warn("There are no kmers to process!")
            return

        _chunk = self.n_chunks

        if n == 0:
            return

        current_kmer = kmer_coords[0].first
        k_unique.push_back(current_kmer)
        k_change.push_back(0)

        with nogil:
            for i in range(1, n):
                if kmer_coords[i].first != current_kmer:
                    k_unique.push_back(kmer_coords[i].first)
                    k_change.push_back(i)
                    current_kmer = kmer_coords[i].first

        # create a numpy arrays
        dims[0] = k_unique.size()
        cdef np.ndarray k_unique_view = np.PyArray_SimpleNewFromData(1, dims, np.NPY_UINT64, <void*>&k_unique[0])
        cdef np.ndarray k_change_view = np.PyArray_SimpleNewFromData(1, dims, np.NPY_UINT32, <void*>&k_change[0])

        dims_b[0] = n
        dims_b[1] = 4
        cdef np.ndarray coords_view = np.PyArray_SimpleNewFromData(2, dims_b, np.NPY_UINT32, <void*>&(data_ptr[0].second))

        self.open(mode='r+')
        self.index.create_dataset(f"index_{_chunk}_indices", data=k_unique_view, dtype=np.uint64)
        self.index.create_dataset(f"index_{_chunk}_indptr", data=k_change_view, dtype=np.uint32)
        self.index.create_dataset(f"index_{_chunk}_data", data=coords_view[:, 0], dtype=np.uint32)
    
        self.n_chunks += 1
        self.index.attrs['n_chunks'] = self.n_chunks
        self.close()

    cdef int add_kmers(self, vector[uint64_t]& kmers, uint64_t cell_bc) nogil:
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

    @cython.wraparound(True)
    def merge_chunks(self, f: str, chunksize: int = 1_000_000):
        # TODO: have in context manager so it closes gracefully upon error
        self.open()

        _indices_lengths = [len(self.index[f'index_{chunk}_indices']) for chunk in range(0, self.n_chunks)]
        which_index = np.argmax(_indices_lengths)
        max_kmer_chunk = _indices_lengths[which_index]
        imin = np.array([0] * self.n_chunks)
        imax = np.array([chunksize] * self.n_chunks)

        result_indices = [None]
        result_indptr = [0]
        result_data = []

        _intm_result = [0, 0]

        output_file = h5py.File(f, 'w')

        # Copying attributes to the output file
        for key, value in self.index.attrs.items():
            output_file.attrs[key] = value
        
        # ... except for chunk size, which is 1
        output_file.attrs['n_chunks'] = 1

        # Create datasets in the output file
        output_file.create_dataset('index_0_indices', (0,), maxshape=(None,), dtype='uint64')
        output_file.create_dataset('index_0_indptr', (0,), maxshape=(None,), dtype='uint64')
        output_file.create_dataset('index_0_data', (0,), maxshape=(None,), dtype='uint32')

        #for _ in track(range(0, max_kmer_chunk, chunksize), description='Merging sorted chunks'):
        for _ in range(0, max_kmer_chunk, chunksize):
            _val_max_ix_0 = self.index[f'index_{which_index}_indices'][imin[0]:imax[0]][-1]
            imax = np.array([np.argwhere(self.index[f'index_{chunk}_indices'] <= _val_max_ix_0)[-1][0] for chunk in range(0, self.n_chunks)]) + 1

            # process for the chunk of N kmers
            chunk_indices = [self.index[f'index_{chunk}_indices'][imin[chunk]:imax[chunk]] for chunk in range(0, self.n_chunks)]
            chunk_indptr_lo = [self.index[f'index_{chunk}_indptr'][imin[chunk]:imax[chunk]] for chunk in range(0, self.n_chunks)]

            # We detect if this is the last chunk, and then we can process this
            chunk_indptr_hi = []
            for chunk in range(0, self.n_chunks):
                _chunk_indptr_hi = self.index[f'index_{chunk}_indptr'][(imin[chunk]+1):(imax[chunk]+1)]
                if len(_chunk_indptr_hi) < len(chunk_indptr_lo[chunk]):
                    _chunk_indptr_hi = np.concatenate([_chunk_indptr_hi, [len(self.index[f'index_{chunk}_data'])]])
                chunk_indptr_hi.append(_chunk_indptr_hi)

            # Compact for sorting
            chunks_indices = np.concatenate(chunk_indices)
            chunks_indptr_lo = np.concatenate(chunk_indptr_lo)
            chunks_indptr_hi = np.concatenate(chunk_indptr_hi)

            # We do mergesort to keep the order between chunks
            # The chunks *must* be sorted beforehand, otherwise this breaks!
            _chunk_idx_sorted = np.argsort(chunks_indices, kind='mergesort')
            chunks_labels = np.repeat(np.arange(self.n_chunks), [len(c) for c in chunk_indices])
            chunks_data = [self.index[f'index_{chunk}_data'][chunk_indptr_lo[chunk][0]:chunk_indptr_hi[chunk][-1]] for chunk in range(0, self.n_chunks) if len(chunk_indptr_hi[chunk]) > 0]

            # Resample and merge the chunk data into a single file
            for ind, l, lo, hi in (zip(chunks_indices[_chunk_idx_sorted],
                                        chunks_labels[_chunk_idx_sorted],
                                        chunks_indptr_lo[_chunk_idx_sorted],
                                        chunks_indptr_hi[_chunk_idx_sorted])):

                if result_indices[-1] != ind:
                    if result_indices[0] is None:
                        result_indices[0] = ind
                    else:
                        result_indices.append(ind)
                        result_indptr += [_intm_result[1]]
                        _intm_result[0] = _intm_result[1]

                _data = chunks_data[l][(lo-chunk_indptr_lo[l][0]):(hi-chunk_indptr_lo[l][0])]
                _intm_result[1] += len(_data)
                result_data.append(_data)

            result_data = np.concatenate(result_data)

            # Write the merged chunk to the output file
            current_indices_length = output_file['index_0_indices'].shape[0]
            current_indptr_length = output_file['index_0_indptr'].shape[0]
            current_data_length = output_file['index_0_data'].shape[0]

            output_file['index_0_indices'].resize((current_indices_length + len(result_indices),))
            output_file['index_0_indices'][current_indices_length:] = result_indices

            output_file['index_0_indptr'].resize((current_indptr_length + len(result_indptr),))
            output_file['index_0_indptr'][current_indptr_length:] = result_indptr

            output_file['index_0_data'].resize((current_data_length + len(result_data),))
            output_file['index_0_data'][current_data_length:] = result_data

            result_indices = [None]
            result_indptr = [0]
            result_data = []

            imin = imax.copy()
            imax = imin + chunksize
        
        output_file.close()
        self.close()

    def write(self):
        cdef vector[pair[uint64_t, uint32_t]].iterator first = self._iter_seqs.begin()
        cdef vector[pair[uint64_t, uint32_t]].iterator last = self._iter_seqs.end()
        # swap trick for out of scope memfree
        cdef vector[pair[uint64_t, uint32_t]] temp
        sort(first, last, &compare_indexed_value)
        
        self.process_kmer(self._iter_seqs)

        self._n_kmers_processed = 0
        self._iter_seqs.clear()
        self._iter_seqs.swap(temp)

    cdef unordered_map[uint64_t, unordered_set[uint32_t]] find_kmer(self, np.ndarray kmers, uint32_t count_at_most=10_000, uint32_t count_at_least=10, uint32_t chunk_id=0):
        # TODO: use unordered_map[uint64_t, unordered_set[uint32_t]]
        # because the dictionaries are too slow!
        cdef:
            unordered_map[uint64_t, unordered_set[uint32_t]] vals = unordered_map[uint64_t, unordered_set[uint32_t]]()
            pair[uint32_t, uint32_t] _indptr
            uint64_t kmer
            np.ndarray _res
            uint32_t _res_item
            unordered_set[uint32_t] _set

        _data = self.index[f'index_{chunk_id}_data']

        for kmer in kmers:
            _indptr = self._index_backed[kmer]
            
            if ((_indptr.second - _indptr.first) >= count_at_most) or ((_indptr.second - _indptr.first) <= count_at_least):
                continue

            _res = _data[_indptr.first:_indptr.second]
            _set = unordered_set[uint32_t]()
            for _res_item in _res:
                _set.insert(_res_item)

            vals[kmer] = _set

        return vals

    cdef void _load_index_to_memory(self, int chunk_id = 0):
        cdef:
            np.ndarray _indices, _indptr
            size_t i = 0
            pair[uint32_t, uint32_t] value

        if self.n_chunks > 1:
            logging.warn(f"Cannot process data split into more than 1 chunk. Processing chunk 0 out of {self.n_chunks}")

        if not self._index_backed.empty():
            return

        # TODO: remove this because it is a subset (for testing)
        _indices = self.index[f'index_{chunk_id}_indices'][:]
        _indptr = self.index[f'index_{chunk_id}_indptr'][:]
        
        length = len(_indices)

        for i in range(length - 1):
            self._index_backed[_indices[i]] = pair[uint32_t, uint32_t](_indptr[i], _indptr[i+1])
        
        i += 1
        self._index_backed[_indices[i]] = pair[uint32_t, uint32_t](_indptr[i], length - 1)

    def where(self, sequence, sliding_size=128, pct_threshold=0.65, count_at_most=10_000, count_at_least=10, chunk_id = 0, *args, **kwargs):
        cdef:
            unordered_map[uint64_t, unordered_set[uint32_t]] current_kmers
            unordered_map[uint32_t, pair[uint32_t, uint32_t]] primary_map = unordered_map[uint32_t, pair[uint32_t, uint32_t]]()
            unordered_map[uint32_t, uint32_t] secondary_map = unordered_map[uint32_t, uint32_t]()
            np.ndarray kmer_locations = np.array([0]), kmer_count = np.array([0])
            float CONST_THRESHOLD = 0
            uint32_t key
            int seq_no = 0
            int idx = 0
            pair[uint32_t, uint32_t] item
            pair[uint32_t, pair[uint32_t, uint32_t]] item_primary

        if len(sequence) < self.kmer_size:
            raise ValueError("Query sequence cannot be smaller than kmer size!")

        if pct_threshold < 0 or pct_threshold > 1:
            raise ValueError("`pct_threshold` must be a valid value between 0 and 1")

        self._load_index_to_memory(chunk_id=chunk_id)

        CONST_THRESHOLD = (sliding_size//self.kmer_size) * pct_threshold

        # TODO: reimplement seq_matches again
        seq_matches = [[0, 1]]

        def get_sliding_sequence(string, k):
            n = len(string)
            return [string[i:i + k] for i in range(n - k + 1)]

        def get_whole_sliding_sequence(string, k):
            return [string[i:] for i in range(k)]
        
        # get data for kmers
        all_kmer_list = []
        whole_sliding_sequences = get_whole_sliding_sequence(sequence, self.kmer_size)
    
        for subseq in whole_sliding_sequences:
            all_kmer_list += [get_kmers_numeric(subseq, self.kmer_size, remove_noncomplex=True)]

        all_kmer_list = np.unique(np.concatenate(all_kmer_list))
        all_kmer_list = all_kmer_list[all_kmer_list != 0]

        if len(all_kmer_list) == 0:
            return (kmer_locations, kmer_count, seq_matches)

        current_kmers = self.find_kmer(all_kmer_list, count_at_most=count_at_most, count_at_least=count_at_least, chunk_id=chunk_id)

        # TODO: these progress bars can be configured (displayed or not, whatever the user prefers)
        #for subseq in track(sliding_sequences, description='Counting kmers per sequence chunk'):
        for subseq in whole_sliding_sequences:
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

                
                # accumulate counts during first sliding_size - but process last iter
                if (idx_kmer + 1) < (sliding_size//self.kmer_size) and (idx_kmer + 1) < len(all_kmer_list):
                    continue

                # TODO: double check if that's correct
                # because we need to set sliding_size = 24 for this to work...
                for item_primary in primary_map:
                    value = item_primary.first
                    if secondary_map.find(value) == secondary_map.end() and primary_map[value].first > CONST_THRESHOLD:
                        secondary_map[value] = 1
                    elif primary_map[value].first > CONST_THRESHOLD:
                        secondary_map[value] += 1
                    if primary_map[value].second - idx_kmer > 1:
                        primary_map[value] = min(<uint32_t>max(0, (<int32_t>(primary_map[value].first) - 1)), (sliding_size//self.kmer_size))
            
            primary_map.clear()
 
        kmer_locations = np.empty(secondary_map.size(), dtype=np.uint32)
        kmer_count = np.empty(secondary_map.size(), dtype=np.uint32)

        for item in secondary_map:
            kmer_locations[idx] = item.first
            kmer_count[idx] = item.second
            idx += 1

        return (kmer_locations, kmer_count, seq_matches)

cdef class SpatialIndex:
    cdef:
        map[uint64_t, uint32_t] index
        uint32_t xmin, xmax, ymin, ymax

    def __cinit__(self):
        self.xmin = self.ymin = 0xFFFFFFFF
        self.xmax = self.ymax = 0

    cdef void add(self, uint64_t cell_bc, uint32_t i) nogil:
        self.index[cell_bc] = i
        # self.num_coords += 1

    # TODO: check if the noexcept has some performance penalty or not
    cdef uint32_t get_key(self, uint64_t key) noexcept nogil:
        return self.index[key]

    def set_bounds(self, uint32_t xmin, uint32_t xmax, uint32_t ymin, uint32_t ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @property
    def bounds(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    def save_binary(self, str filename):
        cdef FILE* file = fopen(filename.encode('ascii'), "wb")
        if file == NULL:
            raise IOError(f"Cannot open file {filename} for writing")

        # Write bounds
        fwrite(&self.xmin, sizeof(uint32_t), 1, file)
        fwrite(&self.xmax, sizeof(uint32_t), 1, file)
        fwrite(&self.ymin, sizeof(uint32_t), 1, file)
        fwrite(&self.ymax, sizeof(uint32_t), 1, file)

        # Write size of the map
        cdef size_t size = self.index.size()
        fwrite(&size, sizeof(size_t), 1, file)

        # Write map contents
        cdef uint64_t key
        cdef uint32_t value
        cdef map[uint64_t, uint32_t].iterator it = self.index.begin()
        cdef map[uint64_t, uint32_t].iterator end = self.index.end()

        while it != end:
            key = move(deref(it).first)
            value = deref(it).second
            fwrite(&key, sizeof(uint64_t), 1, file)
            fwrite(&value, sizeof(uint32_t), 1, file)
            it += 1  # Correctly increment the iterator

        fclose(file)

    def load_binary(self, str filename):
        cdef FILE* file = fopen(filename.encode('ascii'), "rb")
        if file == NULL:
            raise IOError(f"Cannot open file {filename} for reading")

        # Read bounds
        fread(&self.xmin, sizeof(uint32_t), 1, file)
        fread(&self.xmax, sizeof(uint32_t), 1, file)
        fread(&self.ymin, sizeof(uint32_t), 1, file)
        fread(&self.ymax, sizeof(uint32_t), 1, file)

        # Read size of the map
        cdef size_t size
        fread(&size, sizeof(size_t), 1, file)

        # Clear existing map and read new contents
        self.index.clear()
        cdef uint64_t key
        cdef uint32_t value
        for _ in range(size):
            fread(&key, sizeof(uint64_t), 1, file)
            fread(&value, sizeof(uint32_t), 1, file)
            self.index[key] = value

        fclose(file)


def create_spatial_index(str spatial_barcode_file, float rescale_coords=1, float index_resolution=1, bint recenter=True):
    cdef:
        SpatialIndex sindex = SpatialIndex()
        np.ndarray[np.float64_t, ndim=1] _xcoord, _ycoord
        np.ndarray[np.uint32_t, ndim=1] xcoord, ycoord
        np.ndarray[np.uint64_t, ndim=1] cell_bc
        Py_ssize_t i, n
        double xmin, ymin
        uint32_t x, y

    if spatial_barcode_file.endswith('.csv'):
        sep = ','
    elif spatial_barcode_file.endswith('.tsv'):
        sep = '\t'
    else:
        raise ValueError("Unsupported file format. The file must be either a .csv or .tsv file")

    spatial_index = pd.read_csv(spatial_barcode_file, sep=sep)
    spatial_index.drop_duplicates(subset='cell_bc')

    spatial_index['cell_bc_encoded'] = spatial_index.apply(lambda x: encode_kmer(x['cell_bc']), axis=1)

    _xcoord = spatial_index['xcoord'].values.astype(np.float64)
    _ycoord = spatial_index['ycoord'].values.astype(np.float64)
    cell_bc = spatial_index['cell_bc_encoded'].values.astype(np.uint64)

    if rescale_coords <= 0 or index_resolution <= 0:
        raise ValueError("`rescale_coords` and `index_resolution` must be greater than 0")

    if recenter:
        xmin = _xcoord.min()
        ymin = _ycoord.min()
        _xcoord -= xmin
        _ycoord -= ymin

    if rescale_coords != 1:
        _xcoord *= rescale_coords
        _ycoord *= rescale_coords

    if index_resolution != 1:
        _xcoord /= index_resolution
        _ycoord /= index_resolution

    xcoord = np.floor(_xcoord).astype(np.uint32)
    ycoord = np.floor(_ycoord).astype(np.uint32)

    xmin = xcoord.max()
    xmax = xcoord.max()
    ymin = ycoord.max()
    ymax = ycoord.max()

    if xmax >= 65535 or ymax >= 65535:
        raise ValueError("Spatial coordinates must be between 0 and 65,535")

    sindex.set_bounds(xmin, xmax, ymin, ymax)
    n = len(xcoord)
    for i in track(range(n), description='Spatial indexing'):
        x = xcoord[i]
        y = ycoord[i]
        coord_idx = np.ravel_multi_index([x, y], (xmax+1, ymax+1), order='C')
        sindex.add(cell_bc[i], coord_idx)

    return sindex
# prefix_index.pxd
# distutils: language = c++
# cython: language_level=3
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int16_t
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
cimport numpy as np

cdef class PrefixIndex:
    cdef:
        uint64_t* pi_suffix_offsets
        uint64_t* pi_data_offsets
        uint32_t* pi_data_sizes
        const unsigned char* suffix_mmap
        const unsigned char* data_mmap
        uint64_t data_size
        int data_fd
        object data_file_obj
        object _pi_arr_ref
        object _suf_arr_ref
        object _dat_arr_ref
        int l_prefix
        int l_suffix
        int kmer_size
        uint64_t n_prefixes
        uint64_t rshift
        uint64_t suffix_mask
        uint64_t n_kmers
        uint64_t n_cells
        str index_dir
        str pi_path
        str suffix_path
        str data_path
        str meta_path
        bint is_open
        object _last_query_times

        # CSR-mode attributes for query_batch_csr
        bint _csr_mode
        object _csr_km
        object _csr_cells
        object _csr_offsets
        int _csr_n_results

    cdef unordered_map[uint64_t, vector[uint32_t]] query_batch(
            self, uint64_t* query_kmers, uint64_t n_queries,
            uint32_t count_at_most, uint32_t count_at_least)

    cdef int _decompress_suffix_bucket(self, uint64_t suffix_offset,
            uint32_t* suffixes_out, uint64_t* data_offsets_out, uint32_t* data_lengths_out,
            uint32_t capacity)

    cdef const uint8_t* _get_compressed_data_ptr(self, uint64_t off, uint32_t sz) nogil

    cdef int _skip_n_varints(self, const uint8_t* p, int max_bytes, uint32_t count) nogil

    cdef int _decode_data_block(self, const uint8_t* compressed, int compressed_size,
            uint32_t* lengths, int n_entries, uint32_t* out, uint32_t cap)

    cdef int _binary_search_u32(self, const uint32_t* arr, int n, uint32_t target)

    cdef int _decode_single_kmer_data(self, const uint8_t* p, int max_bytes,
            uint32_t cell_count, uint32_t* out) nogil

    cdef int _stream_lookup_and_decode(self, uint64_t so, uint32_t target_suffix,
            uint32_t cam, uint32_t cal,
            const uint8_t* data_block, int data_bytes,
            uint32_t* out, uint32_t out_cap) nogil

    cdef unordered_map[uint64_t, vector[uint32_t]] find_kmer_c(self, np.ndarray kmers,
            uint64_t count_at_most, uint32_t count_at_least)


cdef class PrefixIndexBuilder:
    cdef:
        int l_prefix
        int l_suffix
        int kmer_size
        int jump_amount
        uint64_t n_prefixes
        uint64_t rshift
        uint64_t suffix_mask
        bint verbose
        vector[pair[uint64_t, uint32_t]] buffer
        uint64_t buffer_count
        str output_dir

    cdef void _sort_buffer(self)
    cdef void _write_buffer_to_chunk(self, str chunk_path)
    cdef inline void add_kmer_cell(self, uint64_t kmer, uint32_t cell_id)


# ── io_uring batch reader (Linux) / serial fallback (non-Linux) ───────────
cdef extern from "io_uring_reader.h":
    cdef cppclass IoUringReader:
        IoUringReader() except +
        int init(int fd, int qd) nogil
        int submit_reads(const uint64_t* offsets,
                         const uint32_t* sizes,
                         uint8_t** dests,
                         uint64_t n) nogil
        void destroy() nogil

    uint32_t align_up(uint32_t val, uint32_t alignment) nogil
    void* aligned_alloc_safe(size_t alignment, size_t size) nogil

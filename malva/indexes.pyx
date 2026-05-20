# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython
cimport numpy as np
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy, memset

from libc.stdio cimport FILE, fopen, fwrite, fread, fclose, fseek, SEEK_SET, SEEK_END, ftell
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.algorithm cimport sort as stdsort
from libcpp.unordered_map cimport unordered_map
from cpython.exc cimport PyErr_CheckSignals
from cpython.bytes cimport PyBytes_AS_STRING
import json, logging, numpy as np, os, time
np.import_array()

cdef extern from *:
    """
    #ifdef __linux__
    #include <sys/mman.h>
    static inline void prefetch_range(const void* addr, size_t len) {
        madvise((void*)addr, len, MADV_WILLNEED);
    }
    static inline void release_range(const void* addr, size_t len) {
        madvise((void*)addr, len, MADV_DONTNEED);
    }
    #elif defined(__APPLE__)
    #include <sys/mman.h>
    static inline void prefetch_range(const void* addr, size_t len) {
        madvise((void*)addr, len, MADV_WILLNEED);
    }
    static inline void release_range(const void* addr, size_t len) {
        madvise((void*)addr, len, MADV_FREE);
    }
    #else
    static inline void prefetch_range(const void* addr, size_t len) { (void)addr; (void)len; }
    static inline void release_range(const void* addr, size_t len) { (void)addr; (void)len; }
    #endif

    #define CPU_PREFETCH(addr) __builtin_prefetch((const void*)(addr), 0, 1)
    #define INITIAL_BUCKET_CAPACITY 65536
    #define MAX_DATA_PER_BUCKET 4194304

    #include <stdint.h>
    #include <string.h>

    static inline uint32_t
    decomp_suffixes_only(const uint8_t* __restrict__ buf, int buf_len,
                         uint32_t* __restrict__ suf_out,
                         int* __restrict__ len_section_pos_out)
    {
        if (buf_len < 4) { *len_section_pos_out = buf_len; return 0; }
        uint32_t n;
        memcpy(&n, buf, 4);
        if (n == 0) { *len_section_pos_out = 4; return 0; }
        if (buf_len < 8) { *len_section_pos_out = buf_len; return 0; }
        int pos = 4;
        uint32_t prev;
        memcpy(&prev, buf + pos, 4); pos += 4;
        suf_out[0] = prev;
        for (uint32_t i = 1; i < n; i++) {
            uint32_t delta = 0; uint8_t b;
            b = buf[pos++]; delta  = (uint32_t)(b & 0x7F); if (!(b & 0x80)) goto sd2;
            b = buf[pos++]; delta |= (uint32_t)(b & 0x7F) <<  7; if (!(b & 0x80)) goto sd2;
            b = buf[pos++]; delta |= (uint32_t)(b & 0x7F) << 14; if (!(b & 0x80)) goto sd2;
            b = buf[pos++]; delta |= (uint32_t)(b & 0x7F) << 21; if (!(b & 0x80)) goto sd2;
            b = buf[pos++]; delta |= (uint32_t)(b & 0x7F) << 28;
            sd2: prev += delta; suf_out[i] = prev;
        }
        *len_section_pos_out = pos;
        return n;
    }

    static inline void
    decode_lengths_upto(const uint8_t* __restrict__ buf, int buf_len,
                        int len_pos,
                        uint32_t* __restrict__ len_out,
                        uint32_t stop_at)
    {
        int pos = len_pos;
        for (uint32_t i = 0; i <= stop_at && pos < buf_len; i++) {
            uint32_t L = 0; uint8_t b;
            b = buf[pos++]; L  = (uint32_t)(b & 0x7F); if (!(b & 0x80)) goto ld2;
            b = buf[pos++]; L |= (uint32_t)(b & 0x7F) <<  7; if (!(b & 0x80)) goto ld2;
            b = buf[pos++]; L |= (uint32_t)(b & 0x7F) << 14; if (!(b & 0x80)) goto ld2;
            b = buf[pos++]; L |= (uint32_t)(b & 0x7F) << 21; if (!(b & 0x80)) goto ld2;
            b = buf[pos++]; L |= (uint32_t)(b & 0x7F) << 28;
            ld2: len_out[i] = L;
        }
    }

    static inline void
    build_bo_fast(const uint8_t* __restrict__ blk, int blk_len,
                  const uint32_t* __restrict__ lb, uint32_t ne,
                  uint32_t* __restrict__ bo, uint32_t stop_at)
    {
        int bofs = 0;
        uint32_t limit = stop_at + 1; if (limit > ne) limit = ne;
        for (uint32_t ki = 0; ki < limit; ki++) {
            bo[ki] = (uint32_t)bofs;
            uint32_t L = lb[ki];
            if (L == 0 || bofs + 4 > blk_len) continue;
            bofs += 4;
            uint32_t rem = L - 1;
            while (rem > 0 && bofs < blk_len) {
                uint64_t word;
                int avail = blk_len - bofs;
                if (avail >= 8) {
                    memcpy(&word, blk + bofs, 8);
                    uint64_t stop_mask = ~word & (uint64_t)0x8080808080808080ULL;
                    if (stop_mask) {
                        int n_stops = __builtin_popcountll(stop_mask);
                        if ((uint32_t)n_stops >= rem) {
                            uint64_t m = stop_mask;
                            for (uint32_t s = 1; s < rem; s++) {
                                m &= m - 1;
                            }
                            int stop_bit = __builtin_ctzll(m);
                            int stop_byte = stop_bit / 8;
                            bofs += stop_byte + 1;
                            rem = 0;
                        } else {
                            bofs += 8;
                            rem -= (uint32_t)n_stops;
                        }
                    } else {
                        bofs += 8;
                    }
                } else {
                    while (rem > 0 && bofs < blk_len) {
                        if ((blk[bofs++] & 0x80) == 0) rem--;
                    }
                }
            }
        }
    }

    static inline uint32_t
    scan_to_byte_offset(const uint8_t* __restrict__ blk, int blk_len,
                        const uint32_t* __restrict__ lb, uint32_t pos)
    {
        if (pos == 0) return 0;

        int bofs = 0;
        for (uint32_t ki = 0; ki < pos; ki++) {
            uint32_t L = lb[ki];
            if (L == 0) continue;
            if (bofs + 4 > blk_len) return (uint32_t)blk_len;
            bofs += 4;  /* first cell stored as raw uint32 */
            uint32_t rem = L - 1;
            while (rem > 0 && bofs < blk_len) {
                int avail = blk_len - bofs;
                if (avail >= 8) {
                    uint64_t word;
                    memcpy(&word, blk + bofs, 8);
                    uint64_t stop_mask = ~word & (uint64_t)0x8080808080808080ULL;
                    if (stop_mask) {
                        int n_stops = __builtin_popcountll(stop_mask);
                        if ((uint32_t)n_stops >= rem) {
                            uint64_t m = stop_mask;
                            for (uint32_t s = 1; s < rem; s++) m &= m - 1;
                            bofs += __builtin_ctzll(m) / 8 + 1;
                            rem = 0;
                        } else {
                            bofs += 8;
                            rem -= (uint32_t)n_stops;
                        }
                    } else {
                        bofs += 8;
                    }
                } else {
                    while (rem > 0 && bofs < blk_len) {
                        if ((blk[bofs++] & 0x80) == 0) rem--;
                    }
                }
            }
        }
        return (bofs < blk_len) ? (uint32_t)bofs : (uint32_t)blk_len;
    }

    static inline int
    decode_kmer_cells_fast(const uint8_t* __restrict__ blk, int blk_len,
                           uint32_t byte_off, uint32_t cell_count,
                           uint32_t* __restrict__ out)
    {
        int bpos = (int)byte_off;
        if (cell_count == 0 || bpos + 4 > blk_len) return 0;
        uint32_t prev;
        memcpy(&prev, blk + bpos, 4); bpos += 4;
        out[0] = prev;
        for (uint32_t j = 1; j < cell_count; j++) {
            if (bpos >= blk_len) return (int)j;
            uint32_t delta = 0;
            uint8_t b;
            b = blk[bpos++]; delta  = (uint32_t)(b & 0x7F);        if (!(b & 0x80)) goto cdone;
            b = blk[bpos++]; delta |= (uint32_t)(b & 0x7F) <<  7;  if (!(b & 0x80)) goto cdone;
            b = blk[bpos++]; delta |= (uint32_t)(b & 0x7F) << 14;  if (!(b & 0x80)) goto cdone;
            b = blk[bpos++]; delta |= (uint32_t)(b & 0x7F) << 21;  if (!(b & 0x80)) goto cdone;
            b = blk[bpos++]; delta |= (uint32_t)(b & 0x7F) << 28;
            cdone:
            prev += delta;
            out[j] = prev;
        }
        return (int)cell_count;
    }

    #if defined(__linux__) && __has_include(<liburing.h>)
    #include <liburing.h>
    #include <fcntl.h>
    #include <unistd.h>

    typedef struct {
        uint8_t*  buf;
        uint32_t  bidx;
        uint32_t  buf_sz;
    } UringSlot;

    static int
    uring_read_decode_all(
        int fd,
        const uint64_t* __restrict__ bkt_dbo,
        const uint32_t* __restrict__ bkt_dbs,
        const uint32_t* __restrict__ bkt_pool_off,
        const uint32_t* __restrict__ bkt_ne,
        const uint32_t* __restrict__ bkt_hit_count,
        const uint32_t* __restrict__ bkt_max_hit_pos,
        const uint32_t* __restrict__ lb_pool,
        uint64_t n_bkts,
        const uint64_t* __restrict__ sort_order,
        const uint64_t* __restrict__ hit_km,
        const uint32_t* __restrict__ hit_bidx,
        const uint32_t* __restrict__ hit_spos,
        const uint32_t* __restrict__ hit_dl,
        uint64_t n_hits,
        const uint64_t* __restrict__ bkt_hit_start,
        const uint32_t* __restrict__ bkt_hit_cnt,
        uint32_t sparse_threshold,
        uint64_t* __restrict__ out_km,
        uint32_t* __restrict__ out_cells,
        uint64_t* __restrict__ out_offsets,
        uint64_t  out_cells_cap,
        uint32_t*  tmp_dd,
        uint32_t   tmp_dd_cap,
        uint64_t* bytes_read_out
    )
    {
        const int QD = 64;
        struct io_uring ring;
        int ret = io_uring_queue_init(QD, &ring, 0);
        if (ret < 0) return ret;

        uint32_t max_bsz = 0;
        for (uint64_t i = 0; i < n_bkts; i++) {
            uint32_t sz = (bkt_dbs[i] + 511) & ~511u;  /* 512-align */
            if (sz > max_bsz) max_bsz = sz;
        }
        if (max_bsz == 0) { io_uring_queue_exit(&ring); return 0; }

        UringSlot* slots = (UringSlot*)malloc(QD * sizeof(UringSlot));
        if (!slots) { io_uring_queue_exit(&ring); return -1; }
        for (int s = 0; s < QD; s++) {
            slots[s].buf = (uint8_t*)aligned_alloc(4096,
                ((size_t)max_bsz + 4095) & ~(size_t)4095);
            slots[s].buf_sz = max_bsz;
            slots[s].bidx = (uint32_t)-1;
            if (!slots[s].buf) {
                for (int k = 0; k < s; k++) free(slots[k].buf);
                free(slots);
                io_uring_queue_exit(&ring);
                return -1;
            }
        }

        int* free_slots = (int*)malloc(QD * sizeof(int));
        if (!free_slots) {
            for (int s = 0; s < QD; s++) free(slots[s].buf);
            free(slots); io_uring_queue_exit(&ring); return -1;
        }
        for (int s = 0; s < QD; s++) free_slots[s] = s;
        int n_free = QD;

        uint32_t  bo_cap = 4096;
        uint32_t* bo     = (uint32_t*)malloc(bo_cap * 4);
        if (!bo) {
            for (int s = 0; s < QD; s++) free(slots[s].buf);
            free(slots); free(free_slots); io_uring_queue_exit(&ring); return -1;
        }

        uint64_t n_results  = 0;
        uint64_t cells_used = 0;
        out_offsets[0] = 0;
        uint64_t bytes_read = 0;

        uint64_t next_submit = 0;
        int inflight = 0;

        while (next_submit < n_bkts || inflight > 0) {

            while (inflight < QD && next_submit < n_bkts && n_free > 0) {
                uint32_t bidx = (uint32_t)next_submit;
                next_submit++;
                uint32_t dsz = bkt_dbs[bidx];
                if (dsz == 0 || bkt_hit_cnt[bidx] == 0) continue;

                int slot_id = free_slots[--n_free];
                UringSlot* sl = &slots[slot_id];
                sl->bidx = bidx;

                uint32_t read_sz = (dsz + 511) & ~511u;
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (!sqe) { free_slots[n_free++] = slot_id; break; }
                io_uring_prep_read(sqe, fd, sl->buf, read_sz, bkt_dbo[bidx]);
                io_uring_sqe_set_data64(sqe, (uint64_t)slot_id);
                inflight++;
            }

            if (inflight == 0) break;

            ret = io_uring_submit(&ring);
            if (ret < 0 && ret != -EBUSY) break;

            struct io_uring_cqe* cqe;
            ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) break;

            unsigned head, reaped = 0;
            io_uring_for_each_cqe(&ring, head, cqe) {
                int slot_id = (int)io_uring_cqe_get_data64(cqe);
                UringSlot* sl = &slots[slot_id];
                uint32_t bidx = sl->bidx;
                int res_bytes = cqe->res;
                reaped++;
                inflight--;

                if (res_bytes > 0 && bidx != (uint32_t)-1) {
                    const uint8_t* blk    = sl->buf;
                    int blk_len           = (int)bkt_dbs[bidx];
                    const uint32_t* cur_lb = lb_pool + bkt_pool_off[bidx];
                    uint32_t cur_ne       = bkt_ne[bidx];
                    uint32_t cur_hits     = bkt_hit_cnt[bidx];
                    uint32_t cur_maxpos   = bkt_max_hit_pos[bidx];
                    bytes_read += blk_len;

                    if (cur_hits > sparse_threshold) {
                        if (cur_ne > bo_cap) {
                            bo_cap = cur_ne + 256;
                            free(bo);
                            bo = (uint32_t*)malloc(bo_cap * 4);
                            if (!bo) goto done;
                        }
                        build_bo_fast(blk, blk_len, cur_lb, cur_ne, bo, cur_maxpos);
                    }

                    uint64_t h_start = bkt_hit_start[bidx];
                    for (uint32_t h = 0; h < cur_hits; h++) {
                        uint64_t idx = sort_order[h_start + h];
                        uint32_t pos = hit_spos[idx];
                        uint32_t dl  = hit_dl[idx];

                        if (pos >= cur_ne) continue;

                        uint32_t byte_off;
                        if (cur_hits > sparse_threshold) {
                            byte_off = bo[pos];
                        } else {
                            byte_off = scan_to_byte_offset(blk, blk_len, cur_lb, pos);
                        }
                        if (byte_off >= (uint32_t)blk_len) continue;
                        if (cells_used + dl > out_cells_cap) goto done;
                        if (dl > tmp_dd_cap) continue;

                        int decoded = decode_kmer_cells_fast(
                            blk, blk_len, byte_off, dl, tmp_dd);
                        if (decoded <= 0) continue;

                        uint64_t cell_start = cells_used;
                        out_cells[cells_used++] = tmp_dd[0];
                        for (int j = 1; j < decoded; j++) {
                            if (tmp_dd[j] != tmp_dd[j-1])
                                out_cells[cells_used++] = tmp_dd[j];
                        }
                        if (cells_used > cell_start) {
                            out_km[n_results]          = hit_km[idx];
                            out_offsets[n_results + 1] = cells_used;
                            n_results++;
                        } else {
                            cells_used = cell_start;
                        }
                    }
                }

                free_slots[n_free++] = slot_id;
            }
            io_uring_cq_advance(&ring, reaped);
        }

    done:
        *bytes_read_out = bytes_read;
        free(bo);
        for (int s = 0; s < QD; s++) free(slots[s].buf);
        free(slots);
        free(free_slots);
        io_uring_queue_exit(&ring);
        return (int)n_results;
    }

    #else  /* fallback: Linux without liburing, or non-Linux - serial pread */
    #ifdef __linux__
    #pragma message("liburing not found - building without io_uring support. " \
                    "Install liburing-dev for better I/O performance.")
    #include <fcntl.h>
    #include <unistd.h>
    #endif

    typedef struct { uint8_t* buf; uint32_t bidx; uint32_t buf_sz; } UringSlot;

    static int
    uring_read_decode_all(
        int fd,
        const uint64_t* bkt_dbo, const uint32_t* bkt_dbs,
        const uint32_t* bkt_pool_off, const uint32_t* bkt_ne,
        const uint32_t* bkt_hit_count, const uint32_t* bkt_max_hit_pos,
        const uint32_t* lb_pool, uint64_t n_bkts,
        const uint64_t* sort_order,
        const uint64_t* hit_km, const uint32_t* hit_bidx,
        const uint32_t* hit_spos, const uint32_t* hit_dl, uint64_t n_hits,
        const uint64_t* bkt_hit_start, const uint32_t* bkt_hit_cnt,
        uint32_t sparse_threshold,
        uint64_t* out_km, uint32_t* out_cells, uint64_t* out_offsets,
        uint64_t out_cells_cap, uint32_t* tmp_dd, uint32_t tmp_dd_cap,
        uint64_t* bytes_read_out)
    {
        uint32_t max_bsz = 0;
        for (uint64_t i = 0; i < n_bkts; i++) {
            if (bkt_dbs[i] > max_bsz) max_bsz = bkt_dbs[i];
        }
        uint8_t* buf = (uint8_t*)malloc(max_bsz > 0 ? max_bsz : 1);
        if (!buf) return -1;

        uint64_t n_results = 0, cells_used = 0, bytes_read = 0;
        out_offsets[0] = 0;
        uint32_t bo_cap = 4096;
        uint32_t* bo = (uint32_t*)malloc(bo_cap * 4);
        if (!bo) { free(buf); return -1; }

        for (uint64_t bidx = 0; bidx < n_bkts; bidx++) {
            uint32_t dsz = bkt_dbs[bidx];
            if (dsz == 0 || bkt_hit_cnt[bidx] == 0) continue;
            ssize_t rd = pread(fd, buf, dsz, bkt_dbo[bidx]);
            if (rd <= 0) continue;
            bytes_read += dsz;

            const uint8_t* blk = buf;
            int blk_len = (int)dsz;
            const uint32_t* cur_lb = lb_pool + bkt_pool_off[bidx];
            uint32_t cur_ne = bkt_ne[bidx];
            uint32_t cur_hits = bkt_hit_cnt[bidx];

            if (cur_hits > sparse_threshold) {
                if (cur_ne > bo_cap) {
                    bo_cap = cur_ne + 256; free(bo);
                    bo = (uint32_t*)malloc(bo_cap * 4);
                    if (!bo) { free(buf); return -1; }
                }
                build_bo_fast(blk, blk_len, cur_lb, cur_ne, bo, bkt_max_hit_pos[bidx]);
            }

            uint64_t h_start = bkt_hit_start[bidx];
            for (uint32_t h = 0; h < cur_hits; h++) {
                uint64_t idx = sort_order[h_start + h];
                uint32_t pos = hit_spos[idx]; uint32_t dl = hit_dl[idx];
                if (pos >= cur_ne) continue;
                uint32_t byte_off = (cur_hits > sparse_threshold)
                    ? bo[pos] : scan_to_byte_offset(blk, blk_len, cur_lb, pos);
                if (byte_off >= (uint32_t)blk_len) continue;
                if (cells_used + dl > out_cells_cap || dl > tmp_dd_cap) continue;
                int decoded = decode_kmer_cells_fast(blk, blk_len, byte_off, dl, tmp_dd);
                if (decoded <= 0) continue;
                uint64_t cell_start = cells_used;
                out_cells[cells_used++] = tmp_dd[0];
                for (int j = 1; j < decoded; j++)
                    if (tmp_dd[j] != tmp_dd[j-1]) out_cells[cells_used++] = tmp_dd[j];
                if (cells_used > cell_start) {
                    out_km[n_results] = hit_km[idx];
                    out_offsets[n_results + 1] = cells_used;
                    n_results++;
                } else { cells_used = cell_start; }
            }
        }
        *bytes_read_out = bytes_read;
        free(bo); free(buf);
        return (int)n_results;
    }
    #endif /* __linux__ */
    
    """
    void prefetch_range(const void* addr, size_t len) nogil
    void release_range(const void* addr, size_t len) nogil
    void CPU_PREFETCH(const void* addr) nogil
    enum: INITIAL_BUCKET_CAPACITY
    enum: MAX_DATA_PER_BUCKET


    uint32_t decomp_suffixes_only(
        const uint8_t* buf, int buf_len,
        uint32_t* suf_out,
        int* len_section_pos_out) nogil

    void decode_lengths_upto(
        const uint8_t* buf, int buf_len,
        int len_pos,
        uint32_t* len_out,
        uint32_t stop_at) nogil

    void build_bo_fast(
        const uint8_t* blk, int blk_len,
        const uint32_t* lb, uint32_t ne,
        uint32_t* bo, uint32_t stop_at) nogil

    uint32_t scan_to_byte_offset(
        const uint8_t* blk, int blk_len,
        const uint32_t* lb, uint32_t pos) nogil

    int decode_kmer_cells_fast(
        const uint8_t* blk, int blk_len,
        uint32_t byte_off, uint32_t cell_count,
        uint32_t* out) nogil

    int uring_read_decode_all(
        int fd,
        const uint64_t* bkt_dbo,
        const uint32_t* bkt_dbs,
        const uint32_t* bkt_pool_off,
        const uint32_t* bkt_ne,
        const uint32_t* bkt_hit_count,
        const uint32_t* bkt_max_hit_pos,
        const uint32_t* lb_pool,
        uint64_t n_bkts,
        const uint64_t* sort_order,
        const uint64_t* hit_km,
        const uint32_t* hit_bidx,
        const uint32_t* hit_spos,
        const uint32_t* hit_dl,
        uint64_t n_hits,
        const uint64_t* bkt_hit_start,
        const uint32_t* bkt_hit_cnt,
        uint32_t sparse_threshold,
        uint64_t* out_km,
        uint32_t* out_cells,
        uint64_t* out_offsets,
        uint64_t  out_cells_cap,
        uint32_t* tmp_dd,
        uint32_t  tmp_dd_cap,
        uint64_t* bytes_read_out) nogil

cdef extern from *:
    """
    #include <string.h>
    #include <stdlib.h>
    #include <stdint.h>

    typedef struct {
        uint32_t* keys;  /* 0xFFFFFFFF = empty */
        uint32_t* vals;
        uint32_t  mask;  /* table_size - 1, power-of-2 */
        uint32_t  n_unique;
    } OAHash32;

    static inline void oahash_init(OAHash32* h, uint32_t expected) {
        /* Round up to next power of 2, then 2x for load factor <= 0.5 */
        uint32_t sz = 1;
        while (sz < expected * 2 + 16) sz <<= 1;
        h->keys = (uint32_t*)malloc((size_t)sz * 4);
        h->vals = (uint32_t*)malloc((size_t)sz * 4);
        h->mask = sz - 1;
        h->n_unique = 0;
        memset(h->keys, 0xFF, (size_t)sz * 4);  /* fill with sentinel */
    }

    static inline void oahash_free(OAHash32* h) {
        free(h->keys); free(h->vals);
        h->keys = NULL; h->vals = NULL;
    }

    /* Insert-or-lookup.  Returns compact ID for `key`. */
    static inline uint32_t oahash_insert(OAHash32* h, uint32_t key) {
        uint32_t idx = (key * 2654435761u) & h->mask;  /* Knuth multiplicative hash */
        while (1) {
            uint32_t k = h->keys[idx];
            if (k == key) return h->vals[idx];          /* already present */
            if (k == 0xFFFFFFFFu) {                     /* empty slot */
                h->keys[idx] = key;
                h->vals[idx] = h->n_unique;
                return h->n_unique++;
            }
            idx = (idx + 1) & h->mask;
        }
    }

    static inline int64_t bsearch_u64(const uint64_t* arr, int64_t n, uint64_t target) {
        int64_t lo = 0, hi = n - 1, mid;
        while (lo <= hi) {
            mid = (lo + hi) >> 1;
            if (arr[mid] == target) return mid;
            else if (arr[mid] < target) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    static inline uint32_t remap_cells_flat(
        uint32_t* __restrict__ cells, uint64_t n_cells,
        OAHash32* h)
    {
        for (uint64_t i = 0; i < n_cells; i++) {
            cells[i] = oahash_insert(h, cells[i]);
        }
        return h->n_unique;
    }
    """
    ctypedef struct OAHash32:
        uint32_t* keys
        uint32_t* vals
        uint32_t  mask
        uint32_t  n_unique

    void oahash_init(OAHash32* h, uint32_t expected) nogil
    void oahash_free(OAHash32* h) nogil
    uint32_t oahash_insert(OAHash32* h, uint32_t key) nogil
    int64_t bsearch_u64(const uint64_t* arr, int64_t n, uint64_t target) nogil
    uint32_t remap_cells_flat(uint32_t* cells, uint64_t n_cells, OAHash32* h) nogil


cdef uint32_t MAGIC = 0x4D4C5650
cdef uint32_t VERSION = 2

# Threshold for sparse-hit path in phase2.
# Buckets with <= this many hits use scan_to_byte_offset per hit.
# Buckets with more hits use build_bo_fast (amortises the full scan).
# Value of 4 means: 1-4 hits -> sparse path, 5+ hits -> dense path.
# Given ~99% of buckets are single-hit, almost all buckets use sparse path.
DEF SPARSE_HIT_THRESHOLD = 4

cdef bint compare_kmer_cell(const pair[uint64_t, uint32_t]& a,
                             const pair[uint64_t, uint32_t]& b) nogil:
    if a.first != b.first: return a.first < b.first
    return a.second < b.second


cdef inline int _encode_varint(uint32_t val, uint8_t* buf) nogil:
    cdef int n = 0
    while val >= 0x80:
        buf[n] = <uint8_t>((val & 0x7F) | 0x80); n += 1; val >>= 7
    buf[n] = <uint8_t>(val); n += 1
    return n

cdef inline uint32_t _decode_varint(const uint8_t* buf, int* bytes_read) nogil:
    cdef uint32_t val = 0, shift = 0
    cdef int n = 0
    cdef uint8_t b
    while True:
        b = buf[n]; n += 1
        val |= (<uint32_t>(b & 0x7F)) << shift
        if (b & 0x80) == 0: break
        shift += 7
        if shift >= 35: break
    bytes_read[0] = n
    return val

cdef inline int _skip_n_varints_fast(const uint8_t* p, int max_bytes,
                                      uint32_t count) nogil:
    cdef int pos = 0
    cdef uint32_t found = 0
    while found < count and pos < max_bytes:
        if (p[pos] & 0x80) == 0: found += 1
        pos += 1
    return pos


cdef class PrefixIndex:

    def __cinit__(self):
        self.pi_suffix_offsets = NULL; self.pi_data_offsets = NULL
        self.pi_data_sizes = NULL; self.suffix_mmap = NULL
        self.data_mmap = NULL; self.data_size = 0
        self.data_fd = -1; self.is_open = False
        self._last_query_times = {}
        self._csr_mode = False
        self._csr_km = None
        self._csr_cells = None
        self._csr_offsets = None
        self._csr_n_results = 0

    def __dealloc__(self): self.close()

    def open(self, str index_dir, str mode='r'):
        self.index_dir = index_dir
        self.pi_path      = os.path.join(index_dir, 'pi.bin')
        self.suffix_path  = os.path.join(index_dir, 'suffixes.bin')
        self.data_path    = os.path.join(index_dir, 'data.bin')
        self.meta_path    = os.path.join(index_dir, 'meta.json')
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Not found: {self.meta_path}")
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        self.l_prefix    = meta['l_prefix']
        self.l_suffix    = meta['l_suffix']
        self.kmer_size   = meta['kmer_size']
        self.n_kmers     = meta.get('n_kmers', 0)
        self.n_cells     = meta.get('n_cells', 0)
        self.n_prefixes  = 1 << (2 * self.l_prefix)
        self.rshift      = 2 * self.l_suffix
        self.suffix_mask = (1 << (2 * self.l_suffix)) - 1

        pi_mm = np.memmap(self.pi_path, dtype=np.uint8, mode='r')
        self._pi_arr_ref = pi_mm
        cdef const unsigned char* pp = <const unsigned char*>(<np.ndarray>pi_mm).data
        cdef uint64_t oss = self.n_prefixes * 8
        self.pi_suffix_offsets = <uint64_t*>(pp)
        self.pi_data_offsets   = <uint64_t*>(pp + oss)
        self.pi_data_sizes     = <uint32_t*>(pp + 2 * oss)

        cdef uint64_t ss = os.path.getsize(self.suffix_path)
        if ss > 0:
            suf_mm = np.memmap(self.suffix_path, dtype=np.uint8, mode='r')
            self._suf_arr_ref = suf_mm
            self.suffix_mmap = <const unsigned char*>(<np.ndarray>suf_mm).data
        else:
            self.suffix_mmap = NULL; self._suf_arr_ref = None

        cdef uint64_t ds = os.path.getsize(self.data_path)
        self.data_size = ds
        if ds > 0:
            dat_mm = np.memmap(self.data_path, dtype=np.uint8, mode='r')
            self._dat_arr_ref = dat_mm
            self.data_mmap = <const unsigned char*>(<np.ndarray>dat_mm).data
        else:
            self.data_mmap = NULL; self._dat_arr_ref = None

        self.data_file_obj = None; self.data_fd = -1; self.is_open = True
        logging.info(f"Opened prefix index: {self.n_kmers:,} kmers, "
                     f"l_prefix={self.l_prefix}, data={ds/1e9:.1f}GB")

    def close(self):
        self._pi_arr_ref = None; self._suf_arr_ref = None; self._dat_arr_ref = None
        self.pi_suffix_offsets = NULL; self.pi_data_offsets = NULL
        self.pi_data_sizes = NULL; self.suffix_mmap = NULL
        self.data_mmap = NULL; self.data_fd = -1; self.is_open = False

    @property
    def is_loaded(self): return self.is_open

    cdef int _decompress_suffix_bucket(self, uint64_t so, uint32_t* s,
                                        uint64_t* d, uint32_t* l, uint32_t capacity):
        cdef const unsigned char* p = self.suffix_mmap + so
        cdef uint32_t n, i, prev_suf, delta, cum_offset
        cdef int br
        memcpy(&n, p, 4); p += 4
        if n > capacity: return -<int>n
        if n > 0:
            memcpy(&prev_suf, p, 4); p += 4
            s[0] = prev_suf
            for i in range(1, n):
                delta = _decode_varint(<const uint8_t*>p, &br); p += br
                prev_suf += delta; s[i] = prev_suf
            cum_offset = 0
            for i in range(n):
                l[i] = _decode_varint(<const uint8_t*>p, &br); p += br
                d[i] = <uint64_t>cum_offset
                cum_offset += l[i]
        return <int>n

    cdef const uint8_t* _get_compressed_data_ptr(self, uint64_t off,
                                                   uint32_t sz) nogil:
        if self.data_mmap != NULL and off + sz <= self.data_size:
            return <const uint8_t*>(self.data_mmap + off)
        return NULL

    cdef int _skip_n_varints(self, const uint8_t* p, int max_bytes,
                              uint32_t count) nogil:
        return _skip_n_varints_fast(p, max_bytes, count)

    cdef int _decode_data_block(self, const uint8_t* compressed, int compressed_size,
                                 uint32_t* lengths, int n_entries,
                                 uint32_t* out, uint32_t cap):
        cdef int br, pos = 0
        cdef uint32_t i, j, total = 0, prev_cell, delta, cell_count
        for i in range(n_entries):
            cell_count = lengths[i]
            if total + cell_count > cap: break
            if cell_count > 0 and pos + 4 <= compressed_size:
                memcpy(&prev_cell, &compressed[pos], 4); pos += 4
                out[total] = prev_cell; total += 1
                for j in range(1, cell_count):
                    if pos >= compressed_size: break
                    delta = _decode_varint(&compressed[pos], &br); pos += br
                    prev_cell += delta; out[total] = prev_cell; total += 1
        return <int>total

    cdef int _binary_search_u32(self, const uint32_t* a, int n, uint32_t t):
        cdef int lo = 0, hi = n - 1, mid
        while lo <= hi:
            mid = (lo + hi) >> 1
            if a[mid] == t: return mid
            elif a[mid] < t: lo = mid + 1
            else: hi = mid - 1
        return -1

    cdef int _decode_single_kmer_data(self, const uint8_t* p, int max_bytes,
                                       uint32_t cell_count,
                                       uint32_t* out) nogil:
        return decode_kmer_cells_fast(p, max_bytes, 0, cell_count, out)

    cdef int _stream_lookup_and_decode(self, uint64_t so, uint32_t target_suffix,
                                        uint32_t cam, uint32_t cal,
                                        const uint8_t* data_block, int data_bytes,
                                        uint32_t* out, uint32_t out_cap) nogil:
        cdef const unsigned char* p = self.suffix_mmap + so
        cdef uint32_t n, i, j, prev_suf, delta, length_i, target_length
        cdef int br, found_pos = -1
        memcpy(&n, p, 4); p += 4
        if n == 0: return 0
        memcpy(&prev_suf, p, 4); p += 4
        if prev_suf == target_suffix: found_pos = 0
        elif prev_suf > target_suffix: return 0
        else:
            for i in range(1, n):
                delta = _decode_varint(<const uint8_t*>p, &br); p += br
                prev_suf += delta
                if prev_suf == target_suffix: found_pos = <int>i; break
                elif prev_suf > target_suffix: return 0
            if found_pos < 0: return 0
        for i in range(<uint32_t>(found_pos + 1), n):
            _decode_varint(<const uint8_t*>p, &br); p += br
        cdef int data_pos = 0
        for i in range(<uint32_t>found_pos):
            length_i = _decode_varint(<const uint8_t*>p, &br); p += br
            if length_i > 0 and data_pos + 4 <= data_bytes:
                data_pos += 4
                for j in range(1, length_i):
                    if data_pos >= data_bytes: break
                    _decode_varint(&data_block[data_pos], &br); data_pos += br
        target_length = _decode_varint(<const uint8_t*>p, &br)
        if target_length >= cam or target_length <= cal: return 0
        if target_length == 0: return 0
        if target_length > out_cap: return 0
        return decode_kmer_cells_fast(data_block, data_bytes,
                                      <uint32_t>data_pos, target_length, out)

    cdef unordered_map[uint64_t, vector[uint32_t]] query_batch(
            self, uint64_t* qk, uint64_t nq,
            uint32_t cam, uint32_t cal):
        """Batch k-mer query using prefix-bucketed index with io_uring I/O."""
        cdef unordered_map[uint64_t, vector[uint32_t]] res
        if self.pi_suffix_offsets == NULL or self.suffix_mmap == NULL or nq == 0:
            return res

        cdef double t0_total = time.perf_counter()
        cdef double t0
        cdef double acc_suf_read = 0.0, acc_suf = 0.0, acc_bsearch = 0.0
        cdef double acc_dat_read = 0.0, acc_phase2 = 0.0
        cdef double t_total
        cdef uint64_t n_suf_bytes = 0, n_dat_bytes = 0, n_buckets_touched = 0

        cdef uint64_t i, j, km, pf, so, dbo, qi, qj
        cdef uint64_t bki, ob
        cdef uint32_t sv, dbs, dl, ki
        cdef int pos, ne, ne_ret, decoded
        cdef vector[uint32_t] cl
        cdef bint any_hit

        cdef np.ndarray _csr_km_arr, _csr_offsets_arr, _csr_cells_arr
        cdef uint64_t _total_cells

        cdef uint64_t bkt_cap = min(nq + 256, <uint64_t>16777216)
        cdef uint64_t n_bkts = 0
        cdef uint64_t* bkt_so   = <uint64_t*>malloc(bkt_cap * 8)
        cdef uint64_t* bkt_dbo  = <uint64_t*>malloc(bkt_cap * 8)
        cdef uint32_t* bkt_dbs  = <uint32_t*>malloc(bkt_cap * 4)
        cdef uint32_t* bkt_ne   = <uint32_t*>malloc(bkt_cap * 4)
        cdef uint32_t* bkt_peek = <uint32_t*>malloc(bkt_cap * 4)
        cdef unordered_map[uint64_t, uint32_t] dbo_to_bidx
        cdef uint32_t peek_ne_val

        cdef uint32_t suf_sz = 0
        cdef uint32_t* bkt_suf_sz = NULL

        cdef uint32_t* bkt_pool_off = NULL
        cdef uint64_t  pool_total   = 0
        cdef uint32_t* sb_pool      = NULL
        cdef uint32_t* lb_pool      = NULL
        cdef int*      bkt_len_sec  = NULL
        cdef uint32_t  actual_ne
        cdef uint32_t* bkt_max_hit_pos = NULL

        cdef uint64_t hit_cap = min(nq * 4, <uint64_t>4194304)
        cdef uint64_t n_hits = 0
        cdef uint64_t* hit_km   = NULL
        cdef uint32_t* hit_bidx = NULL
        cdef uint32_t* hit_spos = NULL
        cdef uint32_t* hit_dl   = NULL
        cdef uint32_t  bidx
        cdef uint32_t* sb_ptr  = NULL
        cdef uint32_t* lb_ptr2 = NULL
        cdef uint32_t* lb_ptr3 = NULL

        cdef uint64_t  prov_cap  = 0
        cdef uint64_t  n_prov    = 0
        cdef uint64_t* prov_km   = NULL
        cdef uint32_t* prov_bidx = NULL
        cdef uint32_t* prov_spos = NULL

        cdef uint32_t* bkt_hit_count = NULL

        cdef uint64_t* p2_km         = NULL
        cdef uint32_t* p2_cells      = NULL
        cdef uint64_t* p2_offsets    = NULL
        cdef uint64_t  p2_cells_cap  = 0
        cdef uint32_t  dd_cap        = 65536
        cdef uint32_t* dd            = NULL
        cdef uint64_t  idx2, hi
        cdef uint64_t  r_start, r_end, rr
        cdef int       p2_n_results  = 0
        cdef uint64_t  total_dl      = 0
        cdef uint64_t* so_v_ptr      = NULL
        cdef uint64_t* bkt_hit_start = NULL
        cdef uint32_t* bkt_hit_cnt   = NULL
        cdef uint64_t  bytes_rd      = 0
        cdef int       data_fd       = -1
        cdef uint64_t  running       = 0

        cdef uint64_t[:] suf_order
        cdef uint64_t[:] dat_order
        cdef uint64_t[:] sk_v
        cdef uint64_t[:] so_v

        if not bkt_so or not bkt_dbo or not bkt_dbs or not bkt_ne or not bkt_peek:
            free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne); free(bkt_peek)
            raise MemoryError()

        qi = 0
        while qi < nq:
            pf = qk[qi] >> self.rshift
            qj = qi + 1
            while qj < nq and (qk[qj] >> self.rshift) == pf: qj += 1
            if pf < self.n_prefixes:
                so  = self.pi_suffix_offsets[pf]
                dbo = self.pi_data_offsets[pf]
                dbs = self.pi_data_sizes[pf]
                if so > 0 and dbs > 0 and dbo_to_bidx.find(dbo) == dbo_to_bidx.end():
                    if n_bkts >= bkt_cap:
                        bkt_cap *= 2
                        bkt_so   = <uint64_t*>realloc(bkt_so,   bkt_cap * 8)
                        bkt_dbo  = <uint64_t*>realloc(bkt_dbo,  bkt_cap * 8)
                        bkt_dbs  = <uint32_t*>realloc(bkt_dbs,  bkt_cap * 4)
                        bkt_ne   = <uint32_t*>realloc(bkt_ne,   bkt_cap * 4)
                        bkt_peek = <uint32_t*>realloc(bkt_peek, bkt_cap * 4)
                        if not bkt_so or not bkt_dbo or not bkt_dbs \
                                or not bkt_ne or not bkt_peek:
                            free(bkt_so); free(bkt_dbo); free(bkt_dbs)
                            free(bkt_ne); free(bkt_peek)
                            raise MemoryError()
                    memcpy(&peek_ne_val, self.suffix_mmap + so, 4)
                    dbo_to_bidx[dbo] = <uint32_t>n_bkts
                    bkt_so[n_bkts]   = so
                    bkt_dbo[n_bkts]  = dbo
                    bkt_dbs[n_bkts]  = dbs
                    bkt_ne[n_bkts]   = 0
                    bkt_peek[n_bkts] = peek_ne_val
                    n_bkts += 1
            qi = qj

        if n_bkts == 0:
            free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne); free(bkt_peek)
            self._last_query_times = {
                'n_kmers_in': int(nq), 'n_buckets_touched': 0, 'n_hits': 0,
                't_suf_read': 0.0, 't_suffix_decomp': 0.0, 't_bsearch': 0.0,
                't_dat_read': 0.0, 't_phase2': 0.0,
                't_total': time.perf_counter() - t0_total,
                'n_suf_bytes': 0, 'n_dat_bytes': 0,
            }
            return res

        t0 = time.perf_counter()

        so_arr_np  = np.asarray(<uint64_t[:n_bkts]>bkt_so)
        suf_order  = np.argsort(so_arr_np, kind='stable').astype(np.uint64)

        bkt_suf_sz = <uint32_t*>malloc(n_bkts * 4)
        if not bkt_suf_sz:
            free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne); free(bkt_peek)
            raise MemoryError()
        for bki in range(n_bkts):
            bkt_suf_sz[bki] = 8 + 10 * bkt_peek[bki]

        with nogil:
            for bki in range(n_bkts):
                ob = suf_order[bki]
                prefetch_range(<const void*>(self.suffix_mmap + bkt_so[ob]),
                               <size_t>bkt_suf_sz[ob])
        acc_suf_read += time.perf_counter() - t0


        t0 = time.perf_counter()

        bkt_pool_off    = <uint32_t*>malloc(n_bkts * 4)
        bkt_max_hit_pos = <uint32_t*>malloc(n_bkts * 4)
        bkt_hit_count   = <uint32_t*>malloc(n_bkts * 4)
        bkt_len_sec     = <int*>malloc(n_bkts * 4)
        if not bkt_pool_off or not bkt_max_hit_pos or not bkt_hit_count or not bkt_len_sec:
            free(bkt_suf_sz)
            free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne); free(bkt_peek)
            free(bkt_pool_off); free(bkt_max_hit_pos); free(bkt_hit_count); free(bkt_len_sec)
            raise MemoryError()

        pool_total = 0
        for bki in range(n_bkts):
            bkt_pool_off[bki]    = <uint32_t>pool_total
            bkt_max_hit_pos[bki] = 0xFFFFFFFF
            bkt_hit_count[bki]   = 0
            bkt_len_sec[bki]     = 0
            pool_total += bkt_peek[bki]

        sb_pool = <uint32_t*>malloc((pool_total + 1) * 4)
        lb_pool = <uint32_t*>malloc((pool_total + 1) * 4)
        if not sb_pool or not lb_pool:
            free(bkt_suf_sz)
            free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne); free(bkt_peek)
            free(bkt_pool_off); free(bkt_max_hit_pos); free(bkt_hit_count); free(bkt_len_sec)
            free(sb_pool); free(lb_pool)
            raise MemoryError()

        with nogil:
            for bki in range(n_bkts):
                actual_ne = decomp_suffixes_only(
                    <const uint8_t*>(self.suffix_mmap + bkt_so[bki]),
                    <int>bkt_suf_sz[bki],
                    sb_pool + bkt_pool_off[bki],
                    &bkt_len_sec[bki])
                bkt_ne[bki] = actual_ne

        with nogil:
            for bki in range(n_bkts):
                release_range(<const void*>(self.suffix_mmap + bkt_so[bki]),
                              <size_t>bkt_suf_sz[bki])
        free(bkt_peek)
        acc_suf += time.perf_counter() - t0

        # Record provisional hits (suffix match) - no count filter yet since
        # lengths not decoded. We record (km, bidx, spos) and defer dl lookup.
        t0 = time.perf_counter()

        prov_cap = min(nq * 4, <uint64_t>4194304)
        n_prov   = 0
        prov_km   = <uint64_t*>malloc(prov_cap * 8)
        prov_bidx = <uint32_t*>malloc(prov_cap * 4)
        prov_spos = <uint32_t*>malloc(prov_cap * 4)

        hit_km   = <uint64_t*>malloc(hit_cap * 8)
        hit_bidx = <uint32_t*>malloc(hit_cap * 4)
        hit_spos = <uint32_t*>malloc(hit_cap * 4)
        hit_dl   = <uint32_t*>malloc(hit_cap * 4)

        if not prov_km or not prov_bidx or not prov_spos or not hit_km or not hit_bidx or not hit_spos or not hit_dl:
            free(prov_km); free(prov_bidx); free(prov_spos)
            free(sb_pool); free(lb_pool); free(bkt_pool_off)
            free(bkt_max_hit_pos); free(bkt_hit_count); free(bkt_len_sec)
            free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne)
            free(hit_km); free(hit_bidx); free(hit_spos); free(hit_dl)
            free(bkt_suf_sz)
            raise MemoryError()

        qi = 0
        while qi < nq:
            pf = qk[qi] >> self.rshift
            qj = qi + 1
            while qj < nq and (qk[qj] >> self.rshift) == pf: qj += 1
            if pf >= self.n_prefixes: qi = qj; continue
            dbo = self.pi_data_offsets[pf]
            if self.pi_data_sizes[pf] == 0: qi = qj; continue
            if dbo_to_bidx.find(dbo) == dbo_to_bidx.end(): qi = qj; continue
            bidx = dbo_to_bidx[dbo]
            ne   = <int>bkt_ne[bidx]
            if ne <= 0: qi = qj; continue
            sb_ptr = sb_pool + bkt_pool_off[bidx]
            for i in range(qi, qj):
                km = qk[i]
                sv = <uint32_t>(km & self.suffix_mask)
                pos = self._binary_search_u32(sb_ptr, ne, sv)
                if pos < 0: continue
                if bkt_max_hit_pos[bidx] == 0xFFFFFFFF or <uint32_t>pos > bkt_max_hit_pos[bidx]:
                    bkt_max_hit_pos[bidx] = <uint32_t>pos
                if n_prov >= prov_cap:
                    prov_cap *= 2
                    prov_km   = <uint64_t*>realloc(prov_km,   prov_cap * 8)
                    prov_bidx = <uint32_t*>realloc(prov_bidx, prov_cap * 4)
                    prov_spos = <uint32_t*>realloc(prov_spos, prov_cap * 4)
                    if not prov_km or not prov_bidx or not prov_spos:
                        free(prov_km); free(prov_bidx); free(prov_spos)
                        free(sb_pool); free(lb_pool); free(bkt_pool_off)
                        free(bkt_max_hit_pos); free(bkt_hit_count); free(bkt_len_sec)
                        free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne)
                        free(hit_km); free(hit_bidx); free(hit_spos); free(hit_dl)
                        free(bkt_suf_sz)
                        raise MemoryError()
                prov_km[n_prov]   = km
                prov_bidx[n_prov] = bidx
                prov_spos[n_prov] = <uint32_t>pos
                n_prov += 1
            qi = qj

        free(sb_pool)

        with nogil:
            for bki in range(n_bkts):
                if bkt_max_hit_pos[bki] != <uint32_t>0xFFFFFFFF:
                    decode_lengths_upto(
                        <const uint8_t*>(self.suffix_mmap + bkt_so[bki]),
                        <int>bkt_suf_sz[bki],
                        bkt_len_sec[bki],
                        lb_pool + bkt_pool_off[bki],
                        bkt_max_hit_pos[bki])
        free(bkt_len_sec)
        free(bkt_suf_sz)

        for i in range(n_prov):
            bidx = prov_bidx[i]
            pos  = <int>prov_spos[i]
            lb_ptr3 = lb_pool + bkt_pool_off[bidx]
            dl = lb_ptr3[pos]
            if dl >= cam or dl <= cal: continue
            if bkt_hit_count[bidx] == 0: n_buckets_touched += 1
            bkt_hit_count[bidx] += 1
            if n_hits >= hit_cap:
                hit_cap *= 2
                hit_km   = <uint64_t*>realloc(hit_km,   hit_cap * 8)
                hit_bidx = <uint32_t*>realloc(hit_bidx, hit_cap * 4)
                hit_spos = <uint32_t*>realloc(hit_spos, hit_cap * 4)
                hit_dl   = <uint32_t*>realloc(hit_dl,   hit_cap * 4)
                if not hit_km or not hit_bidx or not hit_spos or not hit_dl:
                    free(prov_km); free(prov_bidx); free(prov_spos)
                    free(lb_pool); free(bkt_pool_off)
                    free(bkt_max_hit_pos); free(bkt_hit_count)
                    free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne)
                    free(hit_km); free(hit_bidx); free(hit_spos); free(hit_dl)
                    raise MemoryError()
            hit_km[n_hits]   = prov_km[i]
            hit_bidx[n_hits] = bidx
            hit_spos[n_hits] = prov_spos[i]
            hit_dl[n_hits]   = dl
            n_hits += 1

        free(prov_km); free(prov_bidx); free(prov_spos)
        acc_bsearch += time.perf_counter() - t0

        if n_hits == 0:
            free(lb_pool); free(bkt_pool_off)
            free(bkt_max_hit_pos); free(bkt_hit_count)
            free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne)
            free(hit_km); free(hit_bidx); free(hit_spos); free(hit_dl)
            self._last_query_times = {
                'n_kmers_in': int(nq), 'n_buckets_touched': 0, 'n_hits': 0,
                't_suf_read': acc_suf_read, 't_suffix_decomp': acc_suf,
                't_bsearch': acc_bsearch, 't_dat_read': 0.0, 't_phase2': 0.0,
                't_total': time.perf_counter() - t0_total,
                'n_suf_bytes': int(n_suf_bytes), 'n_dat_bytes': 0,
            }
            return res

        t0 = time.perf_counter()

        sort_keys_np = np.empty(n_hits, dtype=np.uint64)
        sk_v = sort_keys_np
        for hi in range(n_hits):
            sk_v[hi] = (<uint64_t>hit_bidx[hi] << 32) | <uint64_t>hit_spos[hi]
        so_v = np.argsort(sort_keys_np, kind='stable').astype(np.uint64)
        so_v_ptr = <uint64_t*>&so_v[0]

        bkt_hit_start = <uint64_t*>malloc(n_bkts * 8)
        bkt_hit_cnt   = <uint32_t*>malloc(n_bkts * 4)
        if not bkt_hit_start or not bkt_hit_cnt:
            free(bkt_hit_start); free(bkt_hit_cnt)
            free(lb_pool); free(bkt_pool_off)
            free(bkt_max_hit_pos); free(bkt_hit_count)
            free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne)
            free(hit_km); free(hit_bidx); free(hit_spos); free(hit_dl)
            raise MemoryError()
        memset(bkt_hit_start, 0, n_bkts * 8)
        memset(bkt_hit_cnt,   0, n_bkts * 4)
        for hi in range(n_hits):
            bkt_hit_cnt[hit_bidx[so_v_ptr[hi]]] += 1
        running = 0
        for bki in range(n_bkts):
            bkt_hit_start[bki] = running
            running += bkt_hit_cnt[bki]

        total_dl = 0
        for hi in range(n_hits):
            total_dl += hit_dl[hi]
        p2_km      = <uint64_t*>malloc(n_hits * 8)
        p2_offsets = <uint64_t*>malloc((n_hits + 1) * 8)
        p2_cells_cap = total_dl if total_dl > 0 else 1
        p2_cells   = <uint32_t*>malloc(p2_cells_cap * 4)
        dd         = <uint32_t*>malloc(dd_cap * 4)
        if not p2_km or not p2_offsets or not p2_cells or not dd:
            free(p2_km); free(p2_offsets); free(p2_cells); free(dd)
            free(bkt_hit_start); free(bkt_hit_cnt)
            free(lb_pool); free(bkt_pool_off)
            free(bkt_max_hit_pos); free(bkt_hit_count)
            free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne)
            free(hit_km); free(hit_bidx); free(hit_spos); free(hit_dl)
            raise MemoryError()

        try:
            data_fd = os.open(self.data_path, os.O_RDONLY)
        except OSError as e:
            free(p2_km); free(p2_offsets); free(p2_cells); free(dd)
            free(bkt_hit_start); free(bkt_hit_cnt)
            free(lb_pool); free(bkt_pool_off)
            free(bkt_max_hit_pos); free(bkt_hit_count)
            free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne)
            free(hit_km); free(hit_bidx); free(hit_spos); free(hit_dl)
            raise

        bytes_rd = 0
        with nogil:
            p2_n_results = uring_read_decode_all(
                data_fd,
                bkt_dbo, bkt_dbs,
                bkt_pool_off, bkt_ne,
                bkt_hit_count, bkt_max_hit_pos,
                lb_pool, n_bkts,
                so_v_ptr,
                hit_km, hit_bidx, hit_spos, hit_dl, n_hits,
                bkt_hit_start, bkt_hit_cnt,
                SPARSE_HIT_THRESHOLD,
                p2_km, p2_cells, p2_offsets, p2_cells_cap,
                dd, dd_cap,
                &bytes_rd)
        os.close(data_fd)
        n_dat_bytes = bytes_rd

        acc_dat_read += time.perf_counter() - t0

        free(bkt_hit_start); free(bkt_hit_cnt)

        if p2_n_results < 0:
            free(p2_km); free(p2_offsets); free(p2_cells); free(dd)
            free(lb_pool); free(bkt_pool_off)
            free(bkt_max_hit_pos); free(bkt_hit_count)
            free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne)
            free(hit_km); free(hit_bidx); free(hit_spos); free(hit_dl)
            raise RuntimeError(f"uring_read_decode_all failed: {p2_n_results}")

        t0 = time.perf_counter()

        if self._csr_mode:
            _csr_km_arr = np.empty(p2_n_results, dtype=np.uint64)
            _csr_offsets_arr = np.empty(p2_n_results + 1, dtype=np.uint64)
            _total_cells = p2_offsets[p2_n_results] if p2_n_results > 0 else 0
            _csr_cells_arr = np.empty(_total_cells, dtype=np.uint32)
            if p2_n_results > 0:
                memcpy(<void*>(<np.ndarray>_csr_km_arr).data, p2_km, p2_n_results * 8)
                memcpy(<void*>(<np.ndarray>_csr_offsets_arr).data, p2_offsets, (p2_n_results + 1) * 8)
                memcpy(<void*>(<np.ndarray>_csr_cells_arr).data, p2_cells, _total_cells * 4)
            else:
                (<np.ndarray>_csr_offsets_arr)[0] = 0
            self._csr_km = _csr_km_arr
            self._csr_cells = _csr_cells_arr
            self._csr_offsets = _csr_offsets_arr
            self._csr_n_results = int(p2_n_results)
        else:
            for rr in range(<uint64_t>p2_n_results):
                r_start = p2_offsets[rr]
                r_end   = p2_offsets[rr + 1]
                if r_end <= r_start: continue
                cl.assign(p2_cells + r_start, p2_cells + r_end)
                res[p2_km[rr]] = cl

        acc_phase2 += time.perf_counter() - t0

        free(dd); free(p2_km); free(p2_offsets); free(p2_cells)
        free(lb_pool); free(bkt_pool_off)
        free(bkt_max_hit_pos); free(bkt_hit_count)
        free(bkt_so); free(bkt_dbo); free(bkt_dbs); free(bkt_ne)
        free(hit_km); free(hit_bidx); free(hit_spos); free(hit_dl)

        t_total = time.perf_counter() - t0_total
        self._last_query_times = {
            'n_kmers_in':        int(nq),
            'n_buckets_touched': int(n_buckets_touched),
            'n_hits':            int(n_hits),
            'n_suf_bytes':       int(n_suf_bytes),
            'n_dat_bytes':       int(n_dat_bytes),
            't_suf_read':        acc_suf_read,
            't_suffix_decomp':   acc_suf,
            't_bsearch':         acc_bsearch,
            't_dat_read':        acc_dat_read,
            't_phase2':          acc_phase2,
            't_total':           t_total,
            't_unaccounted':     t_total - acc_suf_read - acc_suf - acc_bsearch
                                         - acc_dat_read - acc_phase2,
        }
        return res

    def query(self, np.ndarray[np.uint64_t, ndim=1] kmers,
              uint32_t count_at_most=10000, uint32_t count_at_least=10):
        if not self.is_open: raise RuntimeError("Not open")
        if len(kmers) == 0 or self.n_kmers == 0: return {}
        cdef np.ndarray[np.uint64_t, ndim=1] sk = np.sort(kmers)
        cdef unordered_map[uint64_t, vector[uint32_t]] rr = self.query_batch(
            <uint64_t*>sk.data, <uint64_t>len(sk), count_at_most, count_at_least)
        t = self._last_query_times
        if t:
            logging.info(
                f"query_batch timing | kmers_in={t.get('n_kmers_in',0):,} "
                f"buckets={t.get('n_buckets_touched',0):,} hits={t.get('n_hits',0):,} "
                f"suf={t.get('n_suf_bytes',0)/1e6:.1f}MB dat={t.get('n_dat_bytes',0)/1e6:.1f}MB | "
                f"suf_read={t.get('t_suf_read',0)*1000:.1f}ms "
                f"suf_decomp={t.get('t_suffix_decomp',0)*1000:.1f}ms "
                f"bsearch={t.get('t_bsearch',0)*1000:.1f}ms "
                f"dat_read={t.get('t_dat_read',0)*1000:.1f}ms "
                f"phase2={t.get('t_phase2',0)*1000:.1f}ms "
                f"unaccounted={t.get('t_unaccounted',0)*1000:.1f}ms "
                f"total={t.get('t_total',0)*1000:.1f}ms"
            )
        cdef unordered_map[uint64_t, vector[uint32_t]].iterator it = rr.begin()
        cdef uint64_t key
        cdef vector[uint32_t] vals
        cdef dict r = {}
        cdef size_t vs
        while it != rr.end():
            key = deref(it).first; vals = deref(it).second; vs = vals.size()
            a = np.empty(vs, dtype=np.uint32)
            if vs > 0: memcpy(<void*>(<np.ndarray>a).data, &vals[0], vs * 4)
            r[key] = a; inc(it)
        return r

    cdef unordered_map[uint64_t, vector[uint32_t]] find_kmer_c(
            self, np.ndarray kmers, uint64_t cam, uint32_t cal):
        cdef np.ndarray[np.uint64_t, ndim=1] sk = np.sort(
            kmers.astype(np.uint64))
        cdef unordered_map[uint64_t, vector[uint32_t]] rr = self.query_batch(
            <uint64_t*>sk.data, <uint64_t>len(sk), <uint32_t>cam, cal)
        t = self._last_query_times
        if t:
            logging.info(
                f"query_batch timing | kmers_in={t.get('n_kmers_in',0):,} "
                f"buckets={t.get('n_buckets_touched',0):,} hits={t.get('n_hits',0):,} "
                f"suf={t.get('n_suf_bytes',0)/1e6:.1f}MB dat={t.get('n_dat_bytes',0)/1e6:.1f}MB | "
                f"suf_read={t.get('t_suf_read',0)*1000:.1f}ms "
                f"suf_decomp={t.get('t_suffix_decomp',0)*1000:.1f}ms "
                f"bsearch={t.get('t_bsearch',0)*1000:.1f}ms "
                f"dat_read={t.get('t_dat_read',0)*1000:.1f}ms "
                f"phase2={t.get('t_phase2',0)*1000:.1f}ms "
                f"unaccounted={t.get('t_unaccounted',0)*1000:.1f}ms "
                f"total={t.get('t_total',0)*1000:.1f}ms"
            )
        return rr

    def query_batch_csr(self, np.ndarray[np.uint64_t, ndim=1] kmers,
                        uint32_t count_at_most=10000, uint32_t count_at_least=10):
        """Query returning CSR arrays instead of unordered_map.

        Returns (kmer_keys, cell_data, cell_indptr, orig_cell_ids, N_unique) where:
          kmer_keys[N_found]       - sorted kmer values (uint64)
          cell_data[total_cells]   - flat packed remapped cell IDs (uint32)
          cell_indptr[N_found+1]   - CSR offsets (uint64)
          orig_cell_ids[N_unique]  - compact_id -> original cell ID mapping
          N_unique                 - number of unique cells

        Uses _csr_mode flag to bypass unordered_map construction inside query_batch.
        """
        if not self.is_open: raise RuntimeError("Not open")
        if len(kmers) == 0:
            return (np.empty(0, dtype=np.uint64),
                    np.empty(0, dtype=np.uint32),
                    np.zeros(1, dtype=np.uint64),
                    np.empty(0, dtype=np.uint32), 0)
        cdef np.ndarray[np.uint64_t, ndim=1] sk = np.sort(kmers)

        self._csr_mode = True
        try:
            self.query_batch(
                <uint64_t*>sk.data, <uint64_t>len(sk), count_at_most, count_at_least)
        finally:
            self._csr_mode = False

        cdef int p2_n = self._csr_n_results
        if p2_n == 0:
            self._csr_km = None; self._csr_cells = None; self._csr_offsets = None
            return (np.empty(0, dtype=np.uint64),
                    np.empty(0, dtype=np.uint32),
                    np.zeros(1, dtype=np.uint64),
                    np.empty(0, dtype=np.uint32), 0)

        cdef double t0 = time.perf_counter()

        cdef np.ndarray[np.uint64_t, ndim=1] raw_km = self._csr_km
        cdef np.ndarray[np.uint32_t, ndim=1] raw_cells = self._csr_cells
        cdef np.ndarray[np.uint64_t, ndim=1] raw_offsets = self._csr_offsets
        self._csr_km = None; self._csr_cells = None; self._csr_offsets = None

        cdef uint64_t n_found = <uint64_t>p2_n
        cdef uint64_t total_cells = raw_offsets[n_found]

        cdef np.ndarray sort_idx = np.argsort(raw_km)
        cdef np.ndarray[np.uint64_t, ndim=1] kmer_keys = raw_km[sort_idx]

        cdef np.ndarray[np.uint64_t, ndim=1] cell_indptr = np.empty(n_found + 1, dtype=np.uint64)
        cdef np.ndarray[np.uint32_t, ndim=1] cell_data = np.empty(total_cells, dtype=np.uint32)
        cdef uint64_t* ni_ptr = <uint64_t*>cell_indptr.data
        cdef uint32_t* nc_ptr = <uint32_t*>cell_data.data
        cdef uint64_t* ro_ptr = <uint64_t*>raw_offsets.data
        cdef uint32_t* rc_ptr = <uint32_t*>raw_cells.data
        cdef uint64_t new_off = 0, old_ki, seg_start, seg_len, ki
        cdef int64_t[:] sort_idx_v = sort_idx.astype(np.int64)
        ni_ptr[0] = 0
        for ki in range(n_found):
            old_ki = sort_idx_v[ki]
            seg_start = ro_ptr[old_ki]
            seg_len = ro_ptr[old_ki + 1] - seg_start
            if seg_len > 0:
                memcpy(nc_ptr + new_off, rc_ptr + seg_start, seg_len * 4)
            new_off += seg_len
            ni_ptr[ki + 1] = new_off

        cdef OAHash32 h
        cdef uint32_t hash_expect = <uint32_t>min(total_cells, <uint64_t>8388608)
        oahash_init(&h, hash_expect)
        cdef uint32_t n_unique
        with nogil:
            n_unique = remap_cells_flat(nc_ptr, total_cells, &h)

        cdef np.ndarray[np.uint32_t, ndim=1] orig_cell_ids = np.empty(n_unique, dtype=np.uint32)
        cdef uint32_t* oc_ptr = <uint32_t*>orig_cell_ids.data
        cdef uint32_t tbl_i, tbl_sz = h.mask + 1
        for tbl_i in range(tbl_sz):
            if h.keys[tbl_i] != 0xFFFFFFFF:
                oc_ptr[h.vals[tbl_i]] = h.keys[tbl_i]
        oahash_free(&h)

        logging.info(f"query_batch_csr: {n_found:,} kmers, {total_cells:,} cell refs, "
                     f"{n_unique:,} unique cells, csr_build={time.perf_counter()-t0:.3f}s")

        return (kmer_keys, cell_data, cell_indptr, orig_cell_ids, int(n_unique))

    def iterate_all_kmers(self, bint verbose=False):
        """Iterate through ALL kmers, yielding CSR-compatible arrays."""
        if not self.is_open: raise RuntimeError("Not open")
        cdef:
            uint64_t pf, kmer_val
            uint32_t buf_cap = INITIAL_BUCKET_CAPACITY
            uint32_t* sb = <uint32_t*>malloc(buf_cap * 4)
            uint64_t* db = <uint64_t*>malloc(buf_cap * 8)
            uint32_t* lb = <uint32_t*>malloc(buf_cap * 4)
            uint32_t data_buf_cap = MAX_DATA_PER_BUCKET
            uint32_t* dd = <uint32_t*>malloc(data_buf_cap * 4)
            int ne, nd, raw_bytes
            uint32_t needed, i, ds, dl, total_cells
            uint64_t so, dbo
            uint32_t dbs
            vector[uint64_t] all_indices
            vector[uint64_t] all_indptr
            vector[uint32_t] all_data
            uint64_t data_pos = 0

        if not sb or not db or not lb or not dd:
            free(sb); free(db); free(lb); free(dd)
            raise MemoryError()

        all_indptr.push_back(0)

        try:
            for pf in range(self.n_prefixes):
                so = self.pi_suffix_offsets[pf]
                if so == 0: continue

                ne = self._decompress_suffix_bucket(so, sb, db, lb, buf_cap)
                if ne < 0:
                    needed = <uint32_t>(-ne); free(sb); free(db); free(lb)
                    buf_cap = needed + 1024
                    sb = <uint32_t*>malloc(buf_cap * 4)
                    db = <uint64_t*>malloc(buf_cap * 8)
                    lb = <uint32_t*>malloc(buf_cap * 4)
                    if not sb or not db or not lb:
                        free(sb); free(db); free(lb); free(dd)
                        raise MemoryError()
                    ne = self._decompress_suffix_bucket(so, sb, db, lb, buf_cap)
                if ne <= 0: continue

                dbo = self.pi_data_offsets[pf]
                dbs = self.pi_data_sizes[pf]
                nd = 0
                if dbs > 0:
                    total_cells = 0
                    for i in range(<uint32_t>ne): total_cells += lb[i]
                    if total_cells > data_buf_cap:
                        data_buf_cap = total_cells + 1024
                        free(dd)
                        dd = <uint32_t*>malloc(data_buf_cap * 4)
                        if not dd:
                            free(sb); free(db); free(lb); free(dd)
                            raise MemoryError()
                    if self.data_mmap != NULL and dbo + dbs <= self.data_size:
                        nd = self._decode_data_block(
                            <const uint8_t*>(self.data_mmap + dbo),
                            <int>dbs, lb, ne, dd, data_buf_cap)
                    else:
                        nd = 0

                for i in range(<uint32_t>ne):
                    kmer_val = (<uint64_t>pf << self.rshift) | <uint64_t>sb[i]
                    ds = <uint32_t>db[i]; dl = lb[i]
                    all_indices.push_back(kmer_val)
                    if nd > 0 and ds + dl <= <uint32_t>nd:
                        for j in range(dl):
                            all_data.push_back(dd[ds + j])
                    data_pos += dl
                    all_indptr.push_back(data_pos)

                if verbose and pf % 1000000 == 0 and pf > 0:
                    logging.info(f"  Iterated {pf:,}/{self.n_prefixes:,} prefixes, "
                                 f"{all_indices.size():,} kmers, {all_data.size():,} cells")
                    PyErr_CheckSignals()

        finally:
            free(sb); free(db); free(lb); free(dd)

        cdef np.ndarray[np.uint64_t, ndim=1] out_indices = np.empty(
            all_indices.size(), dtype=np.uint64)
        cdef np.ndarray[np.uint64_t, ndim=1] out_indptr = np.empty(
            all_indptr.size(), dtype=np.uint64)
        cdef np.ndarray[np.uint32_t, ndim=1] out_data = np.empty(
            all_data.size(), dtype=np.uint32)

        if all_indices.size() > 0:
            memcpy(<void*>out_indices.data, &all_indices[0], all_indices.size() * 8)
        if all_indptr.size() > 0:
            memcpy(<void*>out_indptr.data, &all_indptr[0], all_indptr.size() * 8)
        if all_data.size() > 0:
            memcpy(<void*>out_data.data, &all_data[0], all_data.size() * 4)

        if verbose:
            logging.info(f"  Total: {all_indices.size():,} kmers, "
                         f"{all_data.size():,} cell entries")
        return out_indices, out_indptr, out_data


cdef class PrefixIndexBuilder:
    def __cinit__(self, str output_dir, int kmer_size=24, int l_prefix=12, int jump_amount=0, bint verbose=False):
        self.output_dir = output_dir; self.kmer_size = kmer_size; self.l_prefix = l_prefix
        self.l_suffix = kmer_size - l_prefix; self.jump_amount = kmer_size if jump_amount==0 else jump_amount
        self.n_prefixes = 1 << (2*l_prefix); self.rshift = 2*self.l_suffix
        self.suffix_mask = (1 << (2*self.l_suffix)) - 1; self.verbose = verbose; self.buffer_count = 0
        if not os.path.exists(output_dir): os.makedirs(output_dir)
    cdef void _sort_buffer(self): stdsort(self.buffer.begin(), self.buffer.end(), &compare_kmer_cell)
    cdef void _write_buffer_to_chunk(self, str chunk_path):
        cdef FILE* f = fopen(chunk_path.encode('utf-8'), "wb")
        cdef uint64_t n=self.buffer.size(), i, kv
        cdef uint32_t cv
        if f == NULL: raise IOError(f"Cannot open {chunk_path}")
        fwrite(&n, 8, 1, f)
        for i in range(n):
            kv=self.buffer[i].first; cv=self.buffer[i].second
            fwrite(&kv, 8, 1, f); fwrite(&cv, 4, 1, f)
        fclose(f)
    def write_chunk(self, str chunk_path=None):
        if self.buffer.size()==0: return None
        if chunk_path is None: chunk_path = os.path.join(self.output_dir, f'chunk_{self.buffer_count}.bin')
        self._sort_buffer(); self._write_buffer_to_chunk(chunk_path)
        self.buffer.clear(); self.buffer_count = 0; return chunk_path
    def add_pairs(self, np.ndarray[np.uint64_t, ndim=1] kmers, np.ndarray[np.uint32_t, ndim=1] cell_ids):
        cdef uint64_t n=len(kmers), i
        cdef pair[uint64_t, uint32_t] item
        for i in range(n): item.first=kmers[i]; item.second=cell_ids[i]; self.buffer.push_back(item)
        self.buffer_count += n
    cdef inline void add_kmer_cell(self, uint64_t kmer, uint32_t cell_id):
        cdef pair[uint64_t, uint32_t] item
        item.first=kmer; item.second=cell_id; self.buffer.push_back(item); self.buffer_count+=1


cdef void _flush_bucket_c(uint64_t pc, vector[pair[uint32_t, uint32_t]]& pairs,
    FILE* sfh, FILE* dfh, uint64_t* swp, uint64_t* dwp,
    uint64_t* pso, uint64_t* pdo, uint32_t* pds, uint64_t* tk, uint64_t* nb):
    cdef size_t n=pairs.size(), i
    cdef uint32_t cs, ps=0xFFFFFFFF, cc, pc2=0xFFFFFFFF, cnt=0, ne
    cdef vector[uint32_t] su, dl, ac
    cdef uint8_t* vbuf
    cdef int64_t vpos, dpos
    cdef int vn
    cdef uint32_t first_suf, prev_suf
    cdef uint8_t* dbuf
    cdef uint32_t cell_idx, prev_cell, j_cell

    for i in range(n):
        cs=pairs[i].first; cc=pairs[i].second
        if cs!=ps:
            if ps!=0xFFFFFFFF: dl.push_back(cnt); tk[0]+=1
            su.push_back(cs); cnt=0; ps=cs; pc2=0xFFFFFFFF
        if cc!=pc2: ac.push_back(cc); cnt+=1; pc2=cc
    if ps!=0xFFFFFFFF: dl.push_back(cnt); tk[0]+=1
    ne=<uint32_t>su.size()
    if ne==0: return

    vbuf = <uint8_t*>malloc(<size_t>ne * 10 + 8)
    if vbuf == NULL: nb[0] += 1; return
    vpos = 0

    memcpy(&vbuf[vpos], &ne, 4); vpos += 4
    first_suf = su[0]
    memcpy(&vbuf[vpos], &first_suf, 4); vpos += 4
    prev_suf = first_suf
    for i in range(1, ne):
        vn = _encode_varint(su[i] - prev_suf, &vbuf[vpos]); vpos += vn
        prev_suf = su[i]
    for i in range(ne):
        vn = _encode_varint(dl[i], &vbuf[vpos]); vpos += vn

    pso[pc] = swp[0]
    fwrite(vbuf, 1, <size_t>vpos, sfh); swp[0] += <uint64_t>vpos
    free(vbuf)

    if ac.size() > 0:
        dbuf = <uint8_t*>malloc(ac.size() * 5 + <size_t>ne * 4)
        if dbuf == NULL: return
        dpos = 0
        cell_idx = 0

        for i in range(ne):
            prev_cell = ac[cell_idx]
            memcpy(&dbuf[dpos], &prev_cell, 4); dpos += 4
            cell_idx += 1
            for j_cell in range(1, dl[i]):
                vn = _encode_varint(ac[cell_idx] - prev_cell, &dbuf[dpos]); dpos += vn
                prev_cell = ac[cell_idx]; cell_idx += 1

        pdo[pc] = dwp[0]; pds[pc] = <uint32_t>dpos
        fwrite(dbuf, 1, <size_t>dpos, dfh); dwp[0] += <uint64_t>dpos
        free(dbuf)

    nb[0] += 1


from malva.fastq_processing cimport KmerFastqParser, SequenceFastqParser

cdef void _write_sorted_chunk(uint64_t* ak, uint32_t* ac, uint64_t n, str chunk_path):
    """Sort (kmer, cell) pairs by kmer then cell, write to binary chunk file."""
    cdef vector[pair[uint64_t, uint32_t]] pairs
    cdef uint64_t i
    cdef double ts
    ts = time.time()
    logging.info(f"    Copying {n:,} pairs...")
    pairs.reserve(n)
    for i in range(n):
        pairs.push_back(pair[uint64_t, uint32_t](ak[i], ac[i]))
    logging.info(f"    Sorting {n:,} pairs...")
    stdsort(pairs.begin(), pairs.end(), &compare_kmer_cell)
    logging.info(f"    Sorted in {time.time()-ts:.1f}s, writing to disk...")
    cdef FILE* f = fopen(chunk_path.encode('utf-8'), "wb")
    if f == NULL: raise IOError(f"Cannot open {chunk_path}")
    fwrite(&n, 8, 1, f)
    cdef uint64_t kv
    cdef uint32_t cv
    for i in range(n):
        kv = pairs[i].first; cv = pairs[i].second
        fwrite(&kv, 8, 1, f); fwrite(&cv, 4, 1, f)
    fclose(f)
    logging.info(f"    Chunk written in {time.time()-ts:.1f}s total")

def process_fastq_reads(list reads_in, str output_dir, object spatial_index,
    int kmer_size=24, int l_prefix=12, int jump_amount=0, int trim_start=0, int trim_end=28,
    int chunksize=100000000, int n_report=10000000, int threads=1, bint is_bulk=False):
    """Read FASTQ, build index. Uses radix build for single chunk, k-way merge for multiple."""
    from malva.xopen import xopen
    cdef:
        int BUF=4096*1024
        SequenceFastqParser ir1
        KmerFastqParser ir2
        uint64_t r1bc
        vector[uint64_t] r2k
        uint32_t cid
        int ns=0
        size_t i
        uint64_t* ak = NULL
        uint32_t* ac = NULL
        uint64_t tp=0, cap=50000000
        int chunk_num = 0
    if jump_amount==0: jump_amount=kmer_size
    ak=<uint64_t*>malloc(cap*8); ac=<uint32_t*>malloc(cap*4)
    if ak==NULL or ac==NULL: free(ak);free(ac); raise MemoryError()
    cdef double t0=time.time(), dt

    chunk_dir = os.path.join(output_dir, '_chunks')
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_paths = []
    cdef uint64_t total_pairs = 0

    if is_bulk:
        cid=<uint32_t>int(reads_in[0])
        ir2=KmerFastqParser(xopen(reads_in[1],"rb",threads=max(threads//2,1)),BUF,kmer_size=kmer_size,jump_amount=jump_amount)
        while True:
            try: r2k=ir2.next()
            except StopIteration: break
            ns+=1
            for i in range(r2k.size()):
                if tp>=cap:
                    cap=cap*2; ak=<uint64_t*>realloc(ak,cap*8); ac=<uint32_t*>realloc(ac,cap*4)
                    if ak==NULL or ac==NULL: raise MemoryError()
                ak[tp]=r2k[i]; ac[tp]=cid; tp+=1
            if tp >= <uint64_t>chunksize:
                cp = os.path.join(chunk_dir, f'chunk_{chunk_num}.bin')
                logging.info(f"  Writing chunk {chunk_num} ({tp:,} pairs, total {total_pairs+tp:,})")
                _write_sorted_chunk(ak, ac, tp, cp)
                total_pairs += tp; chunk_paths.append(cp); chunk_num += 1; tp = 0
                t0 = time.time()
            if ns%n_report==0:
                dt=time.time()-t0
                if dt > 0:
                    logging.info(f"  Read {ns:,} seqs, {total_pairs+tp:,} total pairs ({n_report/dt:,.0f} reads/s)")
                t0=time.time()
    else:
        ir1=SequenceFastqParser(xopen(reads_in[0],"rb",threads=max(threads//2,1)),BUF,trim_start=trim_start,trim_end=trim_end)
        ir2=KmerFastqParser(xopen(reads_in[1],"rb",threads=max(threads//2,1)),BUF,kmer_size=kmer_size,jump_amount=jump_amount)
        while True:
            try: r2k=ir2.next(); r1bc=ir1.next()
            except StopIteration: break
            cid=spatial_index.lookup(r1bc)
            if cid==0: ns+=1; continue
            ns+=1
            for i in range(r2k.size()):
                if tp>=cap:
                    cap=cap*2; ak=<uint64_t*>realloc(ak,cap*8); ac=<uint32_t*>realloc(ac,cap*4)
                    if ak==NULL or ac==NULL: raise MemoryError()
                ak[tp]=r2k[i]; ac[tp]=cid; tp+=1
            if tp >= <uint64_t>chunksize:
                cp = os.path.join(chunk_dir, f'chunk_{chunk_num}.bin')
                logging.info(f"  Writing chunk {chunk_num} ({tp:,} pairs, total {total_pairs+tp:,})")
                _write_sorted_chunk(ak, ac, tp, cp)
                total_pairs += tp; chunk_paths.append(cp); chunk_num += 1; tp = 0
                t0 = time.time()
            if ns%n_report==0:
                dt=time.time()-t0
                if dt > 0:
                    logging.info(f"  Read {ns:,} seqs, {total_pairs+tp:,} total pairs ({n_report/dt:,.0f} reads/s)")
                t0=time.time()

    logging.info(f"Read {ns:,} sequences total, {total_pairs+tp:,} pairs, {chunk_num} chunks written")

    # Close FASTQ file handles now so that background feeder threads are joined
    # and bgzip subprocesses are fully terminated before the memory-intensive
    # index build begins.  Without this, a feeder thread may still be alive
    # during _build_index_radix, and concurrent use of the C heap from the
    # feeder thread and the radix build has been observed to cause a SIGSEGV
    # during cleanup of the large temporary arrays.
    if not is_bulk and ir1 is not None:
        try:
            ir1.file.close()
        except Exception:
            pass
    if ir2 is not None:
        try:
            ir2.file.close()
        except Exception:
            pass

    if tp == 0 and chunk_num == 0:
        free(ak); free(ac)
        _write_empty_index(output_dir, kmer_size, l_prefix, kmer_size - l_prefix, 0)
        return []

    if chunk_num == 0:
        logging.info(f"Single chunk with {tp:,} pairs - using radix build")
        _build_index_radix(ak, ac, tp, output_dir, kmer_size, l_prefix, 0)
        free(ak); free(ac)
        return []
    else:
        if tp > 0:
            cp = os.path.join(chunk_dir, f'chunk_{chunk_num}.bin')
            logging.info(f"  Writing final chunk {chunk_num} ({tp:,} pairs)")
            _write_sorted_chunk(ak, ac, tp, cp)
            chunk_paths.append(cp)
        free(ak); free(ac)
        logging.info(f"Written {len(chunk_paths)} chunks - will merge")
        return chunk_paths

cdef void _build_index_radix(uint64_t* ak, uint32_t* ac, uint64_t tp, str output_dir, int kmer_size, int l_prefix, uint64_t n_cells):
    cdef int l_suffix=kmer_size-l_prefix
    cdef uint64_t np2=1<<(2*l_prefix), rshift=2*l_suffix, smask=(1<<(2*l_suffix))-1
    cdef uint64_t i, pc, tk=0, nb=0
    cdef double t0=time.time(), t1, t2, t3, now
    logging.info(f"Radix partition {tp:,} entries -> {np2:,} prefixes...")
    cdef uint32_t* cnt=<uint32_t*>malloc(np2*4)
    if cnt==NULL: raise MemoryError()
    memset(cnt,0,np2*4)
    for i in range(tp): cnt[ak[i]>>rshift]+=1
    t1=time.time(); logging.info(f"  Count: {t1-t0:.2f}s")
    cdef uint64_t* ofs=<uint64_t*>malloc(np2*8)
    cdef uint64_t* wp=<uint64_t*>malloc(np2*8)
    if ofs==NULL or wp==NULL: free(cnt);free(ofs);free(wp); raise MemoryError()
    ofs[0]=0; wp[0]=0
    for i in range(1,np2): ofs[i]=ofs[i-1]+cnt[i-1]; wp[i]=ofs[i]
    cdef uint32_t* ps=<uint32_t*>malloc(tp*4)
    cdef uint32_t* pcc=<uint32_t*>malloc(tp*4)
    if ps==NULL or pcc==NULL: free(cnt);free(ofs);free(wp);free(ps);free(pcc); raise MemoryError()
    cdef uint64_t w
    for i in range(tp):
        pc=ak[i]>>rshift; w=wp[pc]; ps[w]=<uint32_t>(ak[i]&smask); pcc[w]=ac[i]; wp[pc]=w+1
    t2=time.time(); logging.info(f"  Scatter: {t2-t1:.2f}s")
    free(wp)
    pp=os.path.join(output_dir,'pi.bin'); sp=os.path.join(output_dir,'suffixes.bin')
    dp=os.path.join(output_dir,'data.bin'); mp=os.path.join(output_dir,'meta.json')
    cdef np.ndarray[np.uint64_t,ndim=1] pis=np.zeros(np2,dtype=np.uint64)
    cdef np.ndarray[np.uint64_t,ndim=1] pid=np.zeros(np2,dtype=np.uint64)
    cdef np.ndarray[np.uint32_t,ndim=1] pidsz=np.zeros(np2,dtype=np.uint32)
    cdef FILE* sfh=fopen(sp.encode('utf-8'),"wb")
    cdef FILE* dfh=fopen(dp.encode('utf-8'),"wb")
    if sfh==NULL or dfh==NULL: raise IOError("Cannot open output")
    cdef uint32_t z=0
    fwrite(&z,4,1,sfh)
    cdef uint64_t swp=4, dwp=0
    cdef vector[pair[uint32_t,uint32_t]] bp
    cdef uint64_t bs
    cdef uint32_t j
    cdef double last_rpt = t2
    for i in range(np2):
        if cnt[i]==0: continue
        bs=ofs[i]; bp.clear(); bp.reserve(cnt[i])
        for j in range(cnt[i]): bp.push_back(pair[uint32_t,uint32_t](ps[bs+j],pcc[bs+j]))
        stdsort(bp.begin(),bp.end())
        _flush_bucket_c(i,bp,sfh,dfh,&swp,&dwp,<uint64_t*>pis.data,<uint64_t*>pid.data,<uint32_t*>pidsz.data,&tk,&nb)
        if i%500000==0 and i>0:
            now = time.time()
            if now - last_rpt > 5.0:
                logging.info(f"  Write: {i:,}/{np2:,} prefixes ({100.0*i/np2:.1f}%), {tk:,} kmers, {nb:,} buckets")
                last_rpt = now
            PyErr_CheckSignals()
    t3=time.time(); logging.info(f"  Sort+write: {t3-t2:.2f}s")
    fclose(sfh); fclose(dfh); free(cnt); free(ofs); free(ps); free(pcc)
    with open(pp,'wb') as f: f.write(pis.tobytes()); f.write(pid.tobytes()); f.write(pidsz.tobytes())
    meta={'magic':int(MAGIC),'version':int(VERSION),'kmer_size':kmer_size,'l_prefix':l_prefix,'l_suffix':l_suffix,'n_kmers':int(tk),'n_cells':int(n_cells),'n_prefixes':int(np2),'n_buckets_written':int(nb)}
    with open(mp,'w') as f: json.dump(meta,f,indent=2)
    logging.info(f"Build: {tk:,} kmers, {nb:,} buckets ({time.time()-t0:.1f}s)")
    logging.info(f"  PI:{os.path.getsize(pp)/1e6:.1f}MB Suf:{os.path.getsize(sp)/1e6:.1f}MB Data:{os.path.getsize(dp)/1e6:.1f}MB")


cdef struct ChunkReader:
    FILE* fh
    uint64_t remaining, current_kmer
    uint32_t current_cell
    bint valid
cdef bint chunk_reader_next(ChunkReader* cr) nogil:
    cdef uint64_t kv
    cdef uint32_t cv
    if cr.remaining==0: cr.valid=False; return False
    if fread(&kv,8,1,cr.fh)!=1: cr.valid=False; return False
    if fread(&cv,4,1,cr.fh)!=1: cr.valid=False; return False
    cr.current_kmer=kv; cr.current_cell=cv; cr.remaining-=1; cr.valid=True; return True

def build_from_sorted_chunks(list chunk_paths, str output_dir, int kmer_size=24, int l_prefix=12, uint64_t n_cells=0, bint verbose=False):
    cdef int ls=kmer_size-l_prefix, nc=len(chunk_paths), i, mi, ac2=0
    cdef uint64_t np2=1<<(2*l_prefix), rshift=2*ls, smask=(1<<(2*ls))-1
    cdef uint64_t tk=0,nb=0,mk,nc2,kv,pc,ep=0
    cdef uint32_t mc,sc,cv,cp2=0xFFFFFFFF
    cdef uint64_t cpp=0xFFFFFFFFFFFFFFFF
    cdef uint64_t total_entries = 0
    logger=logging.getLogger("PrefixIndex.build")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    pp=os.path.join(output_dir,'pi.bin'); sp=os.path.join(output_dir,'suffixes.bin')
    dp=os.path.join(output_dir,'data.bin'); mp=os.path.join(output_dir,'meta.json')
    logger.info(f"Building from {nc} chunks, kmer_size={kmer_size}, l_prefix={l_prefix}")
    cdef ChunkReader* rd=<ChunkReader*>malloc(nc*sizeof(ChunkReader))
    if rd==NULL: raise MemoryError()
    for i in range(nc):
        pb=chunk_paths[i].encode('utf-8'); rd[i].fh=fopen(pb,"rb")
        if rd[i].fh==NULL: raise IOError(f"Cannot open {chunk_paths[i]}")
        fread(&nc2,8,1,rd[i].fh); rd[i].remaining=nc2; rd[i].valid=False
        total_entries += nc2
        if chunk_reader_next(&rd[i]): ac2+=1
        logger.info(f"  Chunk {i}: {nc2:,} entries")
    logger.info(f"Opened {ac2} chunks, {total_entries:,} total entries to merge")
    cdef np.ndarray[np.uint64_t,ndim=1] pis=np.zeros(np2,dtype=np.uint64)
    cdef np.ndarray[np.uint64_t,ndim=1] pid=np.zeros(np2,dtype=np.uint64)
    cdef np.ndarray[np.uint32_t,ndim=1] pidsz=np.zeros(np2,dtype=np.uint32)
    cdef FILE* sfh=fopen(sp.encode('utf-8'),"wb")
    cdef FILE* dfh=fopen(dp.encode('utf-8'),"wb")
    cdef uint32_t z=0
    fwrite(&z,4,1,sfh)
    cdef uint64_t swp=4,dwp=0
    cdef vector[pair[uint32_t,uint32_t]] bp
    cdef double t0=time.time(), last_report=t0, now
    while ac2>0:
        mk=0xFFFFFFFFFFFFFFFF; mi=-1
        for i in range(nc):
            if rd[i].valid:
                if rd[i].current_kmer<mk or (rd[i].current_kmer==mk and rd[i].current_cell<mc):
                    mk=rd[i].current_kmer; mc=rd[i].current_cell; mi=i
        if mi<0: break
        kv=rd[mi].current_kmer; cv=rd[mi].current_cell; pc=kv>>rshift; sc=<uint32_t>(kv&smask)
        if pc!=cpp:
            if cpp!=0xFFFFFFFFFFFFFFFF and bp.size()>0:
                _flush_bucket_c(cpp,bp,sfh,dfh,&swp,&dwp,<uint64_t*>pis.data,<uint64_t*>pid.data,<uint32_t*>pidsz.data,&tk,&nb)
            cpp=pc; bp.clear()
        bp.push_back(pair[uint32_t,uint32_t](sc,cv))
        if not chunk_reader_next(&rd[mi]): ac2-=1
        ep+=1
        if ep % 1000000 == 0:
            now = time.time()
            if now - last_report > 5.0:
                pct = 100.0 * ep / total_entries if total_entries > 0 else 0
                rate = ep / (now - t0) / 1e6
                logger.info(f"  Merged {ep:,}/{total_entries:,} ({pct:.1f}%) "
                           f"{tk:,} kmers, {nb:,} buckets, {rate:.1f}M/s")
                last_report = now
                PyErr_CheckSignals()
    if cpp!=0xFFFFFFFFFFFFFFFF and bp.size()>0:
        _flush_bucket_c(cpp,bp,sfh,dfh,&swp,&dwp,<uint64_t*>pis.data,<uint64_t*>pid.data,<uint32_t*>pidsz.data,&tk,&nb)
    fclose(sfh); fclose(dfh)
    for i in range(nc):
        if rd[i].fh!=NULL: fclose(rd[i].fh)
    free(rd)
    with open(pp,'wb') as f: f.write(pis.tobytes()); f.write(pid.tobytes()); f.write(pidsz.tobytes())
    meta={'magic':int(MAGIC),'version':int(VERSION),'kmer_size':kmer_size,'l_prefix':l_prefix,'l_suffix':ls,'n_kmers':int(tk),'n_cells':int(n_cells),'n_prefixes':int(np2),'n_buckets_written':int(nb)}
    with open(mp,'w') as f: json.dump(meta,f,indent=2)
    logger.info(f"Merge complete: {tk:,} kmers, {nb:,} buckets ({time.time()-t0:.1f}s)")
    logger.info(f"  PI:{os.path.getsize(pp)/1e6:.1f}MB Suf:{os.path.getsize(sp)/1e6:.1f}MB Data:{os.path.getsize(dp)/1e6:.1f}MB")


def _write_empty_index(str od, int ks, int lp, int ls, uint64_t nc):
    np2=1<<(2*lp); pi=np.zeros(np2,dtype=np.uint64)
    with open(os.path.join(od,'pi.bin'),'wb') as f: f.write(pi.tobytes()); f.write(pi.tobytes()); f.write(np.zeros(np2,dtype=np.uint32).tobytes())
    with open(os.path.join(od,'suffixes.bin'),'wb') as f: f.write(b'\x00\x00\x00\x00')
    with open(os.path.join(od,'data.bin'),'wb') as f: pass
    with open(os.path.join(od,'meta.json'),'w') as f: json.dump({'magic':0x4D4C5650,'version':2,'kmer_size':ks,'l_prefix':lp,'l_suffix':ls,'n_kmers':0,'n_cells':int(nc),'n_prefixes':int(np2),'n_buckets_written':0},f,indent=2)


def write_sorted_chunk_py(np.ndarray[np.uint64_t, ndim=1] ak_arr,
                          np.ndarray[np.uint32_t, ndim=1] ac_arr,
                          Py_ssize_t tp, str chunk_path):
    """Python-accessible wrapper: sort and write a (kmer, cell_id) chunk file."""
    cdef uint64_t n = <uint64_t>tp
    cdef uint64_t* ak = <uint64_t*>ak_arr.data
    cdef uint32_t* ac = <uint32_t*>ac_arr.data
    _write_sorted_chunk(ak, ac, n, chunk_path)


def build_index_radix_py(np.ndarray[np.uint64_t, ndim=1] ak_arr,
                         np.ndarray[np.uint32_t, ndim=1] ac_arr,
                         Py_ssize_t tp, str output_dir, int kmer_size, int l_prefix):
    """Python-accessible wrapper: build the prefix-bucketed index from arrays via radix sort."""
    cdef uint64_t n = <uint64_t>tp
    cdef uint64_t* ak = <uint64_t*>ak_arr.data
    cdef uint32_t* ac = <uint32_t*>ac_arr.data
    _build_index_radix(ak, ac, n, output_dir, kmer_size, l_prefix, 0)


def merge_prefix_indices(list index_dirs, str output_dir, bint merge_projects=False, dict project_mapping=None,
    int project_id_shift=23, uint32_t cell_id_mask=0x007FFFFF, bint verbose=False):
    logger=logging.getLogger("PrefixIndex.merge")
    cdef uint32_t merge_buf_cap = INITIAL_BUCKET_CAPACITY
    cdef uint32_t* sb=<uint32_t*>malloc(merge_buf_cap*4)
    cdef uint64_t* dob=<uint64_t*>malloc(merge_buf_cap*8)
    cdef uint32_t* dlb=<uint32_t*>malloc(merge_buf_cap*4)
    cdef uint32_t merge_data_cap = MAX_DATA_PER_BUCKET
    cdef uint32_t* ddb_dec=<uint32_t*>malloc(merge_data_cap*4)
    cdef int raw_bytes2
    cdef uint32_t total_cells2
    cdef PrefixIndex _fi
    if not sb or not dob or not dlb or not ddb_dec: free(sb);free(dob);free(dlb);free(ddb_dec); raise MemoryError()
    indices=[]
    for d in index_dirs: p=PrefixIndex(); p.open(d); indices.append(p)
    if len(indices)==0: free(sb);free(dob);free(dlb);free(ddb_dec); raise ValueError("No indices")
    _fi=<PrefixIndex>indices[0]; ks=_fi.kmer_size; lp=_fi.l_prefix; ls=_fi.l_suffix; np2=_fi.n_prefixes
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    pp=os.path.join(output_dir,'pi.bin'); sp=os.path.join(output_dir,'suffixes.bin')
    dp2=os.path.join(output_dir,'data.bin'); mp=os.path.join(output_dir,'meta.json')
    cdef np.ndarray[np.uint64_t,ndim=1] oso=np.zeros(np2,dtype=np.uint64)
    cdef np.ndarray[np.uint64_t,ndim=1] odo=np.zeros(np2,dtype=np.uint64)
    cdef np.ndarray[np.uint32_t,ndim=1] ods=np.zeros(np2,dtype=np.uint32)
    cdef FILE* sfh=fopen(sp.encode('utf-8'),"wb")
    cdef FILE* dfh=fopen(dp2.encode('utf-8'),"wb")
    cdef uint32_t zv=0
    fwrite(&zv,4,1,sfh)
    cdef uint64_t swp2=4,dwp2=0,tk2=0,nb2=0
    cdef double t02=time.time()
    cdef int ne2,j2,k2,nd2
    cdef uint32_t sc2,ds2,dlv2,cvm2,needed2
    cdef uint64_t sov2,dbo2
    cdef uint32_t dbsv2
    cdef vector[pair[uint32_t,uint32_t]] mp2
    try:
        for pc2 in range(np2):
            if verbose and pc2%500000==0 and pc2>0:
                dt2=time.time()-t02; logger.info(f"  {pc2:,}/{np2:,} ({100.0*pc2/np2:.1f}%) {tk2:,} kmers {dt2:.1f}s"); PyErr_CheckSignals()
            mp2.clear()
            for ii, po in enumerate(indices):
                pi=<PrefixIndex>po; sov2=pi.pi_suffix_offsets[pc2]
                if sov2==0: continue
                ne2=pi._decompress_suffix_bucket(sov2,sb,dob,dlb,merge_buf_cap)
                if ne2 < 0:
                    needed2 = <uint32_t>(-ne2); free(sb); free(dob); free(dlb)
                    merge_buf_cap = needed2 + 1024
                    sb=<uint32_t*>malloc(merge_buf_cap*4); dob=<uint64_t*>malloc(merge_buf_cap*8); dlb=<uint32_t*>malloc(merge_buf_cap*4)
                    if not sb or not dob or not dlb: free(sb);free(dob);free(dlb);free(ddb_dec); raise MemoryError()
                    ne2=pi._decompress_suffix_bucket(sov2,sb,dob,dlb,merge_buf_cap)
                if ne2<=0: continue
                dbo2=pi.pi_data_offsets[pc2]; dbsv2=pi.pi_data_sizes[pc2]; nd2=0
                if dbsv2>0:
                    total_cells2 = 0
                    for j2 in range(ne2):
                        total_cells2 += dlb[j2]
                    if total_cells2 > merge_data_cap:
                        merge_data_cap = total_cells2 + 1024
                        free(ddb_dec)
                        ddb_dec = <uint32_t*>malloc(merge_data_cap*4)
                        if not ddb_dec:
                            free(sb);free(dob);free(dlb);free(ddb_dec); raise MemoryError()
                    if pi.data_mmap != NULL and dbo2 + dbsv2 <= pi.data_size:
                        nd2 = pi._decode_data_block(<const uint8_t*>(pi.data_mmap + dbo2), <int>dbsv2, dlb, ne2, ddb_dec, merge_data_cap)
                    else:
                        nd2 = 0
                for j2 in range(ne2):
                    sc2=sb[j2]; ds2=<uint32_t>dob[j2]; dlv2=dlb[j2]
                    for k2 in range(dlv2):
                        if (ds2+k2) < <uint32_t>nd2:
                            cvm2=ddb_dec[ds2+k2]
                            if merge_projects and project_mapping is not None:
                                pid2=project_mapping.get(ii,(0,''))[0]
                                cvm2=(cvm2&cell_id_mask)|(<uint32_t>pid2<<project_id_shift)
                            mp2.push_back(pair[uint32_t,uint32_t](sc2,cvm2))
            if mp2.size()==0: continue
            stdsort(mp2.begin(),mp2.end())
            _flush_bucket_c(pc2,mp2,sfh,dfh,&swp2,&dwp2,<uint64_t*>oso.data,<uint64_t*>odo.data,<uint32_t*>ods.data,&tk2,&nb2)
    finally: free(sb);free(dob);free(dlb);free(ddb_dec)
    fclose(sfh); fclose(dfh)
    with open(pp,'wb') as f: f.write(oso.tobytes()); f.write(odo.tobytes()); f.write(ods.tobytes())
    # merge_projects=True: different samples, each with their own whitelist → sum cells.
    # merge_projects=False: same sample, multiple lanes sharing one whitelist → max cells.
    if merge_projects:
        tnc=sum((<PrefixIndex>p).n_cells for p in indices)
    else:
        tnc=max((<PrefixIndex>p).n_cells for p in indices)
    meta={'magic':int(MAGIC),'version':int(VERSION),'kmer_size':ks,'l_prefix':lp,'l_suffix':ls,'n_kmers':int(tk2),'n_cells':int(tnc),'n_prefixes':int(np2),'merge_projects':merge_projects,'project_id_shift':project_id_shift,'cell_id_mask':int(cell_id_mask)}
    if project_mapping: meta['project_mapping']={str(kk):v for kk,v in project_mapping.items()}
    with open(mp,'w') as f: json.dump(meta,f,indent=2)
    for p in indices: p.close()
    logger.info(f"Merge: {tk2:,} kmers from {len(indices)} indices in {time.time()-t02:.1f}s")


from malva.fastq_processing cimport FastKmerProcessor
from malva.kmer_processing import get_kmers_numeric

cdef extern from *:
    """
    #include <stdint.h>
    #include <string.h>

    static int QW_BASE_ENC_storage[256];
    static const int* QW_BASE_ENC = QW_BASE_ENC_storage;
    static struct _QW_Init {
        _QW_Init() {
            for (int i = 0; i < 256; i++) QW_BASE_ENC_storage[i] = 4;
            QW_BASE_ENC_storage[(unsigned char)'A'] = 0;
            QW_BASE_ENC_storage[(unsigned char)'C'] = 1;
            QW_BASE_ENC_storage[(unsigned char)'T'] = 2;
            QW_BASE_ENC_storage[(unsigned char)'U'] = 2;
            QW_BASE_ENC_storage[(unsigned char)'G'] = 3;
            QW_BASE_ENC_storage[(unsigned char)'N'] = 4;  /* treat N as invalid, not G */
        }
    } _qw_init;

    /* Encode non-overlapping kmers from seq[0..seq_len) into out[].
     * Writes 0 for kmers containing 'N'.  Returns number of kmers written. */
    static inline int encode_nonoverlapping_kmers(
        const char* seq, int seq_len, int k,
        uint64_t* out, int out_cap)
    {
        int n = 0;
        for (int i = 0; i + k <= seq_len && n < out_cap; i += k) {
            uint64_t val = 0;
            int has_n = 0;
            for (int j = 0; j < k; j++) {
                int b = QW_BASE_ENC[(unsigned char)seq[i + j]];
                if (b == 4) { has_n = 1; }  /* invalid base */
                val = (val << 2) | (b & 3);
            }
            if (has_n) {
                /* Scan for actual N */
                has_n = 0;
                for (int j = 0; j < k; j++) {
                    if (seq[i+j] == 'N') { has_n = 1; break; }
                }
            }
            out[n++] = has_n ? 0 : val;
        }
        /* Handle tail kmer if sequence doesn't divide evenly */
        int tail_start = (seq_len / k) * k;
        int tail_len = seq_len - tail_start;
        if (tail_len > 0 && tail_len < k && n < out_cap && tail_start >= k) {
            /* Take overlapping nucleotides from previous kmer */
            int start = seq_len - k;
            uint64_t val = 0;
            int has_n2 = 0;
            for (int j = 0; j < k; j++) {
                unsigned char c = seq[start + j];
                if (c == 'N') has_n2 = 1;
                int b = QW_BASE_ENC[c];
                val = (val << 2) | (b & 3);
            }
            out[n++] = has_n2 ? 0 : val;
        }
        return n;
    }
    """
    int encode_nonoverlapping_kmers(
        const char* seq, int seq_len, int k,
        uint64_t* out, int out_cap) nogil

def quantify_where(
    PrefixIndex index, list sequence_groups, int kmer_size,
    int sliding_size=128, float pct_threshold=0.65,
    uint32_t count_at_most=10000, uint32_t count_at_least=10,
    uint32_t max_cell_id=0, bint single_count=False,
    object background_model=None, bint use_background_model=True,
    bint verbose=False,
):
    """Quantify sequence groups against the index, returning per-group cell hits."""
    cdef:
        FastKmerProcessor processor
        np.ndarray all_kmer_list
        list results = []
        float CONST_THRESHOLD
        int BACKGROUND_THRESHOLD = 1
        list seq_matches = [[0, 1]]
        uint32_t kmers_per_read, _sliding_size
        uint64_t kmer_val
        uint32_t i, j, idx_kmer, value, idx
        Py_ssize_t num_groups, gkl_len
        np.ndarray[np.uint32_t, ndim=1] kmer_locations, kmer_count
        object subseq, group, seq, split_sliding_sequences
        double t_start, t_phase
        uint32_t* cc
        uint32_t* cl_ki
        uint32_t* sc
        uint32_t* cu
        uint32_t  cu_n
        uint32_t* sd
        uint32_t  sd_n
        uint32_t  n_unique_cells, compact_id, orig_id
        uint32_t  c_val, kpr
        float     thresh_f
        np.ndarray[np.uint64_t, ndim=1] csr_keys, csr_indptr
        np.ndarray[np.uint32_t, ndim=1] csr_cells, csr_orig
        uint64_t* csr_keys_ptr
        uint32_t* csr_cells_ptr
        uint64_t* csr_indptr_ptr
        uint32_t* csr_orig_ptr
        int64_t   csr_n_keys, csr_idx
        uint64_t  cell_start, cell_end
        uint64_t* kmer_buf
        int       kmer_buf_cap, n_encoded
        const char* subseq_c
        int       subseq_len, max_kmers

    t_start = time.time()
    processor = FastKmerProcessor(kmer_size, True, kmer_size)
    all_kmer_list = processor.process_sequences(sequence_groups)
    num_groups = len(sequence_groups)

    if verbose:
        t_phase = time.time()
        logging.info(f"  Parsed {len(all_kmer_list):,} unique kmers from {num_groups} groups in {t_phase-t_start:.2f}s")

    if len(all_kmer_list) == 0:
        return [(np.array([], dtype=np.uint32), np.array([], dtype=np.uint32), [])] * num_groups

    t_phase = time.time()

    csr_result = (<PrefixIndex>index).query_batch_csr(
        np.asarray(all_kmer_list, dtype=np.uint64),
        count_at_most=count_at_most, count_at_least=count_at_least)
    csr_keys   = csr_result[0]
    csr_cells  = csr_result[1]
    csr_indptr = csr_result[2]
    csr_orig   = csr_result[3]
    n_unique_cells = csr_result[4]

    t = (<PrefixIndex>index)._last_query_times
    if t:
        logging.info(
            f"query_batch timing | kmers_in={t.get('n_kmers_in',0):,} "
            f"buckets={t.get('n_buckets_touched',0):,} hits={t.get('n_hits',0):,} "
            f"suf={t.get('n_suf_bytes',0)/1e6:.1f}MB dat={t.get('n_dat_bytes',0)/1e6:.1f}MB | "
            f"suf_read={t.get('t_suf_read',0)*1000:.1f}ms "
            f"suf_decomp={t.get('t_suffix_decomp',0)*1000:.1f}ms "
            f"bsearch={t.get('t_bsearch',0)*1000:.1f}ms "
            f"dat_read={t.get('t_dat_read',0)*1000:.1f}ms "
            f"phase2={t.get('t_phase2',0)*1000:.1f}ms "
            f"unaccounted={t.get('t_unaccounted',0)*1000:.1f}ms "
            f"total={t.get('t_total',0)*1000:.1f}ms"
        )

    csr_n_keys = len(csr_keys)

    if verbose:
        logging.info(f"  Queried index: {csr_n_keys:,}/{len(all_kmer_list):,} kmers found "
                     f"(count_at_least={count_at_least}, count_at_most={count_at_most}) in {time.time()-t_phase:.2f}s")
        logging.info(f"  Remapped {n_unique_cells:,} unique cells (max_cell_id={max_cell_id:,})")

    if csr_n_keys == 0:
        return [(np.array([], dtype=np.uint32), np.array([], dtype=np.uint32), [])] * num_groups

    t_phase = time.time()

    csr_keys_ptr   = <uint64_t*>csr_keys.data
    csr_cells_ptr  = <uint32_t*>csr_cells.data
    csr_indptr_ptr = <uint64_t*>csr_indptr.data
    csr_orig_ptr   = <uint32_t*>csr_orig.data

    cc    = <uint32_t*>malloc((n_unique_cells + 1) * 4)
    cl_ki = <uint32_t*>malloc((n_unique_cells + 1) * 4)
    sc    = <uint32_t*>malloc((n_unique_cells + 1) * 4)
    cu    = <uint32_t*>malloc((n_unique_cells + 1) * 4)
    sd    = <uint32_t*>malloc((n_unique_cells + 1) * 4)
    kmer_buf_cap = 1024
    kmer_buf = <uint64_t*>malloc(kmer_buf_cap * 8)
    if not cc or not cl_ki or not sc or not cu or not sd or not kmer_buf:
        free(cc); free(cl_ki); free(sc); free(cu); free(sd); free(kmer_buf)
        raise MemoryError()
    memset(cc,    0, (n_unique_cells + 1) * 4)
    memset(cl_ki, 0, (n_unique_cells + 1) * 4)
    memset(sc,    0, (n_unique_cells + 1) * 4)

    try:
        for i in range(num_groups):
            group = sequence_groups[i]

            split_sliding_sequences = set()
            for seq in group:
                _sliding_size = sliding_size if sliding_size > 0 else len(seq) - kmer_size
                for si in range(0, len(seq) - _sliding_size + 1):
                    split_sliding_sequences.add(seq[si:si + _sliding_size])

            sd_n = 0

            for subseq in split_sliding_sequences:
                _sliding_size = sliding_size if sliding_size > 0 else len(subseq)
                CONST_THRESHOLD = (sliding_size // kmer_size) * pct_threshold
                if sliding_size <= 0:
                    CONST_THRESHOLD = (abs(sliding_size) // kmer_size) * pct_threshold
                kmers_per_read = <uint32_t>(_sliding_size // kmer_size)
                kpr = kmers_per_read
                thresh_f = CONST_THRESHOLD

                subseq_bytes = subseq.encode('ascii') if isinstance(subseq, str) else subseq
                subseq_c = subseq_bytes
                subseq_len = len(subseq_bytes)
                if subseq_len < kmer_size:
                    continue
                max_kmers = subseq_len // kmer_size + 2
                if max_kmers > kmer_buf_cap:
                    kmer_buf_cap = max_kmers + 64
                    free(kmer_buf)
                    kmer_buf = <uint64_t*>malloc(kmer_buf_cap * 8)
                    if not kmer_buf:
                        free(cc); free(cl_ki); free(sc); free(cu); free(sd)
                        raise MemoryError()

                n_encoded = encode_nonoverlapping_kmers(
                    subseq_c, subseq_len, kmer_size, kmer_buf, kmer_buf_cap)
                gkl_len = n_encoded
                if gkl_len == 0: continue

                if use_background_model and background_model is not None:
                    for idx_kmer in range(gkl_len):
                        if kmer_buf[idx_kmer] != 0 and \
                                background_model.is_mer_above_cutoff(kmer_buf[idx_kmer], BACKGROUND_THRESHOLD):
                            kmer_buf[idx_kmer] = 0

                cu_n = 0

                with nogil:
                    for idx_kmer in range(gkl_len):
                        kmer_val = kmer_buf[idx_kmer]
                        if kmer_val == 0: continue

                        csr_idx = bsearch_u64(csr_keys_ptr, csr_n_keys, kmer_val)
                        if csr_idx < 0: continue

                        cell_start = csr_indptr_ptr[csr_idx]
                        cell_end   = csr_indptr_ptr[csr_idx + 1]
                        for j in range(<uint32_t>(cell_end - cell_start)):
                            c_val = csr_cells_ptr[cell_start + j]
                            if cc[c_val] == 0:
                                cu[cu_n] = c_val
                                cu_n += 1
                            cc[c_val] += 1
                            cl_ki[c_val] = idx_kmer
                            if cc[c_val] > kpr:
                                cc[c_val] = kpr

                        if ((<uint32_t>(idx_kmer + 1)) < kpr) and (idx_kmer + 1 < gkl_len):
                            continue

                        for j in range(cu_n):
                            c_val = cu[j]
                            if cc[c_val] > <uint32_t>thresh_f:
                                if sc[c_val] == 0:
                                    sd[sd_n] = c_val
                                    sd_n += 1
                                    sc[c_val] = 1
                                elif not single_count:
                                    cc[c_val] = 0
                                    sc[c_val] += 1
                            if cl_ki[c_val] - idx_kmer > 0 and cc[c_val] > 0:
                                cc[c_val] -= 1

                with nogil:
                    for j in range(cu_n):
                        c_val = cu[j]
                        cc[c_val] = 0
                        cl_ki[c_val] = 0

            kmer_locations = np.empty(sd_n, dtype=np.uint32)
            kmer_count     = np.empty(sd_n, dtype=np.uint32)
            for idx in range(sd_n):
                c_val = sd[idx]
                kmer_locations[idx] = csr_orig_ptr[c_val]
                kmer_count[idx]     = sc[c_val]
            results.append((kmer_locations, kmer_count, seq_matches))

            with nogil:
                for j in range(sd_n):
                    sc[sd[j]] = 0
            sd_n = 0

    finally:
        free(cc); free(cl_ki); free(sc); free(cu); free(sd); free(kmer_buf)

    if verbose:
        logging.info(f"  Cell counting for {num_groups} groups completed in {time.time()-t_phase:.2f}s "
                     f"(total {time.time()-t_start:.2f}s)")

    return results


def convert_h5_to_prefix(object indices_ds, object indptr_ds, object data_ds,
    str output_dir, int kmer_size=24, int l_prefix=12, uint64_t n_cells=0,
    int chunk_size=5000000, bint verbose=True):
    """
    Convert HDF5 index arrays to prefix-bucketed compressed format.
    All heavy work in C - zero Python calls in the inner loop.
    """
    cdef:
        int l_suffix = kmer_size - l_prefix
        uint64_t n_prefixes = 1 << (2 * l_prefix)
        uint64_t rshift_val = 2 * l_suffix
        uint64_t suffix_mask_val = (1 << (2 * l_suffix)) - 1
        uint64_t total_indices = len(indices_ds)
        uint64_t total_indptr = len(indptr_ds)
        uint64_t total_data = len(data_ds)
        uint64_t tk = 0, nb = 0
        uint64_t pos, end_pos, i, j
        uint64_t kmer_val, prefix_code, cur_prefix = 0xFFFFFFFFFFFFFFFF
        uint32_t suffix_code, cc, prev_cc
        int64_t n_in
        vector[pair[uint32_t, uint32_t]] bucket_pairs
        uint64_t* idx_ptr
        uint64_t* ip_start_ptr
        uint64_t* ip_end_ptr
        uint32_t* data_ptr
        int64_t ds_min, ds_max, ds, de
        uint64_t data_block_len
        # Typed references to keep numpy arrays alive while raw pointers are in use.
        # Without these, Cython may release the Python objects early, allowing GC
        # to free the backing buffer -> dangling pointer -> SIGSEGV under memory pressure.
        np.ndarray indices_arr, indptr_arr, ends_arr, data_block
        int64_t reduced_n
        uint64_t max_pairs_per_chunk
        uint64_t max_bucket_pairs
        int64_t effective_chunk_size
        int64_t win_start, win_end, win_ds, win_de

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pp = os.path.join(output_dir, 'pi.bin')
    sp = os.path.join(output_dir, 'suffixes.bin')
    dp = os.path.join(output_dir, 'data.bin')
    mp = os.path.join(output_dir, 'meta.json')

    cdef np.ndarray[np.uint64_t, ndim=1] pis = np.zeros(n_prefixes, dtype=np.uint64)
    cdef np.ndarray[np.uint64_t, ndim=1] pid = np.zeros(n_prefixes, dtype=np.uint64)
    cdef np.ndarray[np.uint32_t, ndim=1] pidsz = np.zeros(n_prefixes, dtype=np.uint32)

    cdef FILE* sfh = fopen(sp.encode('utf-8'), "wb")
    cdef FILE* dfh = fopen(dp.encode('utf-8'), "wb")
    if sfh == NULL or dfh == NULL:
        raise IOError("Cannot open output files")

    cdef uint32_t z = 0
    fwrite(&z, 4, 1, sfh)
    cdef uint64_t swp = 4, dwp = 0
    cdef double t0 = time.time(), last_report = t0, now, dt

    # Adaptive chunk size: if cells-per-kmer is high, reduce chunk_size to
    # cap peak memory.  bucket_pairs holds (suffix, cell) pairs - 8 bytes each.
    # data_block holds cell IDs - 4 bytes each.  Peak memory per chunk iteration
    # is roughly: data_block + bucket_pairs ~= 2 * data_block_size.
    # Cap at ~2GB combined -> max ~250M cell references per chunk.
    max_pairs_per_chunk = 250000000  # 250M pairs = ~2GB

    # Hard cap on bucket_pairs size.  When a single prefix accumulates more
    # than this many (suffix, cell) pairs, further cells are dropped.  This
    # prevents unbounded vector growth for pathological kmers (e.g. poly-A
    # sequences matching hundreds of millions of cells).  Such kmers always
    # exceed count_at_most during querying and are filtered out, so the
    # truncation has no practical effect on results.
    max_bucket_pairs = max_pairs_per_chunk

    effective_chunk_size = chunk_size

    pos = 0
    while pos < total_indices:
        end_pos = min(pos + effective_chunk_size, total_indices)

        indices_arr = np.ascontiguousarray(indices_ds[pos:end_pos], dtype=np.uint64)
        ip_end_val = min(end_pos + 1, total_indptr)
        indptr_arr = np.ascontiguousarray(indptr_ds[pos:ip_end_val], dtype=np.uint64)

        n_in = len(indices_arr)
        idx_ptr = <uint64_t*>(<np.ndarray>indices_arr).data
        ip_start_ptr = <uint64_t*>(<np.ndarray>indptr_arr).data

        ends_arr = np.empty(n_in, dtype=np.uint64)
        if len(indptr_arr) > n_in:
            ends_arr[:] = indptr_arr[1:n_in + 1]
        else:
            ends_arr[:n_in - 1] = indptr_arr[1:n_in]
            ends_arr[n_in - 1] = total_data
        ip_end_ptr = <uint64_t*>(<np.ndarray>ends_arr).data

        ds_min = <int64_t>ip_start_ptr[0] if n_in > 0 else 0
        ds_max = <int64_t>ip_end_ptr[n_in - 1] if n_in > 0 else 0

        data_block = None
        if ds_max > ds_min:
            # Check if this chunk's data would exceed the memory cap.
            # If so, reduce end_pos to process fewer kmers this iteration.
            if <uint64_t>(ds_max - ds_min) > max_pairs_per_chunk:
                # Binary search for a smaller end_pos where data fits.
                # Use indptr to estimate: find the kmer whose indptr stays within cap.
                reduced_n = n_in
                while reduced_n > 1:
                    reduced_n = reduced_n * max_pairs_per_chunk // <uint64_t>(ds_max - ds_min)
                    if reduced_n < 1:
                        reduced_n = 1
                    # Recompute ds_max for the reduced range
                    ds_max = <int64_t>ip_end_ptr[reduced_n - 1] if reduced_n > 0 else ds_min
                    if <uint64_t>(ds_max - ds_min) <= max_pairs_per_chunk:
                        break
                # Adjust n_in and end_pos
                n_in = reduced_n
                end_pos = pos + <uint64_t>n_in
                if verbose:
                    logging.info(f"  Reduced chunk to {n_in:,} kmers (data {(ds_max-ds_min):,} cells) to stay within memory cap")

            # If even after reduction the data range is too large (single kmer
            # with a massive cell list), process the data in sub-windows to
            # avoid allocating more than max_pairs_per_chunk cells at a time.
            if <uint64_t>(ds_max - ds_min) > max_pairs_per_chunk:
                if verbose:
                    logging.info(f"  Streaming {n_in} kmer(s) with {(ds_max-ds_min):,} cells in sub-windows")
                win_start = ds_min
                while win_start < ds_max:
                    win_end = min(win_start + <int64_t>max_pairs_per_chunk, ds_max)
                    data_block = np.ascontiguousarray(data_ds[win_start:win_end], dtype=np.uint32)
                    data_ptr = <uint32_t*>(<np.ndarray>data_block).data
                    data_block_len = len(data_block)

                    for i in range(n_in):
                        kmer_val = idx_ptr[i]
                        prefix_code = kmer_val >> rshift_val
                        suffix_code = <uint32_t>(kmer_val & suffix_mask_val)

                        if prefix_code != cur_prefix:
                            if cur_prefix != 0xFFFFFFFFFFFFFFFF and bucket_pairs.size() > 0:
                                _flush_bucket_c(cur_prefix, bucket_pairs, sfh, dfh,
                                    &swp, &dwp, <uint64_t*>pis.data, <uint64_t*>pid.data,
                                    <uint32_t*>pidsz.data, &tk, &nb)
                            cur_prefix = prefix_code
                            bucket_pairs.clear()

                        # Skip if bucket_pairs is already at the cap for this prefix
                        if bucket_pairs.size() >= max_bucket_pairs:
                            continue

                        # Adjust offsets relative to the current window
                        win_ds = <int64_t>ip_start_ptr[i] - win_start
                        win_de = <int64_t>ip_end_ptr[i] - win_start
                        # Clamp to the window boundaries
                        if win_ds < 0:
                            win_ds = 0
                        if win_de > <int64_t>data_block_len:
                            win_de = <int64_t>data_block_len
                        if win_de > win_ds and data_ptr != NULL:
                            for j in range(<uint64_t>win_ds, <uint64_t>win_de):
                                bucket_pairs.push_back(pair[uint32_t, uint32_t](suffix_code, data_ptr[j]))
                                if bucket_pairs.size() >= max_bucket_pairs:
                                    break

                    data_block = None
                    data_ptr = NULL
                    win_start = win_end
                    PyErr_CheckSignals()  # allow Ctrl-C during long streaming
            else:
                # Normal path: data fits in memory
                data_block = np.ascontiguousarray(data_ds[ds_min:ds_max], dtype=np.uint32)
                data_ptr = <uint32_t*>(<np.ndarray>data_block).data
                data_block_len = len(data_block)

                for i in range(n_in):
                    kmer_val = idx_ptr[i]
                    prefix_code = kmer_val >> rshift_val
                    suffix_code = <uint32_t>(kmer_val & suffix_mask_val)

                    if prefix_code != cur_prefix:
                        if cur_prefix != 0xFFFFFFFFFFFFFFFF and bucket_pairs.size() > 0:
                            _flush_bucket_c(cur_prefix, bucket_pairs, sfh, dfh,
                                &swp, &dwp, <uint64_t*>pis.data, <uint64_t*>pid.data,
                                <uint32_t*>pidsz.data, &tk, &nb)
                        cur_prefix = prefix_code
                        bucket_pairs.clear()

                    ds = <int64_t>ip_start_ptr[i] - ds_min
                    de = <int64_t>ip_end_ptr[i] - ds_min
                    if ds >= 0 and de > ds and data_ptr != NULL and de <= <int64_t>data_block_len:
                        for j in range(<uint64_t>ds, <uint64_t>de):
                            bucket_pairs.push_back(pair[uint32_t, uint32_t](suffix_code, data_ptr[j]))

                data_block = None
                data_ptr = NULL
        else:
            data_ptr = NULL
            data_block_len = 0

        # Release source arrays to free memory before next chunk.
        # bucket_pairs holds value copies, so source arrays are no longer needed.
        indices_arr = None
        indptr_arr = None
        ends_arr = None

        pos = end_pos

        now = time.time()
        if verbose and (now - last_report > 2.0 or pos >= total_indices):
            dt = now - t0
            logging.info(f"  {pos:,}/{total_indices:,} ({100.0*pos/total_indices:.1f}%) "
                         f"{pos/dt/1e6:.2f}M/s, {nb:,} buckets, {tk:,} kmers")
            last_report = now

    if cur_prefix != 0xFFFFFFFFFFFFFFFF and bucket_pairs.size() > 0:
        _flush_bucket_c(cur_prefix, bucket_pairs, sfh, dfh,
            &swp, &dwp, <uint64_t*>pis.data, <uint64_t*>pid.data,
            <uint32_t*>pidsz.data, &tk, &nb)

    fclose(sfh)
    fclose(dfh)

    with open(pp, 'wb') as f:
        f.write(pis.tobytes()); f.write(pid.tobytes()); f.write(pidsz.tobytes())

    meta = {'magic': int(MAGIC), 'version': int(VERSION), 'kmer_size': kmer_size,
            'l_prefix': l_prefix, 'l_suffix': l_suffix, 'n_kmers': int(tk),
            'n_cells': int(n_cells), 'n_prefixes': int(n_prefixes),
            'n_buckets_written': int(nb)}
    with open(mp, 'w') as f:
        json.dump(meta, f, indent=2)

    dt = time.time() - t0
    logging.info(f"Conversion done: {tk:,} kmers, {nb:,} buckets ({dt:.1f}s)")
    logging.info(f"  PI:{os.path.getsize(pp)/1e6:.1f}MB Suf:{os.path.getsize(sp)/1e6:.1f}MB Data:{os.path.getsize(dp)/1e6:.1f}MB")

    return int(tk), int(nb)

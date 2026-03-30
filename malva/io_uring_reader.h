// Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
//                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
//
// Non-commercial and academic use only. See LICENSE for full terms.

/*
 * io_uring_reader.h — High-QD batch pread via io_uring
 *
 * Usage:
 *   IoUringReader reader;
 *   reader.init(fd, 64);  // fd opened with O_RDONLY|O_DIRECT
 *   reader.submit_reads(offsets, sizes, dests, n);
 *   reader.destroy();
 *
 * All dest buffers must be 4096-aligned (for O_DIRECT).
 * All sizes must be 512-aligned.
 * This code is NOT thread-safe.
 */
#ifndef IO_URING_READER_H
#define IO_URING_READER_H

#ifdef __linux__

#include <liburing.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>

struct IoUringReader {
    struct io_uring ring;
    int fd;
    int queue_depth;
    bool initialized;

    IoUringReader() : fd(-1), queue_depth(0), initialized(false) {}

    /*
     * Initialise the ring.
     * fd:          file descriptor (O_RDONLY | O_DIRECT recommended)
     * qd:          max outstanding I/Os (32–256 typical)
     * Returns 0 on success, -errno on failure.
     */
    int init(int fd_, int qd) {
        fd = fd_;
        queue_depth = qd;
        int ret = io_uring_queue_init(qd, &ring, 0);
        if (ret < 0) return ret;
        initialized = true;
        return 0;
    }

    /*
     * Submit n reads and wait for all completions.
     *
     * offsets[i]:  byte offset in file
     * sizes[i]:    bytes to read (must be 512-aligned for O_DIRECT)
     * dests[i]:    destination buffer (must be 4096-aligned for O_DIRECT)
     * n:           number of reads
     *
     * Returns 0 on success. On partial I/O, returns the index of the
     * first failed read as -(index+1).
     *
     * Strategy: sliding window of queue_depth SQEs.
     */
    int submit_reads(const uint64_t* offsets,
                     const uint32_t* sizes,
                     uint8_t**       dests,
                     uint64_t        n)
    {
        if (!initialized || n == 0) return 0;

        uint64_t submitted = 0;   /* next index to submit */
        uint64_t completed = 0;   /* how many completions reaped */
        int inflight = 0;
        int ret;

        while (completed < n) {
            /* Fill SQ up to queue_depth */
            while (inflight < queue_depth && submitted < n) {
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (!sqe) break;  /* SQ full, flush first */

                io_uring_prep_read(sqe, fd,
                                   dests[submitted],
                                   sizes[submitted],
                                   offsets[submitted]);
                io_uring_sqe_set_data64(sqe, submitted);
                submitted++;
                inflight++;
            }

            /* Submit whatever is queued */
            ret = io_uring_submit(&ring);
            if (ret < 0 && ret != -EBUSY) return ret;

            /* Reap at least one completion */
            struct io_uring_cqe* cqe;
            ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) return ret;

            /* Drain all available completions */
            unsigned head;
            unsigned reaped = 0;
            io_uring_for_each_cqe(&ring, head, cqe) {
                uint64_t idx = io_uring_cqe_get_data64(cqe);
                if (cqe->res < 0) {
                    /* I/O error — record but keep draining */
                    io_uring_cq_advance(&ring, reaped + 1);
                    return -(int)(idx + 1);
                }
                reaped++;
                completed++;
                inflight--;
            }
            io_uring_cq_advance(&ring, reaped);
        }

        return 0;
    }

    void destroy() {
        if (initialized) {
            io_uring_queue_exit(&ring);
            initialized = false;
        }
    }

    ~IoUringReader() {
        destroy();
    }
};

/*
 * Convenience: round up val to next multiple of alignment.
 * alignment must be power of 2.
 */
static inline uint32_t align_up(uint32_t val, uint32_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

/*
 * Convenience: aligned allocation (for O_DIRECT buffers).
 * Returns NULL on failure.
 */
static inline void* aligned_alloc_safe(size_t alignment, size_t size) {
    if (size == 0) size = alignment;
    /* Ensure size is multiple of alignment (required by aligned_alloc) */
    size = (size + alignment - 1) & ~(alignment - 1);
    return aligned_alloc(alignment, size);
}

#else
/* ── Stub for non-Linux (macOS etc) — falls back to serial pread ──────── */

#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

struct IoUringReader {
    int fd;
    int queue_depth;
    bool initialized;

    IoUringReader() : fd(-1), queue_depth(0), initialized(false) {}

    int init(int fd_, int qd) {
        fd = fd_;
        queue_depth = qd;
        initialized = true;
        return 0;
    }

    int submit_reads(const uint64_t* offsets,
                     const uint32_t* sizes,
                     uint8_t**       dests,
                     uint64_t        n)
    {
        for (uint64_t i = 0; i < n; i++) {
            ssize_t rd = pread(fd, dests[i], sizes[i], offsets[i]);
            if (rd < 0) return -(int)(i + 1);
        }
        return 0;
    }

    void destroy() { initialized = false; }
    ~IoUringReader() { destroy(); }
};

static inline uint32_t align_up(uint32_t val, uint32_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

static inline void* aligned_alloc_safe(size_t alignment, size_t size) {
    if (size == 0) size = alignment;
    size = (size + alignment - 1) & ~(alignment - 1);
    void* p = NULL;
    posix_memalign(&p, alignment, size);
    return p;
}

#endif /* __linux__ */
#endif /* IO_URING_READER_H */
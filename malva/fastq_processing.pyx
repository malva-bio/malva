# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

# This is adapted from the dnaio package for improved throughput
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE, PyBytes_CheckExact
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.string cimport memcmp, memcpy, memchr, memmove
from libc.stdint cimport uint64_t, uint32_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort as stdsort

from libcpp.unordered_set cimport unordered_set
import numpy as np
cimport numpy as np
np.import_array()

# Use a larger lookup table for faster 16-bit chunk encoding
cdef uint32_t[65536] KMER_ENCODE_TABLE

# Initialize the lookup table
cdef int init_kmer_encode_table() nogil:
    cdef:
        uint32_t i, encoded
        unsigned char b1, b2
    for i in range(65536):
        b1, b2 = i & 255, i >> 8
        encoded = ((b1 >> 1) & 3) | (((b2 >> 1) & 3) << 2)
        KMER_ENCODE_TABLE[i] = encoded

init_kmer_encode_table()

cdef inline uint64_t encode_kmer(const unsigned char* sequence, int length) nogil:
    cdef:
        uint64_t result = 0
        int i = 0
    
    for i in range(i, length):
        result = (result << 2) | ((sequence[i] >> 1) & 3)
    
    return result

cdef class FastKmerProcessor:
    def __cinit__(self, int kmer_size=32, bint overlapping=False, int min_valid_sequence_size=0):
        if kmer_size < 8 or kmer_size > 32:
            raise ValueError("kmer_size must be between 8 and 32")
        self.kmer_size = kmer_size
        self.overlapping = overlapping
        self.min_valid_sequence_size = max(min_valid_sequence_size, kmer_size)
        
    cdef int process_sequence_chunk(self, const unsigned char* seq_ptr, Py_ssize_t length) except -1 nogil:
        cdef:
            uint64_t kmer
            int remaining = length
            int num_kmers
            int step = 1 if self.overlapping else self.kmer_size

        if seq_ptr == NULL or length < self.kmer_size:
            return 0

        num_kmers = (length - self.kmer_size + step) // step

        if seq_ptr[0] != b'N' and seq_ptr[0] != b'n':
            kmer = encode_kmer(seq_ptr, self.kmer_size)
            if kmer != 0:
                self.unique_kmers.insert(kmer)

        seq_ptr += step
        for i in range(1, num_kmers):
            if seq_ptr[self.kmer_size-1] != b'N' and seq_ptr[self.kmer_size-1] != b'n':
                kmer = encode_kmer(seq_ptr, self.kmer_size)
                if kmer != 0:
                    self.unique_kmers.insert(kmer)
            seq_ptr += step

        return 0

    cdef np.ndarray process_sequences(self, sequences):
        cdef:
            const unsigned char* seq_ptr
            Py_ssize_t seq_len
            bytes seq_bytes
            vector[uint64_t] sorted_kmers
            unordered_set[uint64_t].iterator it
            unordered_set[uint64_t].iterator end
            np.ndarray[np.uint64_t, ndim=1] result

        self.unique_kmers.clear()
        
        for group in sequences:
            for seq in group:
                if not isinstance(seq, str):
                    raise TypeError("Sequences must be strings")
                
                seq_bytes = seq.encode('ascii')
                seq_len = len(seq_bytes)
                
                if seq_len >= self.min_valid_sequence_size:
                    seq_ptr = <const unsigned char*>PyBytes_AS_STRING(seq_bytes)
                    with nogil:
                        self.process_sequence_chunk(seq_ptr, seq_len)

        sorted_kmers.reserve(self.unique_kmers.size())
        it = self.unique_kmers.begin()
        end = self.unique_kmers.end()
        
        while it != end:
            sorted_kmers.push_back(deref(it))
            inc(it)
            
        with nogil:
            stdsort(sorted_kmers.begin(), sorted_kmers.end())
            
        # Create numpy array from sorted vector
        result = np.empty(sorted_kmers.size(), dtype=np.uint64)
        memcpy(result.data, &sorted_kmers[0], sorted_kmers.size() * sizeof(uint64_t))
        
        return result

cdef class KmerFastqParser:
    """
    Parse a FASTQ file and yield k-mer arrays

    Arguments:
        file: a file-like object, opened in binary mode (it must have a readinto
            method)

        buffer_size: size of the initial buffer. This is automatically grown
            if a FASTQ record is encountered that does not fit.
        
        kmer_size: when sequence is converted to numeric, the k size of the k-mers.
            This can be 32 as maximum!

        jump_amount: what's the offset that's used to jump to the next k-mer. Setting this to 1
            with overlapping = False is equivalent to overlapping = True. Setting this to a value
            equal to kmer_size will lead to non-overlapping k-mers

        overlapping: whether k-mers will be fully overlapping (jump_amount = 1) or not. This
            overrides any jump_amount parameter value.


    Yields:
        An array of uint64 values, numerically encoding the (non) overlapping k-mers of the
        input sequence 
    """
    def __cinit__(self, file, Py_ssize_t buffer_size, int kmer_size = 32, int jump_amount = 16, bint overlapping = False):
        self.buffer_size = buffer_size
        self.buffer = <char *>PyMem_Malloc(self.buffer_size)
        if self.buffer == NULL:
            raise MemoryError()
        self.bytes_in_buffer = 0
        self.number_of_records = 0
        self.extra_newline = False
        self.eof = False
        self.record_start = self.buffer
        self.file = file
        self.kmer_size = kmer_size
        self.overlapping = overlapping
        self.jump_amount = jump_amount
        if buffer_size < 1:
            raise ValueError("Starting buffer size too small")

    def __dealloc__(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass
        PyMem_Free(self.buffer)

    cdef _read_into_buffer(self):
        # This function sets self.record_start at 0 and makes sure self.buffer
        # starts at the start of a FASTQ record. Any incomplete FASTQ remainder
        # of the already processed buffer is moved to the start of the buffer
        # and the rest of the buffer is filled up with bytes from the file.

        cdef char *tmp
        cdef Py_ssize_t remaining_bytes, empty_bytes_in_buffer, filechunk_size
        cdef object filechunk

        if self.record_start == self.buffer and self.bytes_in_buffer == self.buffer_size:
            # buffer too small, double it
            self.buffer_size *= 2
            tmp = <char *>PyMem_Realloc(self.buffer, self.buffer_size)
            if tmp == NULL:
                raise MemoryError()
            self.buffer = tmp
        else:
            # Move the incomplete record from the end of the buffer to the beginning.
            remaining_bytes = self.bytes_in_buffer - (self.record_start - self.buffer)
            memmove(self.buffer, self.record_start, remaining_bytes)
            self.bytes_in_buffer = remaining_bytes
    
        self.record_start = self.buffer
        empty_bytes_in_buffer = self.buffer_size - self.bytes_in_buffer

        # Read new data into the buffer
        filechunk = self.file.read(empty_bytes_in_buffer)
        if not PyBytes_CheckExact(filechunk):
            raise TypeError("self.file is not a binary file reader.")
    
        filechunk_size = PyBytes_GET_SIZE(filechunk)
        if filechunk_size > empty_bytes_in_buffer:
            raise ValueError(f"read() returned too much data: "
                             f"{empty_bytes_in_buffer} bytes requested, "
                             f"{filechunk_size} bytes returned.")

        memcpy(self.buffer + self.bytes_in_buffer, PyBytes_AS_STRING(filechunk), filechunk_size)
        self.bytes_in_buffer += filechunk_size

        if filechunk_size == 0:  # End of file
            if self.bytes_in_buffer == 0:
                self.eof = True
            elif not self.extra_newline and self.buffer[self.bytes_in_buffer - 1] != b'\n':
                self.buffer[self.bytes_in_buffer] = b'\n'
                self.bytes_in_buffer += 1
                self.extra_newline = True
            elif self.extra_newline:
                self.bytes_in_buffer -= 1
                raise Exception('Premature end of file encountered.')
            else:
                raise Exception('Premature end of file encountered.')

    def __iter__(self):
        return self

    cdef vector[uint64_t] next(self):
        cdef:
            char *name_end
            char *sequence_start
            char *sequence_end
            char *second_header_end
            char *qualities_start
            char *qualities_end
            char *buffer_end
            size_t remaining_bytes
            Py_ssize_t sequence_length
            int num_kmers
            int jump_amount
            int jump_mem
            vector[uint64_t] kmer_array
            unsigned long long result = 0
            int j
        # Repeatedly attempt to parse the buffer until we have found a full record.
        # If an attempt fails, we read more data before retrying.
        while True:
            buffer_end = self.buffer + self.bytes_in_buffer
            if self.eof:
                raise StopIteration()
    
            ### Check for a complete record (i.e 4 newlines are present)
            # Use libc memchr, this optimizes looking for characters by
            # using 64-bit integers. See:
            # https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=string/memchr.c;hb=HEAD
            # void *memchr(const void *str, int c, size_t n)
            name_end = <char *>memchr(self.record_start, b'\n', <size_t>(buffer_end - self.record_start))
            if name_end == NULL:
                self._read_into_buffer()
                continue
    
            # self.bytes_in_buffer - sequence_start is always nonnegative:
            # - name_end is at most self.bytes_in_buffer - 1
            # - thus sequence_start is at most self.bytes_in_buffer
            sequence_start = name_end + 1
            sequence_end = <char *>memchr(sequence_start, b'\n', <size_t>(buffer_end - sequence_start))
            if sequence_end == NULL:
                self._read_into_buffer()
                continue

            second_header_start = sequence_end + 1
            remaining_bytes = (buffer_end - second_header_start)

            # Usually there is no second header, so we skip the memchr call.
            if remaining_bytes > 2 and memcmp(second_header_start, b"+\n", 2) == 0:
                second_header_end = second_header_start + 1
            else:
                second_header_end = <char *>memchr(second_header_start, b'\n', <size_t>(remaining_bytes))
                if second_header_end == NULL:
                    self._read_into_buffer()
                    continue

            qualities_start = second_header_end + 1
            qualities_end = <char *>memchr(qualities_start, b'\n', <size_t>(buffer_end - qualities_start))
            if qualities_end == NULL:
                self._read_into_buffer()
                continue

            sequence_length = sequence_end - sequence_start
            # Check for \r\n line-endings and compensate
            if (sequence_end - 1)[0] == b'\r':
                sequence_length -= 1

            # Calculate the number of kmers
            if self.overlapping:
                jump_amount = 1
                num_kmers = (sequence_length - self.kmer_size) // jump_amount + 1
            else:
                jump_amount = self.jump_amount
                num_kmers = (sequence_length + self.kmer_size - 1) // self.kmer_size

            # Allocate memory for the kmer array
            kmer_array.resize(num_kmers)
    
            try:
                # Process the sequence in kmer_size chunks
                result = encode_kmer(<unsigned char*>sequence_start, self.kmer_size)
                kmer_array[0] = result
    
                for j in range(1, num_kmers):
                    start_index = j * jump_amount
                    if sequence_length - start_index < self.kmer_size:
                        jump_mem = sequence_length - start_index
                    else:
                        jump_mem = jump_amount
                    
                    sequence_start += jump_mem
                    result = encode_kmer(<unsigned char*>sequence_start, self.kmer_size)
                    kmer_array[j] = result

                self.number_of_records += 1
                self.record_start = qualities_end + 1
                return kmer_array
            except Exception:
                kmer_array.clear()
                raise

    def __repr__(self):
        return f"<KmerFastqParser records_processed={self.number_of_records}>"
    
    def __next__(self):
        return self.next()

cdef class SequenceFastqParser:
    """
    Parse sequences from FASTQ file and yield numeric encodings of short sequences (e.g., barcodes).
    Input sequence can be trimmed (start and end position).

    Adapted from dnaio.FastqIter

    Arguments:
        file: a file-like object, opened in binary mode (it must have a readinto
            method)

        buffer_size: size of the initial buffer. This is automatically grown
            if a FASTQ record is encountered that does not fit.
        
        kmer_size: when sequence is converted to numeric, the k size of the k-mers.
            This can be 32 as maximum!

    Yields:
        A uint64 value, encoding the (preprocessed) input sequence 
    """
    def __cinit__(self, file, Py_ssize_t buffer_size, int trim_start = 0, int trim_end = 32):
        self.buffer_size = buffer_size
        self.buffer = <char *>PyMem_Malloc(self.buffer_size)
        if self.buffer == NULL:
            raise MemoryError()
        self.bytes_in_buffer = 0
        self.number_of_records = 0
        self.extra_newline = False
        self.eof = False
        self.record_start = self.buffer
        self.file = file
        if buffer_size < 1:
            raise ValueError("Starting buffer size too small")

        self.trim_start = trim_start

        if trim_end < 0 or trim_start < 0:
            raise ValueError("`trim_start` and `trim_end` must be provided as positive integer numbers")
        self.trim_end = trim_end


    def __dealloc__(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass
        PyMem_Free(self.buffer)

    cdef _read_into_buffer(self):
        cdef char *tmp
        cdef Py_ssize_t remaining_bytes, empty_bytes_in_buffer, filechunk_size
        cdef object filechunk

        if self.record_start == self.buffer and self.bytes_in_buffer == self.buffer_size:
            # buffer too small, double it
            self.buffer_size *= 2
            tmp = <char *>PyMem_Realloc(self.buffer, self.buffer_size)
            if tmp == NULL:
                raise MemoryError()
            self.buffer = tmp
        else:
            # Move the incomplete record from the end of the buffer to the beginning.
            remaining_bytes = self.bytes_in_buffer - (self.record_start - self.buffer)
            memmove(self.buffer, self.record_start, remaining_bytes)
            self.bytes_in_buffer = remaining_bytes
    
        self.record_start = self.buffer
        empty_bytes_in_buffer = self.buffer_size - self.bytes_in_buffer

        # Read new data into the buffer
        filechunk = self.file.read(empty_bytes_in_buffer)
        if not PyBytes_CheckExact(filechunk):
            raise TypeError("self.file is not a binary file reader.")
    
        filechunk_size = PyBytes_GET_SIZE(filechunk)
        if filechunk_size > empty_bytes_in_buffer:
            raise ValueError(f"read() returned too much data: "
                             f"{empty_bytes_in_buffer} bytes requested, "
                             f"{filechunk_size} bytes returned.")

        memcpy(self.buffer + self.bytes_in_buffer, PyBytes_AS_STRING(filechunk), filechunk_size)
        self.bytes_in_buffer += filechunk_size

        if filechunk_size == 0:  # End of file
            if self.bytes_in_buffer == 0:
                self.eof = True
            elif not self.extra_newline and self.buffer[self.bytes_in_buffer - 1] != b'\n':
                self.buffer[self.bytes_in_buffer] = b'\n'
                self.bytes_in_buffer += 1
                self.extra_newline = True
            elif self.extra_newline:
                self.bytes_in_buffer -= 1
                raise Exception('Premature end of file encountered.')
            else:
                raise Exception('Premature end of file encountered.')

    def __iter__(self):
        return self

    cdef uint64_t next(self):
        cdef:
            char *name_end
            char *sequence_start
            char *sequence_end
            char *second_header_end
            char *qualities_start
            char *qualities_end
            char *buffer_end
            size_t remaining_bytes
            Py_ssize_t sequence_length
            uint64_t result = 0

        while True:
            buffer_end = self.buffer + self.bytes_in_buffer
            if self.eof:
                raise StopIteration()
    
            # Find the four newlines in one go
            name_end = <char *>memchr(self.record_start, b'\n', <size_t>(buffer_end - self.record_start))
            if name_end == NULL:
                self._read_into_buffer()
                continue

            sequence_start = name_end + 1
            sequence_end = <char *>memchr(sequence_start, b'\n', <size_t>(buffer_end - sequence_start))
            if sequence_end == NULL:
                self._read_into_buffer()
                continue

            second_header_start = sequence_end + 1
            remaining_bytes = (buffer_end - second_header_start)
            # Usually there is no second header, so we skip the memchr call.
            if remaining_bytes > 2 and memcmp(second_header_start, b"+\n", 2) == 0:
                second_header_end = second_header_start + 1
            else:
                second_header_end = <char *>memchr(second_header_start, b'\n', <size_t>(remaining_bytes))
                if second_header_end == NULL:
                    self._read_into_buffer()
                    continue

            qualities_start = second_header_end + 1
            qualities_end = <char *>memchr(qualities_start, b'\n', <size_t>(buffer_end - qualities_start))
            if qualities_end == NULL:
                self._read_into_buffer()
                continue

            sequence_length = sequence_end - sequence_start
            # Check for \r\n line-endings and compensate
            if (sequence_end - 1)[0] == b'\r':
                sequence_length -= 1

            # Apply trimming
            sequence_start += self.trim_start
            sequence_length = self.trim_end - self.trim_start
            
            result = encode_kmer(<unsigned char*>sequence_start, sequence_length)

            self.number_of_records += 1
            self.record_start = qualities_end + 1
            return result

    def __repr__(self):
        return f"<SequenceFastqParser records_processed={self.number_of_records}>"

    def __next__(self):
        return self.next()


def process_sra_batch(
    tuple segs_batch,
    int barcode_segment,
    int cdna_segment,
    object spatial_index,
    np.ndarray[np.uint64_t, ndim=1] ak,
    np.ndarray[np.uint32_t, ndim=1] ac,
    Py_ssize_t tp,
    int trim_start,
    int trim_end,
    int kmer_size,
    int jump_amount,
    Py_ssize_t cap,
):
    """
    Batch-process SRA spots from iter_raw_batched() output.

    segs_batch: tuple of list[bytes], one list per segment (from SRAReader.iter_raw_batched)
    barcode_segment: index of the segment containing the barcode
    cdna_segment: index of the segment containing the cDNA
    spatial_index: SpatialIndex object with lookup(uint64) -> uint32
    ak / ac: output arrays of k-mers and cell IDs (pre-allocated, length >= cap)
    tp: current write offset into ak/ac
    trim_start / trim_end: slice applied to the barcode segment bytes
    kmer_size / jump_amount: k-mer extraction parameters
    cap: capacity of ak/ac buffers

    Returns: new tp value, or -1 if buffer was full (caller should resize and retry).
    """
    cdef:
        list bc_list = segs_batch[barcode_segment]
        list cdna_list = segs_batch[cdna_segment]
        int n = len(bc_list)
        int i, j, num_kmers
        bytes bc_bytes, cdna_bytes
        const unsigned char* bc_ptr
        const unsigned char* cdna_ptr
        Py_ssize_t cdna_len, bc_raw_len
        uint64_t r1bc, kmer
        uint32_t cid
        int bc_len = trim_end - trim_start
        uint64_t* ak_ptr = <uint64_t*>ak.data
        uint32_t* ac_ptr = <uint32_t*>ac.data

    for i in range(n):
        bc_bytes = bc_list[i]
        bc_raw_len = len(bc_bytes)
        if bc_raw_len < trim_end:
            continue
        bc_ptr = <const unsigned char*>PyBytes_AS_STRING(bc_bytes)
        r1bc = encode_kmer(bc_ptr + trim_start, bc_len)
        if r1bc == 0:
            continue
        cid = <uint32_t>spatial_index.lookup(r1bc)
        if cid == 0:
            continue

        cdna_bytes = cdna_list[i]
        cdna_ptr = <const unsigned char*>PyBytes_AS_STRING(cdna_bytes)
        cdna_len = len(cdna_bytes)

        if cdna_len < kmer_size:
            continue

        num_kmers = (cdna_len - kmer_size) // jump_amount + 1
        for j in range(num_kmers):
            if j * jump_amount + kmer_size > cdna_len:
                break
            if tp >= cap:
                return <Py_ssize_t>(-1)
            kmer = encode_kmer(cdna_ptr + j * jump_amount, kmer_size)
            ak_ptr[tp] = kmer
            ac_ptr[tp] = cid
            tp += 1

    return tp


def process_bam_batch(
    list barcode_strs,
    list cdna_strs,
    object spatial_index,
    np.ndarray[np.uint64_t, ndim=1] ak,
    np.ndarray[np.uint32_t, ndim=1] ac,
    Py_ssize_t tp,
    int kmer_size,
    int jump_amount,
    Py_ssize_t cap,
):
    """
    Batch-process BAM records.

    barcode_strs: list of str (or None) — barcode tag values
    cdna_strs: list of str (or None) — sequence values (query_sequence or tag)
    spatial_index: SpatialIndex object with lookup(uint64) -> uint32
    ak / ac: output arrays (pre-allocated, length >= cap)
    tp: current write offset
    kmer_size / jump_amount: k-mer extraction parameters
    cap: buffer capacity

    Returns: new tp value, or -1 if buffer was full.
    """
    cdef:
        int n = len(barcode_strs)
        int i, j, num_kmers
        object bc_obj, cdna_obj
        bytes bc_bytes, cdna_bytes
        const unsigned char* bc_ptr
        const unsigned char* cdna_ptr
        Py_ssize_t bc_len, cdna_len
        uint64_t r1bc, kmer
        uint32_t cid
        uint64_t* ak_ptr = <uint64_t*>ak.data
        uint32_t* ac_ptr = <uint32_t*>ac.data

    for i in range(n):
        bc_obj = barcode_strs[i]
        if bc_obj is None:
            continue
        if isinstance(bc_obj, str):
            bc_bytes = (<str>bc_obj).encode('ascii')
        else:
            bc_bytes = bc_obj
        bc_ptr = <const unsigned char*>PyBytes_AS_STRING(bc_bytes)
        bc_len = len(bc_bytes)
        if bc_len == 0:
            continue
        r1bc = encode_kmer(bc_ptr, bc_len)
        if r1bc == 0:
            continue
        cid = <uint32_t>spatial_index.lookup(r1bc)
        if cid == 0:
            continue

        cdna_obj = cdna_strs[i]
        if cdna_obj is None:
            continue
        if isinstance(cdna_obj, str):
            cdna_bytes = (<str>cdna_obj).encode('ascii')
        else:
            cdna_bytes = cdna_obj
        cdna_ptr = <const unsigned char*>PyBytes_AS_STRING(cdna_bytes)
        cdna_len = len(cdna_bytes)

        if cdna_len < kmer_size:
            continue

        num_kmers = (cdna_len - kmer_size) // jump_amount + 1
        for j in range(num_kmers):
            if j * jump_amount + kmer_size > cdna_len:
                break
            if tp >= cap:
                return <Py_ssize_t>(-1)
            kmer = encode_kmer(cdna_ptr + j * jump_amount, kmer_size)
            ak_ptr[tp] = kmer
            ac_ptr[tp] = cid
            tp += 1

    return tp
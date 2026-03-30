# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport cython
cimport numpy as np

from libc.stdint cimport uint16_t, uint32_t, uint64_t
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.stdio cimport FILE, fopen, fwrite, fread, fclose, getline
from libcpp.vector cimport vector
from libcpp.utility cimport pair, move
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref
from cpython.exc cimport PyErr_CheckSignals

import logging
import numpy as np

from malva.fast_map cimport map
from malva.kmer_processing import encode_kmer, get_sliding_kmers_numeric

from rich.progress import track

np.import_array()

cdef extern from "<cstdio>" nogil:
    double atof(const char* nptr)

cdef struct LineData:
    float x
    float y
    uint64_t cell_bc

# SpatialIndex

cdef class SpatialIndex:
    """
    Maps 2-bit–encoded cell barcodes to spatial coordinates (or sequential cell IDs).

    Supports float32 (x, y) coordinates for standard spatial platforms and
    uint16 (x, y) for the compact STOmics format.
    """
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

    def lookup(self, uint64_t key):
        """Python-accessible barcode lookup. Returns 0 if not found."""
        return self.get_key(key)

    def get_coords(self):
        """Return all coordinates as an (N, 2) float32 array."""
        cdef np.ndarray[np.float32_t, ndim=2] arr = np.empty((self.coords.size(), 2), dtype=np.float32)
        cdef size_t i
        for i in range(self.coords.size()):
            arr[i, 0] = self.coords[i].first
            arr[i, 1] = self.coords[i].second
        return arr

    def num_items(self):
        """Number of registered barcodes."""
        return self.index.size()

    cdef uint32_t get_key(self, uint64_t key) noexcept nogil:
        """Return the index for a barcode (0 if absent)."""
        return self.index[key]

    def save_binary(self, str filename):
        """Write index to binary: [size: size_t][key: uint64, x: float, y: float, ...]."""
        cdef FILE* file = fopen(filename.encode('ascii'), "wb")
        if file == NULL:
            raise IOError(f"Cannot open file {filename} for writing")

        cdef size_t size = self.index.size()
        fwrite(&size, sizeof(size_t), 1, file)

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
            it += 1

        fclose(file)

    def load_binary(self, str filename):
        """Read index from a binary file written by save_binary()."""
        cdef FILE* file = fopen(filename.encode('ascii'), "rb")
        if file == NULL:
            raise IOError(f"Cannot open file {filename} for reading")

        cdef size_t size
        fread(&size, sizeof(size_t), 1, file)

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

    def load_binary_stomics(self, str filename, int barcode_length=25):
        """Load a STOmics .bin file; barcodes are bit-reversed on load (STOmics encoding)."""
        if barcode_length <= 0 or barcode_length > 32:
            raise ValueError("Barcode length must be between 1 and 32 base pairs")

        cdef FILE* file = fopen(filename.encode('ascii'), "rb")
        if file == NULL:
            raise IOError(f"Cannot open file {filename} for reading")

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

cdef void process_line(const char* line, int line_length, LineData* data,
                        bint encode=True) noexcept nogil:
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

cdef void process_line_whitelist(const char* line, int line_length,
                                  LineData* data) noexcept nogil:
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

def create_spatial_index(str spatial_barcode_file):
    """
    Build a SpatialIndex from a CSV/TSV file (header + rows of barcode, x, y).

    Returns:
        SpatialIndex: index mapping 2-bit–encoded barcodes to (x, y) float32 coords.
    """
    cdef:
        SpatialIndex sindex = SpatialIndex()
        FILE* file
        char* line = NULL
        size_t len = 0
        ssize_t read
        LineData data
        vector[pair[float, float]] coords
        uint32_t line_count = 0
        Py_ssize_t report_interval = 10_000_000

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

    sindex.coords = coords

    free(line)
    fclose(file)

    logging.info(f"Spatial index creation completed with {line_count:,} barcodes.")

    return sindex

def create_singlecell_index(str whitelist_file):
    """
    Build a SpatialIndex from a barcode whitelist (one barcode per line).

    Barcodes are assigned sequential IDs starting at 1 (0 = not found).
    """
    cdef:
        SpatialIndex sindex = SpatialIndex()
        FILE* file
        char* line = NULL
        size_t len = 0
        ssize_t read
        LineData data
        uint32_t line_count = 0
        Py_ssize_t report_interval = 10_000_000

    logging.info("Starting single-cell barcode index creation...")

    file = fopen(whitelist_file.encode('ascii'), "r")
    if file == NULL:
        raise IOError(f"Cannot open file {whitelist_file} for reading")

    getline(&line, &len, file)

    while True:
        read = getline(&line, &len, file)
        if read == -1:
            break
        process_line_whitelist(line, read, &data)

        sindex.add(data.cell_bc, line_count + 1)  # 1-based; 0 means "not found"

        line_count += 1
        if line_count % report_interval == 0:
            PyErr_CheckSignals()
            logging.info(f"Processed {line_count:,} barcodes.")

    logging.info(f"Single-cell index creation completed with {line_count:,} barcodes.")

    free(line)
    fclose(file)

    return sindex

# BackgroundModel

cdef class BackgroundModel:
    """
    K-mer frequency model built from a reference FASTA.

    Tracks how many distinct reference entries contain each k-mer; used to
    down-weight or skip ubiquitous k-mers during spatial queries.
    """
    cdef:
        map[uint64_t, uint16_t] model
        size_t total_mers
        size_t kmer_size
        bint verbose

    def __cinit__(self, int kmer_size, bint verbose=True):
        self.model = map[uint64_t, uint16_t]()
        self.total_mers = 0
        self.kmer_size = kmer_size
        self.verbose = verbose

    def create_from_reference(self, str filename, bint consecutive_genes=True):
        """
        Build the model from a FASTA file.

        Each entry (or group of consecutive entries with the same gene name prefix)
        contributes at most 1 count per k-mer to avoid over-weighting abundant genes.
        """
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
            iterator = track(iterate_fasta(filename),
                             description=f'Computing background {self.kmer_size}-mer')
        else:
            iterator = iterate_fasta(filename)

        for seq in iterator:
            it_gene_name = seq[0].split(":")[0]

            all_kmer_seq = get_sliding_kmers_numeric(seq[1], self.kmer_size,
                                                     remove_noncomplex=True)
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
                    self.model[key] += 1
                it += 1

            current_gene = it_gene_name
            temp_model.clear()

        if self.verbose:
            logging.info(f"Processed {self.model.size()} unique {self.kmer_size}-mers "
                         f"across {self.total_mers} occurrences")

    def is_mer_above_cutoff(self, uint64_t kmer, uint16_t cutoff):
        """Return True if kmer appears in more than cutoff reference entries (unknown → False)."""
        if self.model.find(kmer) == self.model.end():
            return False
        return self.model[kmer] > cutoff

    def save(self, str filename):
        """Write model to binary: [size: size_t][key: uint64, count: uint16, ...]."""
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
        """Load model from a binary file written by save()."""
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
        """Import from a Jellyfish FASTA (record name = count, body = k-mer sequence)."""
        from malva.reader import iterate_fasta
        from malva.kmer_processing import encode_kmer as _encode_kmer

        cdef:
            uint64_t key
            uint16_t count

        self.model.clear()
        self.total_mers = 0

        if self.verbose:
            iterator = track(iterate_fasta(filename),
                             description=f'Computing background {self.kmer_size}-mer frequency from {filename}')
        else:
            iterator = iterate_fasta(filename)

        for seq in iterator:
            self.model[_encode_kmer(seq[1], self.kmer_size)] = <uint16_t>int(seq[0])
            self.total_mers += 1

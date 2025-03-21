# distutils: language = c++
#cython: boundscheck=False, wraparound=False, initializedcheck=False, overflowcheck=False, cdivision=True, language_level=3

from libcpp.vector cimport vector
from libcpp.string cimport string as cpp_string
from cython.operator cimport dereference as deref
from libc.string cimport memcpy, strlen
from libcpp.utility cimport move
from libc.stdint cimport uint16_t, uint32_t, int32_t, uint64_t
from libcpp.algorithm cimport remove_if

# Define base encoding as a constant array
cdef int[256] BASE_ENCODING
for i in range(256):
    BASE_ENCODING[i] = 4  # Invalid base
BASE_ENCODING[<int>b'A'] = 0
BASE_ENCODING[<int>b'N'] = 3
BASE_ENCODING[<int>b'C'] = 1
BASE_ENCODING[<int>b'T'] = 2
BASE_ENCODING[<int>b'U'] = 2
BASE_ENCODING[<int>b'G'] = 3

BASE_REV_ENCODING = {0: 'A', 1: 'C', 2: 'T', 3: 'G'}

cdef inline int encode_base(char base):
    """Encode a DNA base to a numeric value."""
    return BASE_ENCODING[base]

def encode_kmer(str kmer):
    """Encode a DNA k-mer to a numeric value."""
    cdef uint64_t value = 0
    cdef int base
    for base_char in kmer:
        base = encode_base(base_char)
        value = (value << 2) | base
    return value

def decode_kmer(uint64_t value, int k):
    cdef str kmer = ""
    """Decode a numeric value to a DNA k-mer."""
    for i in range(k):
        base = value & 3
        kmer = BASE_REV_ENCODING[base] + kmer
        value = value >> 2
    
    return kmer

cdef inline uint64_t _internal_encode_kmer(str kmer):
    """Encode a DNA k-mer to a numeric value."""
    cdef uint64_t value = 0
    cdef int base
    for base_char in kmer:
        base = encode_base(base_char)
        value = (value << 2) | base
    return value

def get_kmers_numeric(str string, int k, remove_noncomplex=False):
    """Get a list of numerically encoded non-overlapping k-mers from a DNA string."""
    cdef:
        int n = len(string)
        int i
        str kmer, prev_kmer
        uint64_t encoded_kmer
        list kmers = []

    for i in range(0, n, k):
        kmer = string[i:i+k]
        if len(kmer) < k:
            # Take overlapping nucleotides from the previous k-mer
            prev_kmer = string[max(0, i-k):i]
            kmer = prev_kmer[-(k-len(kmer)):] + kmer

        if remove_noncomplex and 'N' in kmer:
            kmers.append(0)
            continue
    
        encoded_kmer = _internal_encode_kmer(kmer)
        kmers.append(encoded_kmer)

    return kmers

def get_sliding_kmers_numeric(str string, int k, remove_noncomplex=False):
    """Get a list of numerically encoded overlapping k-mers from a DNA string."""
    cdef:
        int n = len(string)
        int i
        str kmer, prev_kmer
        uint64_t encoded_kmer
        list kmers = []

    for i in range(0, n - k + 1):
        kmer = string[i:i+k]

        if remove_noncomplex and 'N' in kmer:
            kmers.append(0)
            continue
    
        encoded_kmer = _internal_encode_kmer(kmer)
        kmers.append(encoded_kmer)

    return kmers

cdef inline bint has_N(const char* seq, size_t length) nogil:
    """Check for N in sequence using byte comparison."""
    cdef size_t i
    for i in range(length):
        if seq[i] == b'N':
            return True
    return False

cdef inline uint64_t fast_encode_kmer(const char* kmer, int k) nogil:
    """Fast k-mer encoding using byte-level operations."""
    cdef:
        uint64_t value = 0
        int i
        char base
        
    for i in range(k):
        base = BASE_ENCODING[<char>kmer[i]]
        value = (value << 2) | base
    return value

cdef class FastKmerExtractor:
    """Efficient k-mer extraction and processing class."""
    def __cinit__(self, int kmer_size, bint remove_noncomplex=True):
        self.kmer_size = kmer_size
        self.remove_noncomplex = remove_noncomplex
        
    cdef void get_sliding_sequences(self, const char* sequence, int sliding_size) nogil:
        """Extract sliding sequences efficiently."""
        cdef:
            size_t seq_len = strlen(sequence)
            size_t i
            cpp_string substr
            
        self.sliding_seqs.clear()
        
        if seq_len < sliding_size:
            return
            
        for i in range(seq_len - sliding_size + 1):
            substr = cpp_string(sequence + i, sliding_size)
            self.sliding_seqs.push_back(substr)
            
    cdef void process_sequence(self, const char* seq, int sliding_size) nogil:
        """Process a single sequence to generate k-mers."""
        cdef:
            size_t i
            cpp_string current_seq
            const char* subseq
            uint64_t encoded_kmer
            size_t j, n_kmers
            size_t remaining_size
            
        self.get_sliding_sequences(seq, sliding_size)
        
        for i in range(self.sliding_seqs.size()):
            current_seq = self.sliding_seqs[i]
            subseq = current_seq.c_str()
            
            if self.remove_noncomplex and has_N(subseq, current_seq.size()):
                continue
                
            remaining_size = sliding_size - self.kmer_size + 1
            n_kmers = remaining_size / self.kmer_size
            
            j = 0
            while j < n_kmers:
                encoded_kmer = fast_encode_kmer(subseq + (j * self.kmer_size), self.kmer_size)
                self.kmers.push_back(encoded_kmer)
                j += 1
                
    cdef vector[uint64_t] process_sequence_group(self, list sequences, int sliding_size) except *:
        """Process a group of sequences to generate k-mers."""
        cdef:
            str seq
            bytes seq_bytes
            vector[uint64_t] result_kmers
            int local_sliding_size
            
        self.kmers.clear()
        
        for seq in sequences:
            local_sliding_size = sliding_size
            if local_sliding_size == 0:
                local_sliding_size = max(self.kmer_size, len(seq) - self.kmer_size)
                
            seq_bytes = seq.encode('ascii')
            self.process_sequence(seq_bytes, local_sliding_size)
                
        result_kmers = move(self.kmers)
        return result_kmers

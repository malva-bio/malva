# distutils: language = c++
#cython: boundscheck=False, wraparound=False, initializedcheck=False, overflowcheck=False, cdivision=True, language_level=3

import array
from collections import Counter
cimport cython

ctypedef unsigned long long uint64_t
ctypedef unsigned int uint32_t

cdef int BASE_ENCODING[128]
BASE_ENCODING[ord('A')] = 0
BASE_ENCODING[ord('N')] = 3
BASE_ENCODING[ord('C')] = 1
BASE_ENCODING[ord('T')] = 2
BASE_ENCODING[ord('U')] = 2
BASE_ENCODING[ord('G')] = 3

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
    cdef int n = len(string)
    cdef int i, j, kmer_len
    cdef str kmer, prev_kmer
    cdef uint64_t encoded_kmer

    if k > 16:
        kmers = array.array('Q')
    else:
        kmers = array.array('I')

    for i in range(0, n, k):
        kmer_len = k
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

def get_overlapping_kmers_numeric(str string, int k):
    """Get a list of numerically encoded non-overlapping k-mers from a DNA string."""
    cdef int n = len(string)
    cdef int i, j, kmer_len
    cdef str kmer, prev_kmer
    cdef uint64_t encoded_kmer

    if k > 16:
        kmers = array.array('Q')
    else:
        kmers = array.array('I')

    for i in range(0, n):
        kmer_len = k
        kmer = string[i:i+k]
        if len(kmer) < k:
            # Take overlapping nucleotides from the previous k-mer
            prev_kmer = string[max(0, i-k):i]
            kmer = prev_kmer[-(k-len(kmer)):] + kmer
        else:
            encoded_kmer = _internal_encode_kmer(kmer)
            kmers.append(encoded_kmer)

    return kmers
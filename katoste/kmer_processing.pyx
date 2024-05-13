#cython: boundscheck=False, wraparound=False, initializedcheck=False, overflowcheck=False, cdivision=True, language_level=3

import array
from collections import Counter

BASE_ENCODING = {'A': 0, 'N': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}

def encode_base(base):
    """Encode a DNA base to a numeric value."""
    return BASE_ENCODING[base]

def encode_kmer(kmer):
    """Encode a DNA k-mer to a numeric value."""
    value = 0
    for base in kmer:
        value = (value << 2) | encode_base(base)
    return value

def get_kmers_numeric(string, k, remove_noncomplex=False):
    """Get a list of numerically encoded non-overlapping k-mers from a DNA string."""
    n = len(string)
    if k > 16:
        kmers = array.array('Q')
    else:
        kmers = array.array('I')

    for i in range(0, n, k):
        kmer = string[i:i+k]
        if len(kmer) < k:
            # Take overlapping nucleotides from the previous k-mer
            prev_kmer = string[i-k:i]
            kmer = prev_kmer[-(k-len(kmer)):] + kmer

        if remove_noncomplex:
            _max_counter = (k - int(k/2))
            counter = Counter(kmer)
            if max(counter.values()) > _max_counter:
                continue

        encoded_kmer = encode_kmer(kmer)
        kmers.append(encoded_kmer)

    return kmers
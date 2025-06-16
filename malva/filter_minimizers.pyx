# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True, initializedcheck=True, cdivision=False

"""
a MinHash-based short word filtering approach for clustering k-mers
compress billions of k-mers into a much smaller number of clusters by 
grouping k-mers with similar composition of short words (w-mers).
"""

cimport cython
cimport numpy as np
from libc.stdint cimport uint64_t, int32_t
import numpy as np


cdef inline uint64_t __murmur(uint64_t x) nogil:
    """
    murmur hash for a w-mer.
    
    Parameters:
        x (uint64_t): Input w-mer as an integer.
        
    Returns:
        uint64_t: Hashed value.
    """
    cdef uint64_t h = x
    h ^= h >> 33
    h *= <uint64_t>0xff51afd7ed558ccd
    h ^= h >> 33
    h *= <uint64_t>0xc4ceb9fe1a85ec53
    h ^= h >> 33
    return h

cdef inline uint64_t __xxh64(uint64_t x) nogil:
    """
    xxh64 hash for a w-mer.
    
    Parameters:
        x (uint64_t): Input w-mer as an integer.
        
    Returns:
        uint64_t: Hashed value.
    """
    cdef uint64_t h = x
    h ^= h >> 33
    h *= <uint64_t>0x9e3779b185ebca87
    h ^= h >> 33
    h *= <uint64_t>0xc2b2ae3d27d4eb4f
    h ^= h >> 33
    return h

cdef inline uint64_t compute_signature(uint64_t kmer, int k, int w) nogil:
    """
    Compute a composite signature for a 64-bit encoded k-mer based on its w-mer composition.
    
    For each contiguous w-mer in the k-mer, two hash values (via __murmur and __xxh64) are computed.
    The minimum hash values over all w-mers are then combined into a single 64-bit composite signature.
    
    Parameters:
        kmer (uint64_t): 64-bit encoded k-mer.
        k (int): Length of the k-mer (in nucleotides).
        w (int): Length of the short word (w-mer) used for signature computation.
        
    Returns:
        uint64_t: Composite signature.
    """
    cdef int n_words = k - w + 1
    cdef int i
    cdef uint64_t current_word, min1_val, min2_val, h1_val, h2_val, composite
    # Initialize minima to the maximum 64-bit value.
    min1_val = <uint64_t>-1
    min2_val = <uint64_t>-1
    for i in range(n_words):
        # Extract the w-mer at position i.
        current_word = (kmer >> (2 * (k - w - i))) & ((<uint64_t>1 << (2 * w)) - 1)
        h1_val = __murmur(current_word)
        h2_val = __xxh64(current_word)
        if h1_val < min1_val:
            min1_val = h1_val
        if h2_val < min2_val:
            min2_val = h2_val
    # Combine the two minima into one composite signature.
    composite = min1_val ^ (min2_val * <uint64_t>0x9e3779b97f4a7c15)
    return composite


def filter_kmers(np.ndarray[np.uint64_t, ndim=1] kmers, int k, int w, int num_buckets):
    """
    Stream and filter 64-bit encoded k-mers into a fixed set of buckets.
    
        bucket = compute_signature(kmer, k, w) % num_buckets
    
    Parameters:
        kmers (np.ndarray[np.uint64_t]): 1D numpy array of 64-bit encoded k-mers.
        k (int): Length of the k-mer (in nucleotides).
        w (int): Length of the short word (w-mer) for signature computation.
        num_buckets (int): Predefined number of buckets.
    
    Returns:
        np.ndarray[np.int32]: 1D array (of length kmers.shape[0]) containing the bucket index
                              for each k-mer.
    """
    cdef Py_ssize_t n = kmers.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] buckets_out = np.empty(n, dtype=np.int32)
    cdef Py_ssize_t i
    cdef uint64_t sig
    cdef int bucket_idx
    for i in range(n):
        sig = compute_signature(kmers[i], k, w)
        bucket_idx = <int>(sig % num_buckets)
        buckets_out[i] = bucket_idx
    return buckets_out

cdef class KmerFilter:
    """
    encapsulates the filter approach so that k-mers can be processed on the fly
    (e.g., from a sorted stream) without dynamic reallocation or global state.
    
    Attributes:
        k (int): Length of the k-mers (in nucleotides).
        w (int): Length of the short word (w-mer) used for signature computation.
        num_buckets (int): Predefined number of buckets.
    """
    cdef public int k
    cdef public int w
    cdef public int num_buckets

    def __cinit__(self, int k, int w, int num_buckets):
        """
        Initialize the KmerFilter.
        
        Parameters:
            k (int): Length of the k-mers.
            w (int): Length of the short word (w-mer) for signature computation.
            num_buckets (int): Predefined number of buckets.
        
        Raises:
            ValueError: If k is less than w.
        """
        if k < w:
            raise ValueError("k (k-mer length) must be greater than or equal to w (w-mer length)")
        self.k = k
        self.w = w
        self.num_buckets = num_buckets

    cpdef np.ndarray filter_stream(self, np.ndarray[np.uint64_t, ndim=1] kmers):
        """
        Filter a stream of 64-bit encoded k-mers, assigning each to a bucket.
        
        Parameters:
            kmers (np.ndarray[np.uint64_t]): 1D numpy array of encoded k-mers.
        
        Returns:
            np.ndarray: 1D array (int32) of bucket assignments.
        """
        return filter_kmers(kmers, self.k, self.w, self.num_buckets)
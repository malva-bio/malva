from malva.kmer_processing import encode_kmer, decode_kmer, get_kmers_numeric, get_sliding_kmers_numeric

class TestKmerProcessing:
    def test_encode_decode_kmer(self):
        test_kmers = ['ACGT', 'AAAA', 'CCCC', 'GGGG', 'TTTT']
        for kmer in test_kmers:
            encoded = encode_kmer(kmer)
            decoded = decode_kmer(encoded, len(kmer))
            assert decoded == kmer
    
    def test_get_kmers_numeric(self):
        sequence = 'ACGTACGT'
        k = 4
        kmers = get_kmers_numeric(sequence, k)
        assert len(kmers) == 2
        
        # Test with non-complex sequences
        sequence_with_n = 'ACGTNACGT'
        kmers = get_kmers_numeric(sequence_with_n, k, remove_noncomplex=True)
        assert 0 in kmers
    
    def test_get_sliding_kmers_numeric(self):
        sequence = 'ACGTACGT'
        k = 4
        kmers = get_sliding_kmers_numeric(sequence, k)
        assert len(kmers) == 5
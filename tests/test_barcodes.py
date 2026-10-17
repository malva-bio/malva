import struct

import numpy as np
import pytest

from malva.barcodes import (
    BackgroundModel,
    SpatialIndex,
    create_singlecell_index,
    create_spatial_index,
)
from malva.fastq_processing import process_bam_batch, process_sra_batch
from malva.kmer_processing import encode_kmer


class TestSpatialIndex:
    def test_create_from_csv_and_coords(self, tmp_path):
        csv = tmp_path / "spatial.csv"
        csv.write_text("barcode,x,y\nACGTACGT,1.5,2.5\nTTTTGGGG,3.0,4.0\n")
        sindex = create_spatial_index(str(csv))
        assert sindex.num_items() == 2
        coords = sindex.get_coords()
        assert coords.shape == (2, 2)
        assert coords.dtype == np.float32

    def test_lookup_known_barcode(self, tmp_path):
        csv = tmp_path / "spatial.csv"
        csv.write_text("barcode,x,y\nACGTACGT,1.5,2.5\nTTTTGGGG,3.0,4.0\n")
        sindex = create_spatial_index(str(csv))
        # First barcode maps to id 0, second to id 1.
        assert sindex.lookup(encode_kmer("ACGTACGT")) == 0
        assert sindex.lookup(encode_kmer("TTTTGGGG")) == 1

        missing = encode_kmer("CCCCAAAA")
        assert sindex.lookup(missing, 0xFFFFFFFF) == 0xFFFFFFFF
        # Looking up an absent barcode must not mutate the index.
        assert sindex.num_items() == 2

    def test_sra_batch_preserves_spatial_cell_zero(self, tmp_path):
        csv = tmp_path / "spatial.csv"
        csv.write_text("barcode,x,y\nACGTACGT,1.5,2.5\nTTTTGGGG,3.0,4.0\n")
        sindex = create_spatial_index(str(csv))
        ak = np.zeros(8, dtype=np.uint64)
        ac = np.zeros(8, dtype=np.uint32)
        seen = np.zeros(2, dtype=np.uint8)

        result = process_sra_batch(
            ([b"ACGTACGT", b"CCCCAAAA"], [b"ACGTACGT", b"ACGTACGT"]),
            0,
            1,
            sindex,
            ak,
            ac,
            0,
            0,
            8,
            4,
            4,
            len(ak),
            seen,
        )

        assert result == (2, 2, 1, 16, 8)
        assert ac[:2].tolist() == [0, 0]
        assert seen.tolist() == [1, 0]

    def test_bam_batch_preserves_spatial_cell_zero(self, tmp_path):
        csv = tmp_path / "spatial.csv"
        csv.write_text("barcode,x,y\nACGTACGT,1.5,2.5\nTTTTGGGG,3.0,4.0\n")
        sindex = create_spatial_index(str(csv))
        ak = np.zeros(8, dtype=np.uint64)
        ac = np.zeros(8, dtype=np.uint32)
        seen = np.zeros(2, dtype=np.uint8)

        result = process_bam_batch(
            ["ACGTACGT", "CCCCAAAA"],
            ["ACGTACGT", "ACGTACGT"],
            sindex,
            ak,
            ac,
            0,
            4,
            4,
            len(ak),
            seen,
        )

        assert result == (2, 2, 1, 16, 8)
        assert ac[:2].tolist() == [0, 0]
        assert seen.tolist() == [1, 0]

    def test_save_and_load_binary_roundtrip(self, tmp_path):
        csv = tmp_path / "spatial.csv"
        csv.write_text("barcode,x,y\nACGTACGT,1.5,2.5\nTTTTGGGG,3.0,4.0\n")
        sindex = create_spatial_index(str(csv))

        out = tmp_path / "spatial.bin"
        sindex.save_binary(str(out))
        loaded = SpatialIndex()
        loaded.load_binary(str(out))

        assert loaded.num_items() == sindex.num_items()
        np.testing.assert_allclose(loaded.get_coords(), sindex.get_coords())

    def test_load_binary_stomics(self, tmp_path):
        bin_file = tmp_path / "spatial_stomics.bin"
        with open(bin_file, "wb") as f:
            for i in range(3):
                f.write(struct.pack("<Q", i + 1))
                f.write(struct.pack("<II", i * 10, i * 5))

        sindex = SpatialIndex()
        sindex.load_binary_stomics(str(bin_file), barcode_length=25)
        coords = sindex.get_coords_stomics()
        assert coords.shape == (3, 2)
        assert coords.dtype == np.uint16

    def test_load_binary_stomics_rejects_bad_length(self, tmp_path):
        empty = tmp_path / "empty.bin"
        empty.write_bytes(b"")
        sindex = SpatialIndex()
        with pytest.raises(ValueError):
            sindex.load_binary_stomics(str(empty), barcode_length=0)

    def test_singlecell_index_is_one_based(self, tmp_path):
        whitelist = tmp_path / "whitelist.txt"
        whitelist.write_text("ACGTACGTACGTACGT\nTTTTGGGGCCCCAAAA\n")
        sindex = create_singlecell_index(str(whitelist))
        assert sindex.num_items() == 2
        # 0 is the not-found sentinel, so the first barcode maps to 1.
        assert sindex.lookup(encode_kmer("ACGTACGTACGTACGT")) == 1


class TestBackgroundModel:
    @pytest.fixture
    def reference_fasta(self, tmp_path):
        fasta = tmp_path / "reference.fa"
        fasta.write_text(">geneA:tx1\nACGTACGTACGT\n>geneB:tx2\nACGTACGTACGT\n")
        return fasta

    def test_counts_distinct_genes(self, reference_fasta):
        model = BackgroundModel(4, verbose=False)
        model.create_from_reference(str(reference_fasta), consecutive_genes=True)

        shared = encode_kmer("ACGT")
        absent = encode_kmer("TTTT")
        assert model.is_mer_above_cutoff(shared, 1) is True
        assert model.is_mer_above_cutoff(shared, 2) is False
        assert model.is_mer_above_cutoff(absent, 0) is False

    def test_save_and_load_roundtrip(self, reference_fasta, tmp_path):
        model = BackgroundModel(4, verbose=False)
        model.create_from_reference(str(reference_fasta), consecutive_genes=True)

        out = tmp_path / "background.bin"
        model.save(str(out))
        loaded = BackgroundModel(4, verbose=False)
        loaded.load(str(out))

        shared = encode_kmer("ACGT")
        assert loaded.is_mer_above_cutoff(shared, 1) == model.is_mer_above_cutoff(shared, 1)

    def test_export_fasta_not_implemented(self, tmp_path):
        model = BackgroundModel(4, verbose=False)
        with pytest.raises(NotImplementedError):
            model.export_fasta(str(tmp_path / "out.fa"))

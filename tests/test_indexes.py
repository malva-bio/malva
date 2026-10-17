"""Tests for the prefix-bucketed index (``malva.malva_index``) and spatial barcodes.

The index was refactored from the legacy HDF5 layout to the prefix-bucketed
``pi.bin`` / ``suffixes.bin`` / ``data.bin`` layout, with ``MalvaIndex`` moved to
``malva.malva_index`` and the barcode index to ``malva.barcodes``. These tests
exercise the current API end to end on small, deterministic datasets.
"""

import numpy as np

from malva.barcodes import SpatialIndex, create_singlecell_index, create_spatial_index
from malva.kmer_processing import encode_kmer
from malva.malva_index import MalvaIndex
from tests.index_helpers import (
    ABSENT,
    OPENST_BARCODES,
    QUERY,
    SC_BARCODES,
    build_index,
)


class TestSpatialIndex:
    def test_spatial_index_from_csv(self, tmp_path):
        spatial_file = tmp_path / "spatial.csv"
        with open(spatial_file, "w") as f:
            f.write("barcode,x,y\n")
            for i, barcode in enumerate(OPENST_BARCODES):
                f.write(f"{barcode},{i * 10.0},{i * 5.0}\n")

        sindex = create_spatial_index(str(spatial_file))
        assert isinstance(sindex, SpatialIndex)
        assert sindex.num_items() == len(OPENST_BARCODES)
        coords = sindex.get_coords()
        assert coords.shape == (len(OPENST_BARCODES), 2)
        assert coords.dtype == np.float32

    def test_singlecell_index_from_whitelist(self, tmp_path):
        whitelist = tmp_path / "whitelist.txt"
        whitelist.write_text("\n".join(SC_BARCODES) + "\n")
        sindex = create_singlecell_index(str(whitelist))
        assert sindex.num_items() == len(SC_BARCODES)


class TestIndexBuildAndQuery:
    def test_build_reopen_and_query_spatial(self, tmp_path):
        index_dir = build_index(tmp_path, OPENST_BARCODES, (2, 27), QUERY, spatial=True)

        index = MalvaIndex(str(index_dir))
        index.open()
        try:
            assert index.n_spatial == len(OPENST_BARCODES)
            assert "spatial_coord" in index.index

            results = index.where(QUERY, sliding_size=24, pct_threshold=0.65, count_at_least=1)
            # One sequence -> one group; a group is (locations, counts, matches).
            assert len(results) == 1
            locations, counts, matches = results[0]
            assert isinstance(locations, np.ndarray) and locations.dtype == np.uint32
            assert isinstance(counts, np.ndarray) and counts.dtype == np.uint32
            assert isinstance(matches, list)

            # Every crafted read carries QUERY, including spatial cell id 0.
            assert set(locations.tolist()) == set(range(len(OPENST_BARCODES)))
        finally:
            index.close()

    def test_absent_sequence_returns_empty(self, tmp_path):
        index_dir = build_index(tmp_path, OPENST_BARCODES, (2, 27), QUERY, spatial=True)
        index = MalvaIndex(str(index_dir))
        index.open()
        try:
            locations, counts, _ = index.where(
                ABSENT, sliding_size=24, pct_threshold=0.65, count_at_least=1
            )[0]
            assert locations.size == 0
            assert counts.size == 0
        finally:
            index.close()

    def test_absent_kmers_after_present_kmers_do_not_crash(self, tmp_path):
        """Regression: a batch with absent kmers after a present batch must not
        dereference the stale CSR buffers left by the previous query."""
        index_dir = build_index(tmp_path, OPENST_BARCODES, (2, 27), QUERY, spatial=True)
        index = MalvaIndex(str(index_dir))
        index.open()
        try:
            present = index.where(QUERY, sliding_size=24, pct_threshold=0.65, count_at_least=1)
            assert present[0][0].size > 0

            absent = index.where(ABSENT, sliding_size=24, pct_threshold=0.65, count_at_least=1)
            assert absent[0][0].size == 0

            # A mixed batch keeps the present hits and drops the absent ones.
            mixed = index.where(
                [QUERY, ABSENT], sliding_size=24, pct_threshold=0.65, count_at_least=1
            )
            assert len(mixed) == 1
        finally:
            index.close()

    def test_where_groups_list_of_sequences_as_one_group(self, tmp_path):
        index_dir = build_index(tmp_path, OPENST_BARCODES, (2, 27), QUERY, spatial=True)
        index = MalvaIndex(str(index_dir))
        index.open()
        try:
            # A flat list is a single group; a list of lists is one group each.
            assert len(index.where([QUERY, ABSENT], sliding_size=24)) == 1
            assert len(index.where([[QUERY], [ABSENT]], sliding_size=24)) == 2
        finally:
            index.close()

    def test_build_reopen_and_query_singlecell(self, tmp_path):
        index_dir = build_index(tmp_path, SC_BARCODES, (0, 16), QUERY, spatial=False)
        index = MalvaIndex(str(index_dir))
        index.open()
        try:
            assert index.n_spatial == len(SC_BARCODES)
            locations, _, _ = index.where(
                QUERY, sliding_size=24, pct_threshold=0.65, count_at_least=1
            )[0]
            assert locations.size >= 1
            # Single-cell ids are 1-based (0 is the not-found sentinel).
            assert np.all(locations >= 1)
            assert np.all(locations <= len(SC_BARCODES))
        finally:
            index.close()


class TestMalvaIndexMetadata:
    def test_new_index_is_not_open(self, tmp_path):
        index = MalvaIndex(str(tmp_path / "empty"), kmer_size_initialize=24)
        assert index.n_chunks == 0
        assert "spatial_coord" not in index.index

    def test_ids_are_identity_without_merged_projects(self, tmp_path):
        index = MalvaIndex(str(tmp_path / "empty"), kmer_size_initialize=24)
        assert index.get_project_id(7) == 7
        assert index.get_cell_id(7) == 7

    def test_find_kmer_and_metadata_after_build(self, tmp_path):
        index_dir = build_index(tmp_path, OPENST_BARCODES, (2, 27), QUERY, spatial=True)
        index = MalvaIndex(str(index_dir))
        index.open()
        try:
            assert index.n_chunks == 1
            assert index.index_file.endswith("meta.json")
            assert index.n_spatial == len(OPENST_BARCODES)

            index.load_index_to_memory()
            result = index.find_kmer([encode_kmer(QUERY)], count_at_least=1)
            assert isinstance(result, dict)
            assert len(result) >= 1
        finally:
            index.close()

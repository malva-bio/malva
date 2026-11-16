import numpy as np

from malva.cellxmer import (
    _get_csr_arrays,
    _get_csr_arrays_chunk,
    malva_to_cellxmer,
    malva_to_filtered_cellxmer_chunked,
)
from malva.malva_index import MalvaIndex
from tests.index_helpers import OPENST_BARCODES, QUERY, SC_BARCODES, build_index


class TestCsrArrays:
    def test_full_arrays(self, tmp_path):
        index_dir = build_index(tmp_path, OPENST_BARCODES, (2, 27), QUERY, spatial=True)
        index = MalvaIndex(str(index_dir))
        indices, indptr, data, num_cells = _get_csr_arrays(index, verbose=False)

        assert num_cells == len(OPENST_BARCODES)
        assert indptr.dtype == np.int64
        assert len(indptr) >= 1
        assert data.min() == 1

    def test_chunk_is_a_prefix_of_the_whole(self, tmp_path):
        index_dir = build_index(tmp_path, OPENST_BARCODES, (2, 27), QUERY, spatial=True)
        index = MalvaIndex(str(index_dir))
        indices, indptr, data, num_cells, total = _get_csr_arrays_chunk(index, 0, 2, verbose=False)

        assert num_cells == len(OPENST_BARCODES)
        assert total >= len(indices)
        assert indptr.dtype == np.int64

    def test_malva_to_cellxmer_builds_anndata(self, tmp_path):
        import anndata

        index_dir = build_index(tmp_path, OPENST_BARCODES, (2, 27), QUERY, spatial=True)
        index = MalvaIndex(str(index_dir))
        adata = malva_to_cellxmer(index, verbose=False)

        assert isinstance(adata, anndata.AnnData)
        assert adata.n_obs == len(OPENST_BARCODES)

    def test_singlecell_ids_are_zero_based_for_csr(self, tmp_path):
        index_dir = build_index(tmp_path, SC_BARCODES, (0, 16), QUERY, spatial=False)
        index = MalvaIndex(str(index_dir))
        _, _, data, num_cells = _get_csr_arrays(index, verbose=False)

        assert num_cells == len(SC_BARCODES)
        assert data.min() == 0
        assert data.max() == num_cells - 1

    def test_filtered_cellxmer_handles_singlecell_ids(self, tmp_path):
        index_dir = build_index(tmp_path, SC_BARCODES, (0, 16), QUERY, spatial=False)
        index = MalvaIndex(str(index_dir))
        index.open()
        try:
            adata = malva_to_filtered_cellxmer_chunked(
                index,
                count_at_least=0,
                count_at_most=10_000,
                w_size=16,
                num_buckets=32,
                chunk_size=1000,
                verbose=False,
            )
        finally:
            index.close()

        assert adata.n_obs == len(SC_BARCODES)
        assert np.all(np.asarray(adata.X.sum(axis=1)).ravel() > 0)

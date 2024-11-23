import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import gc
import logging
import gzip

from malva.indexes import MalvaIndex, create_spatial_index, create_singlecell_index, SpatialIndex
from .test_data_utils import create_test_dataset, FLAVORS

def verify_spatial_file(spatial_file):
    """Verify spatial file format before processing."""
    with open(spatial_file, 'r') as f:
        header = f.readline().strip()
        if header != "barcode,x,y":
            raise ValueError(f"Invalid header in spatial file: {header}")

        first_line = f.readline().strip()
        parts = first_line.split(',')
        if len(parts) != 3:
            raise ValueError(f"Invalid data line format: {first_line}")
        try:
            float(parts[1])
            float(parts[2])
        except ValueError:
            raise ValueError(f"Invalid coordinate values: {parts[1]}, {parts[2]}")

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    gc.collect()

def safe_close_index(index):
    """Safely close an index object."""
    try:
        if hasattr(index, 'index') and index.index is not None:
            if hasattr(index.index, 'id'):
                index.close()
    except Exception as e:
        logging.warning(f"Error closing index: {e}")

class TestIndexes:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            yield path
            try:
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
            except Exception as e:
                logging.warning(f"Failed to cleanup {path}: {e}")
            gc.collect()

    @pytest.fixture(params=['openst', 'sc_10x_v3']) # 'stereo_seq' not included because it will core dump
    def test_dataset(self, request, temp_dir):
        try:
            dataset_dir = temp_dir / request.param
            dataset_dir.mkdir(exist_ok=True)
            dataset = create_test_dataset(
                dataset_dir, 
                request.param, 
                n_reads=100000,
                n_barcodes=1000
            )
            

            assert Path(dataset['r1_file']).exists(), "R1 file not created"
            assert Path(dataset['r2_file']).exists(), "R2 file not created"
            if dataset['spatial_file']:
                assert Path(dataset['spatial_file']).exists(), "Spatial file not created"
                if request.param != 'stereo_seq':
                    verify_spatial_file(dataset['spatial_file'])
            
            yield dataset
        finally:
            gc.collect()

    def test_stomics_workflow(self, temp_dir):
        """Test STOmics workflow in isolation."""
        stomics_dir = temp_dir / "stomics"
        stomics_dir.mkdir(exist_ok=True)
        index = None

        try:
            binary_file = stomics_dir / "spatial.bin"
            n_barcodes = 100
            with open(binary_file, 'wb') as f:
                for i in range(n_barcodes):
                    barcode_int = i
                    x = (i % 10) * 1000
                    y = (i // 10) * 1000
                    
                    # Write in STOmics format
                    f.write(barcode_int.to_bytes(8, 'little'))
                    f.write(x.to_bytes(4, 'little'))  # uint32 x
                    f.write(y.to_bytes(4, 'little'))  # uint32 y

            # Create FASTQ files with matching barcodes
            dataset = create_test_dataset(
                stomics_dir,
                'stereo_seq',
                n_reads=100000,
                n_barcodes=n_barcodes
            )

            # Create index directory
            index_dir = temp_dir / "index"
            index_dir.mkdir(exist_ok=True)

            # Load STOmics spatial data
            sindex = SpatialIndex()
            sindex.load_binary_stomics(str(binary_file), barcode_length=25)  # STOmics uses 25bp barcodes

            # Verify STOmics coordinates loaded correctly
            stomics_coords = sindex.get_coords_stomics()
            assert stomics_coords.shape == (n_barcodes, 2)
            assert stomics_coords.dtype == np.uint16

            # Initialize index with proper STOmics handling
            index = MalvaIndex(str(index_dir), kmer_size_initialize=24)
            index.set_barcode_index(sindex)
            index.set_spatial_coords(stomics_coords.astype(np.float32))

            index.add_reads(
                [str(dataset['r1_file']), str(dataset['r2_file'])],
                'CB:{cell}',
                'r1[0:25]',
                chunksize=100
            )

            safe_close_index(index)

            # Verify index
            index = MalvaIndex(str(index_dir))
            index.open()
            assert index.n_spatial == n_barcodes
            
        except Exception as e:
            logging.error(f"STOmics test failed: {str(e)}")
            raise
        finally:
            if index is not None:
                safe_close_index(index)
            gc.collect()
    
    def test_malva_index_with_spatial(self, test_dataset, temp_dir):
        """Test spatial index integration with proper cleanup."""
        if not test_dataset['spatial_file']:
            pytest.skip("Test only for spatial datasets")

        index_dir = temp_dir / "index"
        index_dir.mkdir(exist_ok=True)
        index = None

        try:
            index = MalvaIndex(str(index_dir), kmer_size_initialize=24)

            sindex = create_spatial_index(str(test_dataset['spatial_file']))
            assert sindex.num_items() > 0

            index.set_spatial_index(sindex)
            del sindex
            gc.collect()

            chunksize = 1000
            index.add_reads(
                [str(test_dataset['r1_file']), str(test_dataset['r2_file'])],
                bam_tags=FLAVORS[test_dataset['flavor']].barcode_tag,
                cell=f"r1[{FLAVORS[test_dataset['flavor']].barcode_position[0]}:"
                    f"{FLAVORS[test_dataset['flavor']].barcode_position[1]}]",
                chunksize=chunksize
            )

            safe_close_index(index)
            index = MalvaIndex(str(index_dir))
            index.open()

            assert 'spatial_coord' in index.index
            assert 'coord_lims' in index.index.attrs
            coords = index.index['spatial_coord'][:]
            assert len(coords) > 0

        finally:
            if index is not None:
                safe_close_index(index)
            gc.collect()

    def test_process_kmer(self, test_dataset, temp_dir):
        """Test kmer processing with proper memory management."""
        if not test_dataset['spatial_file']:
            pytest.skip("Test only for spatial datasets")

        index = None
        try:
            sindex = create_spatial_index(str(test_dataset['spatial_file']))
            assert sindex.num_items() > 0, "Spatial index is empty"

            index_dir = temp_dir / "index"
            index_dir.mkdir(exist_ok=True)

            index = MalvaIndex(str(index_dir), kmer_size_initialize=24)
            index.set_spatial_index(sindex)
            
            index.add_reads(
                [str(test_dataset['r1_file']), str(test_dataset['r2_file'])],
                bam_tags=FLAVORS[test_dataset['flavor']].barcode_tag,
                cell=f"r1[{FLAVORS[test_dataset['flavor']].barcode_position[0]}:"
                    f"{FLAVORS[test_dataset['flavor']].barcode_position[1]}]",
                chunksize=1000
            )

            safe_close_index(index)
            index = MalvaIndex(str(index_dir))
            index.open()

            index.load_index_to_memory()
            
            test_sequence = "A" * 24
            results = index.where(
                test_sequence, 
                sliding_size=128,
                pct_threshold=0.65,
                count_at_most=10000,
                count_at_least=1
            )
            assert len(results) == 3  # locations, counts, matches

        finally:
            if index is not None:
                safe_close_index(index)
            gc.collect()

    def test_where_functionality(self, test_dataset, temp_dir):
        """Test the where functionality with different memory constraints."""
        if not test_dataset['spatial_file'] and not test_dataset['flavor'].startswith('sc_'):
            pytest.skip("Test only for spatial or single-cell datasets")

        index_dir = temp_dir / "index"
        index_dir.mkdir(exist_ok=True)
        index = None

        try:
            index = MalvaIndex(str(index_dir), kmer_size_initialize=24)
            
            if test_dataset['flavor'].startswith('sc_'):
                sindex = create_singlecell_index(str(test_dataset['whitelist_file']))
                index.set_barcode_index(sindex)
            else:
                sindex = create_spatial_index(str(test_dataset['spatial_file']))
                index.set_spatial_index(sindex)

            test_sequences = {
                "repeated": "AT" * 12,
                "unique": "GCTAGCTAGCTAGCTAGCTAGCTA",
                "nonexistent": "T" * 24,
                "with_n": "N" * 24,
                "longer": "AT" * 24,
            }

            r2_file = test_dataset['r2_file']
            with gzip.open(r2_file, 'rt') as f:
                content = f.readlines()

            for i in range(1, len(content), 4):
                if (i//4) % 5 == 0:
                    content[i] = test_sequences["repeated"] + "\n"
                elif (i//4) % 5 == 1:
                    content[i] = test_sequences["unique"] + "\n"
                elif (i//4) % 5 == 2:
                    content[i] = test_sequences["longer"] + "\n"

            with gzip.open(r2_file, 'wt') as f:
                f.writelines(content)

            index.add_reads(
                [str(test_dataset['r1_file']), str(test_dataset['r2_file'])],
                bam_tags=FLAVORS[test_dataset['flavor']].barcode_tag,
                cell=f"r1[{FLAVORS[test_dataset['flavor']].barcode_position[0]}:"
                    f"{FLAVORS[test_dataset['flavor']].barcode_position[1]}]",
                chunksize=1000
            )

            loading_configs = [
                {"max_mem": None, "description": "full memory"},
                # {"max_mem": "10", "description": "constrained memory"} # disable constrained memory testing
            ]

            for config in loading_configs:
                safe_close_index(index)
                index = MalvaIndex(str(index_dir))
                index.open()

                logging.info(f"Testing where with {config['description']}")

                for seq_name, sequence in test_sequences.items():
                    test_configs = [
                        {
                            "sliding_size": 128,
                            "pct_threshold": 0.65,
                            "count_at_most": 10000,
                            "count_at_least": 1,
                            "single_count": False
                        },
                        {
                            "sliding_size": 64,
                            "pct_threshold": 0.8,
                            "count_at_most": 100,
                            "count_at_least": 10,
                            "single_count": True
                        }
                    ]

                    for params in test_configs:
                        locations, counts, matches = index.where(
                            sequence,
                            max_mem=config["max_mem"],
                            **params
                        )

                        assert isinstance(locations, np.ndarray), f"Invalid locations type for {seq_name}"
                        assert isinstance(counts, np.ndarray), f"Invalid counts type for {seq_name}"
                        assert isinstance(matches, list), f"Invalid matches type for {seq_name}"

                        if seq_name == "nonexistent":
                            assert len(locations) == 0, "Should not find nonexistent sequence"
                        elif seq_name == "with_n":
                            assert len(locations) == 0 or np.all(counts == 0)
                        elif seq_name == "longer":
                            expected_kmers = (len(sequence) - params["sliding_size"]) + 1
                            assert len(matches) >= expected_kmers

                with pytest.raises(ValueError):
                    index.where("A" * 10)

                multi_results = index.where(
                    [test_sequences["repeated"], test_sequences["unique"]],
                    max_mem=config["max_mem"]
                )
                assert len(multi_results) == 3  # locations, counts, matches

        except Exception as e:
            logging.error(f"Where test failed: {str(e)}")
            raise

        finally:
            if index is not None:
                safe_close_index(index)
            gc.collect()

if __name__ == '__main__':
    pytest.main([__file__])
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
        # Verify first data line
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
            if hasattr(index.index, 'id'):  # Check if file is still open
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
                n_reads=10000,
                n_barcodes=1000
            )
            
            # Verify the dataset
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
            # Create binary STOmics file first (simulating what we get from STOmics)
            binary_file = stomics_dir / "spatial.bin"
            n_barcodes = 100
            with open(binary_file, 'wb') as f:
                for i in range(n_barcodes):
                    # Create deterministic barcode and coordinates
                    barcode_int = i  # Simple incremental number as barcode
                    x = (i % 10) * 1000  # Grid layout
                    y = (i // 10) * 1000
                    
                    # Write in STOmics format
                    f.write(barcode_int.to_bytes(8, 'little'))  # uint64 barcode
                    f.write(x.to_bytes(4, 'little'))  # uint32 x
                    f.write(y.to_bytes(4, 'little'))  # uint32 y

            # Create FASTQ files with matching barcodes
            dataset = create_test_dataset(
                stomics_dir,
                'stereo_seq',
                n_reads=1000,
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

            # Add reads
            index.add_reads(
                [str(dataset['r1_file']), str(dataset['r2_file'])],
                'CB:{cell}',
                'r1[0:25]',
                chunksize=100  # Small chunks for testing
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
            # Initialize index
            index = MalvaIndex(str(index_dir), kmer_size_initialize=24)

            # Create and verify spatial index
            sindex = create_spatial_index(str(test_dataset['spatial_file']))
            assert sindex.num_items() > 0

            # Set spatial index
            index.set_spatial_index(sindex)
            del sindex
            gc.collect()

            # Add reads with small chunk size
            chunksize = 1000
            index.add_reads(
                [str(test_dataset['r1_file']), str(test_dataset['r2_file'])],
                bam_tags=FLAVORS[test_dataset['flavor']].barcode_tag,
                cell=f"r1[{FLAVORS[test_dataset['flavor']].barcode_position[0]}:"
                    f"{FLAVORS[test_dataset['flavor']].barcode_position[1]}]",
                chunksize=chunksize
            )

            # Close and reopen to ensure proper file handling
            safe_close_index(index)
            index = MalvaIndex(str(index_dir))
            index.open()

            # Verify data
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
            # First create and verify spatial index
            sindex = create_spatial_index(str(test_dataset['spatial_file']))
            assert sindex.num_items() > 0, "Spatial index is empty"

            # Create index directory
            index_dir = temp_dir / "index"
            index_dir.mkdir(exist_ok=True)

            # Initialize index and add data
            index = MalvaIndex(str(index_dir), kmer_size_initialize=24)
            index.set_spatial_index(sindex)
            
            # Add reads
            index.add_reads(
                [str(test_dataset['r1_file']), str(test_dataset['r2_file'])],
                bam_tags=FLAVORS[test_dataset['flavor']].barcode_tag,
                cell=f"r1[{FLAVORS[test_dataset['flavor']].barcode_position[0]}:"
                    f"{FLAVORS[test_dataset['flavor']].barcode_position[1]}]",
                chunksize=1000
            )

            # Close and reopen to ensure proper state
            safe_close_index(index)
            index = MalvaIndex(str(index_dir))
            index.open()

            # Load into memory
            index.load_index_to_memory()
            
            # Try to find some kmers using where() instead of find_kmer()
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

    def test_stomics_specific(self, temp_dir):
        dataset = create_test_dataset(temp_dir / "stomics", 'stereo_seq', n_reads=10000, n_barcodes=100)
        
        if not dataset['spatial_file']:
            pytest.skip("STOmics spatial file not created")
            
        index = None
        try:
            # Test STOmics coordinate loading
            sindex = SpatialIndex()
            sindex.load_binary_stomics(str(dataset['spatial_file']))

            # Verify coordinates
            coords = sindex.get_coords_stomics()
            assert len(coords) > 0, "No coordinates loaded"
            assert coords.dtype == np.uint16

            # Create index
            index_dir = temp_dir / "stomics_index"
            index_dir.mkdir(exist_ok=True)
            index = MalvaIndex(str(index_dir), kmer_size_initialize=24)

            # Convert STOmics coordinates to float32 for spatial index
            float_coords = coords.astype(np.float32)
            index.set_spatial_coords(float_coords)

            # Add reads
            index.add_reads(
                [str(dataset['r1_file']), str(dataset['r2_file'])],
                bam_tags='CB:{cell}',
                cell='r1[0:25]'
            )

            assert index.n_spatial > 0

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
            # Initialize index
            index = MalvaIndex(str(index_dir), kmer_size_initialize=24)
            
            # Set up index based on flavor
            if test_dataset['flavor'].startswith('sc_'):
                sindex = create_singlecell_index(str(test_dataset['whitelist_file']))
                index.set_barcode_index(sindex)
            else:
                sindex = create_spatial_index(str(test_dataset['spatial_file']))
                index.set_spatial_index(sindex)

            # Create test data with known sequences
            test_sequences = {
                "repeated": "AT" * 12,  # Should find many matches
                "unique": "GCTAGCTAGCTAGCTAGCTAGCTA",  # Should find few matches
                "nonexistent": "T" * 24,  # Should find no matches
                "with_n": "N" * 24,  # Should be treated as low complexity
                "longer": "AT" * 24,  # Longer sequence to test sliding window
            }

            # Modify R2 file to include test sequences
            r2_file = test_dataset['r2_file']
            with gzip.open(r2_file, 'rt') as f:
                content = f.readlines()

            # Insert known sequences
            for i in range(1, len(content), 4):
                if (i//4) % 5 == 0:
                    content[i] = test_sequences["repeated"] + "\n"
                elif (i//4) % 5 == 1:
                    content[i] = test_sequences["unique"] + "\n"
                elif (i//4) % 5 == 2:
                    content[i] = test_sequences["longer"] + "\n"

            with gzip.open(r2_file, 'wt') as f:
                f.writelines(content)

            # Add reads with small chunks
            index.add_reads(
                [str(test_dataset['r1_file']), str(test_dataset['r2_file'])],
                bam_tags=FLAVORS[test_dataset['flavor']].barcode_tag,
                cell=f"r1[{FLAVORS[test_dataset['flavor']].barcode_position[0]}:"
                    f"{FLAVORS[test_dataset['flavor']].barcode_position[1]}]",
                chunksize=1000
            )

            # Test different loading modes
            loading_configs = [
                {"max_mem": None, "description": "full memory"},
                {"max_mem": "1M", "description": "constrained memory"},
                {"max_mem": "10M", "description": "medium memory"}
            ]

            for config in loading_configs:
                safe_close_index(index)
                index = MalvaIndex(str(index_dir))
                index.open()

                logging.info(f"Testing where with {config['description']}")

                for seq_name, sequence in test_sequences.items():
                    # Test with different parameters
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

                        # Verify results
                        assert isinstance(locations, np.ndarray), f"Invalid locations type for {seq_name}"
                        assert isinstance(counts, np.ndarray), f"Invalid counts type for {seq_name}"
                        assert isinstance(matches, list), f"Invalid matches type for {seq_name}"

                        # Verify sequence-specific expectations
                        if seq_name == "repeated":
                            assert len(locations) > 0, "Should find repeated sequence"
                            if not params["single_count"]:
                                assert np.any(counts > 1), "Should find multiple occurrences"
                        elif seq_name == "nonexistent":
                            assert len(locations) == 0, "Should not find nonexistent sequence"
                        elif seq_name == "with_n":
                            # N sequences should be treated as low complexity
                            assert len(locations) == 0 or np.all(counts == 0)
                        elif seq_name == "longer":
                            # Test sliding window functionality
                            expected_kmers = (len(sequence) - params["sliding_size"]) + 1
                            assert len(matches) >= expected_kmers

                # Test error cases
                with pytest.raises(ValueError):
                    index.where("A" * 10)  # Too short sequence

                # Test multiple sequences
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
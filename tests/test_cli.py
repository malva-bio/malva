import pytest
import argparse
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path
import gzip
import sys

from malva.cli import (
    get_index_parser, get_show_parser, get_quant_parser, 
    get_cellxmer_parser, get_serve_parser, get_combine_parser,
    cmdline_main,
)

class TestCLI:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_fastq_files(self, temp_dir):
        r1_file = temp_dir / "r1.fastq.gz"
        r2_file = temp_dir / "r2.fastq.gz"
        
        content = "@read1\nACGT\n+\nIIII\n"
        for file in [r1_file, r2_file]:
            with gzip.open(file, 'wt') as f:
                f.write(content)
        
        return r1_file, r2_file

    @pytest.fixture
    def mock_spatial_file(self, temp_dir):
        spatial_file = temp_dir / "spatial.csv"
        content = "barcode,x,y\nACGT,1,1\nTGCA,2,2\n"
        
        with open(spatial_file, 'w') as f:
            f.write(content)
        
        return spatial_file

    def test_version_command(self, monkeypatch):
        with patch.object(sys, 'argv', ['malva', '--version']):
            with patch('importlib.metadata.version', return_value='0.2.0'):
                cmdline_main()

    def test_command_error_handling(self, temp_dir):
        test_fastq = temp_dir / "test.fastq"
        test_fastq.touch()
        test_spatial = temp_dir / "test.csv"
        test_spatial.touch()

        # Test missing required arguments
        with patch.object(sys, 'argv', ['malva', 'index']):
            with pytest.raises(SystemExit):
                cmdline_main()
        
        # Test invalid flavor with existing files
        with patch.object(sys, 'argv', [
            'malva', 'index',
            '--reads-in', str(test_fastq), str(test_fastq),
            '--spatial-bc-in', str(test_spatial),
            '--index-out', str(temp_dir),
            '--flavor', 'invalid'
        ]):
            with pytest.raises(Exception):
                cmdline_main()

    def test_command_function_calls(self, temp_dir):
        def mock_run(*args, **kwargs):
            return True

        with patch('malva.index._run_index', mock_run):

            # Test index command
            with patch.object(sys, 'argv', [
                'malva', 'index',
                '--reads-in', 'test1.fq', 'test2.fq',
                '--spatial-bc-in', 'spatial.csv',
                '--index-out', str(temp_dir)
            ]):
                cmdline_main()

    def test_index_parser(self):
        parser = get_index_parser()
        
        # Test required arguments
        with pytest.raises(SystemExit):
            parser.parse_args([])
        
        # Test all arguments
        args = parser.parse_args([
            '--reads-in', 'r1.fq', 'r2.fq',
            '--spatial-bc-in', 'spatial.csv',
            '--index-out', 'index_dir',
            '--flavor', 'visium',
            '--kmer-length', '24',
            '--chunksize', '1000000',
            '--overlapping',
            '--merge-chunks',
            '--threads', '4'
        ])
        
        assert args.reads_in == ['r1.fq', 'r2.fq']
        assert args.spatial_bc_in == 'spatial.csv'
        assert args.index_out == 'index_dir'
        assert args.flavor == 'visium'
        assert args.kmer_length == 24
        assert args.chunksize == 1000000
        assert args.overlapping is True
        assert args.merge_chunks is True
        assert args.threads == 4

    def test_show_parser(self):
        parser = get_show_parser()
        
        # Test required arguments
        with pytest.raises(SystemExit):
            parser.parse_args([])
        
        # Test all arguments
        args = parser.parse_args([
            '--index-in', 'index_dir',
            '--query', 'query.fa',
            '--image-out', 'images',
            '--multichannel',
            '--save-npy',
            '--scalebar',
            '--render-scale', '2.0',
            '--render-smoothing', '1.5'
        ])
        
        assert args.index_in == 'index_dir'
        assert args.query == 'query.fa'
        assert args.image_out == 'images'
        assert args.multichannel is True
        assert args.save_npy is True
        assert args.scalebar is True
        assert args.render_scale == 2.0
        assert args.render_smoothing == 1.5

    def test_quant_parser(self):
        parser = get_quant_parser()
        
        # Test all arguments
        args = parser.parse_args([
            '--index-in', 'index_dir',
            '--reference', 'human_utr',
            '--background-model', 'bg.model',
            '--folder-out', 'quant_out',
            '--h5ad',
            '--bin-size', '10',
            '--sliding-size', '128',
            '--pct-threshold', '0.65',
            '--kmer-min', '10',
            '--kmer-max', '10000',
            '--single-count'
        ])
        
        assert args.index_in == 'index_dir'
        assert args.reference == 'human_utr'
        assert args.background_model == 'bg.model'
        assert args.folder_out == 'quant_out'
        assert args.h5ad is True
        assert args.bin_size == 10
        assert args.sliding_size == 128
        assert args.pct_threshold == 0.65
        assert args.kmer_min == 10
        assert args.kmer_max == 10000
        assert args.single_count is True

    def test_cellxmer_parser(self):
        parser = get_cellxmer_parser()
        
        args = parser.parse_args([
            '--index-in', 'index_dir',
            '--h5ad-out', 'out.h5ad',
            '--kmer-min', '10',
            '--kmer-max', '10000',
            '--bin-size', '10'
        ])
        
        assert args.index_in == 'index_dir'
        assert args.h5ad_out == 'out.h5ad'
        assert args.kmer_min == 10
        assert args.kmer_max == 10000
        assert args.bin_size == 10

    def test_serve_parser(self):
        parser = get_serve_parser()
        
        args = parser.parse_args([
            '--index-in', 'index_dir',
            '--uuid', '12345',
            '--port', '8888',
            '--address', '127.0.0.1',
            '--max-mem', '4G',
            '--max-len', '1000',
            '--rescale-coords', '2.0'
        ])
        
        assert args.index_in == 'index_dir'
        assert args.uuid == '12345'
        assert args.port == 8888
        assert args.address == '127.0.0.1'
        assert args.max_mem == '4G'
        assert args.max_len == 1000
        assert args.rescale_coords == 2.0

    def test_combine_parser(self):
        parser = get_combine_parser()
        
        args = parser.parse_args([
            '--index-in', 'index_dir',
            '--merge-chunks'
        ])
        
        assert args.index_in == 'index_dir'
        assert args.merge_chunks is True

    @patch('malva.cli.cmd_run_index')
    def test_index_command(self, mock_run_index, temp_dir, mock_fastq_files, mock_spatial_file):
        r1_file, r2_file = mock_fastq_files
        index_dir = temp_dir / "index"
        os.makedirs(index_dir)
        
        with patch('sys.argv', [
            'malva', 'index',
            '--reads-in', str(r1_file), str(r2_file),
            '--spatial-bc-in', str(mock_spatial_file),
            '--index-out', str(index_dir),
            '--flavor', 'visium'
        ]):
            cmdline_main()
        
        mock_run_index.assert_called_once()

    def test_invalid_command(self):
        with patch('sys.argv', ['malva', 'invalid']):
            with pytest.raises(SystemExit):
                cmdline_main()

if __name__ == '__main__':
    pytest.main([__file__])
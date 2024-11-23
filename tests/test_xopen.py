import pytest
import os
import tempfile
import gzip
from pathlib import Path
from io import StringIO, BytesIO
import subprocess

from malva.xopen import xopen

def has_bgzip():
    """Check if bgzip is available in the system."""
    try:
        subprocess.run(['bgzip', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

class TestXopen:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_file_like_objects(self):
        content = b"File-like\nObject\n"
        
        # BytesIO
        bytes_io = BytesIO(content)
        with xopen(bytes_io, 'rb') as f:
            assert f.read() == content

    @pytest.mark.skipif(not has_bgzip(), reason="bgzip not available")
    def test_bgzip_compression(self, temp_dir):
        input_file = temp_dir / 'test.fastq'
        output_file = temp_dir / 'test.fastq.gz'
        content = "@read1\nACGT\n+\nIIII\n"
        
        with open(input_file, 'w') as f:
            f.write(content)
        
        subprocess.run(['bgzip', '-c', str(input_file)], 
                      stdout=open(str(output_file), 'wb'),
                      check=True)
        
        with xopen(output_file, 'r') as f:
            assert f.read() == content

    def test_pipe_handling(self):
        # Test reading from stdin
        with pytest.raises(ValueError):
            with xopen('-', 'r') as f:
                f.read()

    def test_read_text_file(self, temp_dir):
        text_file = temp_dir / 'test.txt'
        content = "Hello\nWorld\n"
        
        with open(text_file, 'w') as f:
            f.write(content)
        
        with xopen(text_file, 'r') as f:
            assert f.read() == content

    def test_read_gzip_file(self, temp_dir):
        gz_file = temp_dir / 'test.gz'
        content = "Compressed\nContent\n"
        
        with gzip.open(gz_file, 'wt') as f:
            f.write(content)
        
        with xopen(gz_file, 'r') as f:
            assert f.read() == content

    def test_append_modes(self, temp_dir):
        text_file = temp_dir / 'append.txt'
        
        with xopen(text_file, 'w') as f:
            f.write("First\n")
        
        with xopen(text_file, 'a') as f:
            f.write("Second\n")
        
        with open(text_file, 'r') as f:
            assert f.read() == "First\nSecond\n"

    def test_binary_modes(self, temp_dir):
        binary_file = temp_dir / 'binary.dat'
        content = b"Binary\nContent\n"
        
        with xopen(binary_file, 'wb') as f:
            f.write(content)
        
        with xopen(binary_file, 'rb') as f:
            assert f.read() == content

    def test_error_handling(self, temp_dir):
        # Test invalid mode
        with pytest.raises(ValueError):
            xopen(temp_dir / 'test.txt', 'x')
        
        # Test invalid compression format
        with pytest.raises(ValueError):
            xopen(temp_dir / 'test.txt', 'r', format='invalid')
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            xopen(temp_dir / 'nonexistent.txt', 'r')

if __name__ == '__main__':
    pytest.main([__file__])
import pytest
import os
import tempfile
import gzip
import random
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys

# Test Data Generation Helper Functions
def generate_random_dna(length):
    """Generate random DNA sequence."""
    return ''.join(random.choice('ACGT') for _ in range(length))

def generate_fastq_record(sequence, name="@read"):
    """Generate a FASTQ record."""
    quality = ''.join(['I' for _ in sequence])
    return f"{name}\n{sequence}\n+\n{quality}\n"

def write_fastq_file(filename, records, compress=True):
    """Write FASTQ records to a file."""
    content = ''.join(records)
    if compress:
        with gzip.open(filename, 'wt') as f:
            f.write(content)
    else:
        with open(filename, 'w') as f:
            f.write(content)

def generate_spatial_coordinates(n_spots, grid_size=100):
    """Generate spatial coordinates for spots."""
    coords = []
    for i in range(n_spots):
        x = random.uniform(0, grid_size)
        y = random.uniform(0, grid_size)
        barcode = ''.join(random.choice('ACGT') for _ in range(16))
        coords.append(f"{barcode},{x},{y}\n")
    return coords

@contextmanager
def create_test_files():
    """Context manager to create and cleanup test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create spatial coordinates file
        spatial_file = os.path.join(tmpdir, 'spatial.csv')
        with open(spatial_file, 'w') as f:
            f.write("barcode,x,y\n")
            coords = generate_spatial_coordinates(100)
            f.writelines(coords)
        
        # Create FASTQ files
        r1_file = os.path.join(tmpdir, 'R1.fastq.gz')
        r2_file = os.path.join(tmpdir, 'R2.fastq.gz')
        
        # Generate paired reads
        r1_records = []
        r2_records = []
        for i in range(1000):
            barcode = generate_random_dna(16)
            umi = generate_random_dna(10)
            r1_seq = barcode + umi
            r2_seq = generate_random_dna(100)
            
            r1_records.append(generate_fastq_record(r1_seq, f"@read{i}"))
            r2_records.append(generate_fastq_record(r2_seq, f"@read{i}"))
        
        write_fastq_file(r1_file, r1_records)
        write_fastq_file(r2_file, r2_records)
        
        yield {
            'spatial_file': spatial_file,
            'r1_file': r1_file,
            'r2_file': r2_file,
            'tmpdir': tmpdir
        }

def pytest_sessionstart(session):
    """Build Cython modules before running tests."""
    print("Building Cython modules...")
    
    # Get the project root directory (where setup.py is located)
    root_dir = Path(__file__).parent.parent
    
    # Run pip install in development mode
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(root_dir)])

@pytest.fixture(scope="session")
def temp_build_dir(tmp_path_factory):
    """Create a temporary directory for building test files."""
    return tmp_path_factory.mktemp("build")

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent
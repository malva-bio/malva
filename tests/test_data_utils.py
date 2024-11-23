import numpy as np
import gzip
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
from pathlib import Path
import struct

@dataclass
class FlavorConfig:
    name: str
    barcode_position: Tuple[int, int]
    barcode_tag: str
    is_spatial: bool = True

FLAVORS = {
    'openst': FlavorConfig('openst', (2, 27), 'CB:{cell}'),
    'seq_scope_v1': FlavorConfig('seq_scope_v1', (0, 20), 'CB:{cell}'),
    'stereo_seq': FlavorConfig('stereo_seq', (0, 25), 'CB:{cell}'),
    'visium': FlavorConfig('visium', (0, 16), 'CR:{cell}'),
    'slide_seq': FlavorConfig('slide_seq', (0, 14), 'XC:{cell}'),
    'sc_10x_v3': FlavorConfig('sc_10x_v3', (0, 16), 'CB:{cell}', False),
    'sc_10x_v1': FlavorConfig('sc_10x_v1', (0, 14), 'CB:{cell}', False)
}

def encode_barcode_stomics(barcode: str) -> int:
    """Encode barcode for STOmics format."""
    # STOmics uses a specific encoding for barcodes
    encoded = 0
    for i, base in enumerate(barcode):
        value = {'A': 0, 'C': 1, 'G': 2, 'T': 3}[base]
        encoded |= (value & 0b11) << (2 * i)
    return encoded

def generate_random_sequence(length: int, valid_bases: str = 'ACGT') -> str:
    """Generate random DNA sequence."""
    return ''.join(random.choices(valid_bases, k=length))

def create_spatial_barcode_list(n_barcodes: int, barcode_length: int) -> List[str]:
    """Create list of spatial barcodes."""
    # Ensure no duplicate barcodes
    barcodes = set()
    while len(barcodes) < n_barcodes:
        barcodes.add(generate_random_sequence(barcode_length))
    return list(barcodes)

def generate_fastq_files(
    output_dir: Path,
    flavor: str,
    n_reads: int = 10000,
    spatial_barcodes: Optional[List[str]] = None,
    r2_length: int = 90
) -> Tuple[Path, Path]:
    """Generate paired FASTQ files with matching barcodes."""
    flavor_config = FLAVORS[flavor]
    r1_file = output_dir / f"{flavor}_R1.fastq.gz"
    r2_file = output_dir / f"{flavor}_R2.fastq.gz"
    
    barcode_start, barcode_end = flavor_config.barcode_position
    barcode_length = barcode_end - barcode_start
    r1_length = barcode_end + 10  # Add some padding after barcode
    
    with gzip.open(r1_file, 'wt') as f1, gzip.open(r2_file, 'wt') as f2:
        for i in range(n_reads):
            # Generate R1 read with proper barcode placement
            if spatial_barcodes and i % 5 < 2:  # 40% of reads have valid barcodes
                barcode = random.choice(spatial_barcodes)
                r1_seq = ('N' * barcode_start + 
                         barcode + 
                         generate_random_sequence(r1_length - barcode_length - barcode_start))
            else:
                r1_seq = generate_random_sequence(r1_length)
            
            # Generate R2 read with proper length
            r2_seq = generate_random_sequence(r2_length)
            
            # Write R1
            r1_qual = 'I' * len(r1_seq)  # High quality scores
            f1.write(f"@read{i}\n{r1_seq}\n+\n{r1_qual}\n")
            
            # Write R2
            r2_qual = 'I' * len(r2_seq)
            f2.write(f"@read{i}\n{r2_seq}\n+\n{r2_qual}\n")
    
    return r1_file, r2_file

def generate_stomics_coords(output_dir: Path, barcodes: List[str]) -> Path:
    """Generate STOmics-specific coordinate file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coords_file = output_dir / "spatial.bin"
    
    with open(coords_file, 'wb') as f:
        # Write STOmics format
        for barcode in barcodes:
            # Ensure 25bp barcode
            padded_barcode = barcode.ljust(25, 'A')[:25]
            
            # Encode barcode as per STOmics spec
            barcode_int = encode_barcode_stomics(padded_barcode)
            
            # Generate coordinates (uint32)
            x = random.randint(0, 65535)
            y = random.randint(0, 65535)
            
            # Write binary data in correct format
            f.write(struct.pack('<Q', barcode_int))  # uint64 barcode, little-endian
            f.write(struct.pack('<II', x, y))  # uint32 coordinates, little-endian
    
    return coords_file

def generate_spatial_coords(
    output_dir: Path,
    barcodes: List[str],
    flavor: str,
    grid_size: float = 100.0
) -> Optional[Path]:
    """Generate spatial coordinate file based on flavor."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    flavor_config = FLAVORS[flavor]
    if not flavor_config.is_spatial:
        return None
    
    if flavor == 'stereo_seq':
        return generate_stomics_coords(output_dir, barcodes)
    
    # Generate grid-like coordinates for regular spatial data
    coords_file = output_dir / f"{flavor}_spatial.csv"
    with open(coords_file, 'w') as f:
        f.write("barcode,x,y\n")  # Header required
        
        n_side = int(np.ceil(np.sqrt(len(barcodes))))
        step = grid_size / n_side
        
        for i, barcode in enumerate(barcodes):
            row = i // n_side
            col = i % n_side
            # Generate deterministic but slightly noisy coordinates
            x = col * step + random.uniform(0, step/3)
            y = row * step + random.uniform(0, step/3)
            f.write(f"{barcode},{x:.6f},{y:.6f}\n")
    
    return coords_file

def generate_whitelist(output_dir: Path, barcodes: List[str]) -> Path:
    """Generate whitelist file for single-cell data."""
    whitelist_file = output_dir / "whitelist.txt"
    with open(whitelist_file, 'w') as f:
        f.write('\n'.join(barcodes) + '\n')  # One barcode per line
    return whitelist_file

def create_test_dataset_stomics(
    output_dir: Path,
    n_reads: int = 1000,
    n_barcodes: int = 100
) -> dict:
    """Create STOmics-specific test dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create deterministic barcodes for STOmics
    barcodes = []
    for i in range(n_barcodes):
        # Convert index to 25bp barcode
        barcode = format(i, '025b').replace('0', 'A').replace('1', 'T')
        barcodes.append(barcode)
    
    # Generate FASTQ files ensuring some barcodes match
    r1_file = output_dir / "stereo_seq_R1.fastq.gz"
    r2_file = output_dir / "stereo_seq_R2.fastq.gz"
    
    with gzip.open(r1_file, 'wt') as f1, gzip.open(r2_file, 'wt') as f2:
        for i in range(n_reads):
            # Use valid barcode for 30% of reads
            if i % 10 < 3:
                barcode = random.choice(barcodes)
                r1_seq = barcode + generate_random_sequence(25)  # Add random sequence after barcode
            else:
                r1_seq = generate_random_sequence(50)  # Random sequence for non-matching reads
            
            r2_seq = generate_random_sequence(90)
            
            # Write reads
            f1.write(f"@read{i}\n{r1_seq}\n+\n{'I' * len(r1_seq)}\n")
            f2.write(f"@read{i}\n{r2_seq}\n+\n{'I' * len(r2_seq)}\n")
    
    return {
        'r1_file': r1_file,
        'r2_file': r2_file,
        'barcodes': barcodes,
        'flavor': 'stereo_seq'
    }

def create_test_dataset(
    output_dir: Path,
    flavor: str,
    n_reads: int = 10000,
    n_barcodes: int = 100
) -> dict:
    """Create test dataset for given flavor."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if flavor == 'stereo_seq':
        return create_test_dataset_stomics(output_dir, n_reads, n_barcodes)
        
    flavor_config = FLAVORS[flavor]
    barcode_length = flavor_config.barcode_position[1] - flavor_config.barcode_position[0]
    
    # Create barcodes
    barcodes = create_spatial_barcode_list(n_barcodes, barcode_length)
    
    # Generate coordinate files first
    if flavor_config.is_spatial:
        coords_file = generate_spatial_coords(output_dir, barcodes, flavor)
        whitelist_file = None
    else:
        coords_file = None
        whitelist_file = generate_whitelist(output_dir, barcodes)
    
    # Generate FASTQ files
    r1_file, r2_file = generate_fastq_files(
        output_dir=output_dir,
        flavor=flavor,
        n_reads=n_reads,
        spatial_barcodes=barcodes,
        r2_length=90
    )
    
    return {
        'r1_file': r1_file,
        'r2_file': r2_file,
        'spatial_file': coords_file,
        'whitelist_file': whitelist_file,
        'barcodes': barcodes,
        'flavor': flavor
    }
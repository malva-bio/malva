#!/usr/bin/env python3
# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.


import argparse
import logging
import multiprocessing
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import dnaio
from tqdm import tqdm

# Constants
CHUNK_SIZE = 10_000_000
MAX_FILES = 50_000
MEM_PER_CORE = 4  # in gigabytes

log_lock = threading.Lock()

def run_command(cmd: List[str], description: str) -> Tuple[int, str, str]:
    """Run a command and return its exit code, stdout, and stderr."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

def log_output(description: str, returncode: int, stdout: str, stderr: str):
    """Log the output of a command in a synchronized manner."""
    with log_lock:
        logging.debug(f"\n{'=' * 40}\n{description}\n{'=' * 40}")
        if stdout:
            logging.debug("STDOUT:\n%s", stdout)
        if stderr and returncode:
            logging.warning("STDERR:\n%s", stderr)
        logging.debug("Return code: %d\n", returncode)

def extract_barcodes_from_chunk(fastq_file, start_offset, end_offset, output_file, barcode_slice, verbose=False):
    """Extract barcodes from a chunk of an uncompressed FASTQ file."""
    progress_interval = 1000000  # Report progress every 1M barcodes
    barcode_count = 0
    chunk_id = f"{os.path.basename(fastq_file)}:{start_offset}"
    
    with open(fastq_file, 'r') as f:
        f.seek(start_offset)
        
        # Skip to the beginning of a new record if not starting from beginning
        if start_offset > 0:
            line = f.readline()
            while not line.startswith('@'):
                line = f.readline()
        
        with open(output_file, 'w') as out_f:
            line_count = 0
            current_position = f.tell()
            
            while current_position < end_offset if end_offset else True:
                line = f.readline()
                if not line:  # End of file
                    break
                
                line_count += 1
                
                # Every 4th line starting from 2 is the sequence line in FASTQ
                if line_count % 4 == 2:
                    # Extract the barcode using the slice
                    barcode = line.strip()[barcode_slice]
                    if barcode:
                        out_f.write(f"{barcode}\n")
                        barcode_count += 1
                        
                        # Log progress at intervals
                        if verbose and barcode_count % progress_interval == 0:
                            with log_lock:
                                logging.debug(f"Chunk {chunk_id}: Processed {barcode_count:,} barcodes")
                
                current_position = f.tell()
    
    # Final count
    if verbose:
        with log_lock:
            logging.debug(f"Chunk {chunk_id}: Completed with {barcode_count:,} barcodes extracted")

def extract_barcodes_with_dnaio(fastq_file, chunk_dir, num_chunks, barcode_slice, verbose=False):
    """Extract barcodes from a FASTQ file using dnaio, directly into multiple chunk files."""
    progress_interval = 1000000  # Report progress every 1M barcodes
    barcode_count = 0
    total_barcodes = 0
    
    # Create chunk file handles
    chunk_files = []
    chunk_handles = []
    
    for i in range(num_chunks):
        chunk_file = os.path.join(chunk_dir, f"split_chunk_{i}.txt")
        chunk_files.append(chunk_file)
        chunk_handles.append(open(chunk_file, 'w'))
    
    try:
        # Open FASTQ file with dnaio - handles both gzipped and uncompressed files
        with dnaio.open(fastq_file, mode='r') as reader:
            for record in reader:
                # Extract the barcode using the slice
                barcode = record.sequence[barcode_slice]
                if barcode:
                    # Distribute barcodes across chunk files using hash
                    chunk_idx = hash(barcode) % num_chunks
                    chunk_handles[chunk_idx].write(f"{barcode}\n")
                    barcode_count += 1
                    total_barcodes += 1
                    
                    # Log progress at intervals
                    if verbose and barcode_count >= progress_interval:
                        with log_lock:
                            logging.debug(f"File {os.path.basename(fastq_file)}: Processed {total_barcodes:,} barcodes")
                        barcode_count = 0
    
    finally:
        # Close all chunk file handles
        for handle in chunk_handles:
            handle.close()
    
    # Final count
    if verbose:
        with log_lock:
            logging.debug(f"Completed extraction from {os.path.basename(fastq_file)} with {total_barcodes:,} barcodes extracted into {num_chunks} chunks")
    
    return chunk_files

def calculate_offsets(file_path, num_processes):
    """Calculate file offsets for parallel processing of uncompressed files."""
    if file_path.endswith('.gz'):
        return [0]  # We can only process with dnaio
    
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_processes
    offsets = []

    with open(file_path, 'rb') as f:
        offset = 0
        for _ in range(num_processes):
            offsets.append(offset)
            f.seek(chunk_size, 1)
            # Read until we get to a new record (4 lines in FASTQ)
            for _ in range(4):
                f.readline()
            offset = f.tell()
            if offset >= file_size:
                break

    return offsets

def sort_chunk_file(input_file, output_file, temp_dir):
    """Sort a chunk file and remove duplicates."""
    cmd = f"LC_ALL=C sort -u -T {temp_dir} {input_file} > {output_file}"
    subprocess.run(["bash", "-c", cmd], check=True)
    return output_file

def merge_sorted_files(sorted_files, output_file, temp_dir, num_threads):
    """Merge multiple sorted files into one, removing duplicates."""
    cmd = f"LC_ALL=C sort --parallel={num_threads} -m -u -T {temp_dir} {' '.join(sorted_files)} > {output_file}"
    subprocess.run(["bash", "-c", cmd], check=True)
    return output_file

def process_fastq(fastq_file, barcode_slice, temp_dir, num_processes, verbose=False):
    """Process a single FASTQ file to extract barcodes."""
    logging.info(f"Processing {fastq_file}")
    
    # Create temporary directory for this file
    file_temp_dir = os.path.join(temp_dir, os.path.basename(fastq_file))
    os.makedirs(file_temp_dir, exist_ok=True)
    
    try:
        # Create chunks directory
        chunks_dir = os.path.join(file_temp_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # For gzipped files or if dnaio should be used for all files
        if fastq_file.endswith('.gz'):
            logging.info(f"Using dnaio to extract barcodes from {fastq_file} into {num_processes} chunks")
            chunk_files = extract_barcodes_with_dnaio(
                fastq_file,
                chunks_dir,
                num_processes,
                barcode_slice,
                verbose
            )
        else:
            # For uncompressed files, we can still use parallel processing with offset calculation
            offsets = calculate_offsets(fastq_file, num_processes)
            num_processes_actual = len(offsets)
            offsets.append(None)  # Add None as end marker for the last chunk
            
            with ProcessPoolExecutor(max_workers=num_processes_actual) as executor:
                futures = []
                
                for i in range(len(offsets) - 1):
                    chunk_out = os.path.join(chunks_dir, f"split_chunk_{i}.txt")
                    futures.append(
                        executor.submit(
                            extract_barcodes_from_chunk,
                            fastq_file,
                            offsets[i],
                            offsets[i+1],
                            chunk_out,
                            barcode_slice,
                            verbose
                        )
                    )
                
                for future in tqdm(futures, desc=f"Extracting barcodes from {os.path.basename(fastq_file)}"):
                    future.result()
                    
            # Get all chunk files
            chunk_files = [os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.startswith("split_chunk_")]
        
        # Sort chunks in parallel
        sorted_dir = os.path.join(file_temp_dir, "sorted")
        os.makedirs(sorted_dir, exist_ok=True)
        
        sorted_files = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            
            for i, chunk_file in enumerate(chunk_files):
                sorted_file = os.path.join(sorted_dir, f"sorted_{i}.txt")
                futures.append(
                    executor.submit(
                        sort_chunk_file,
                        chunk_file,
                        sorted_file,
                        sorted_dir
                    )
                )
            
            for future in tqdm(futures, desc=f"Sorting barcode chunks for {os.path.basename(fastq_file)}"):
                sorted_files.append(future.result())
        
        # Final merge of sorted files
        merged_file = os.path.join(file_temp_dir, "merged.txt")
        merge_sorted_files(sorted_files, merged_file, file_temp_dir, num_processes)
        
        return merged_file
        
    except Exception as e:
        logging.error(f"Error processing {fastq_file}: {str(e)}")
        raise

def extract_barcodes(fastq_files, barcode_slice, output_file, temp_dir, num_processes, verbose=False):
    """Extract unique barcodes from multiple FASTQ files."""
    # Process each FASTQ file in sequence
    file_results = []
    for fastq_file in fastq_files:
        file_results.append(process_fastq(fastq_file, barcode_slice, temp_dir, num_processes, verbose))
    
    # Create final temporary directory for merging all files
    final_temp_dir = os.path.join(temp_dir, "final_merge")
    os.makedirs(final_temp_dir, exist_ok=True)
    
    # Final merge of all processed files
    logging.info("Performing final merge of all files")
    merge_sorted_files(file_results, output_file, final_temp_dir, num_processes)
    
    # Count number of unique barcodes
    cmd = f"wc -l {output_file}"
    returncode, stdout, stderr = run_command(["bash", "-c", cmd], "Count barcodes")
    if returncode == 0:
        barcode_count = stdout.strip().split()[0]
        logging.info(f"Total unique barcodes: {barcode_count}")
    
    # Cleanup temporary directories
    if os.path.exists(temp_dir) and not os.path.samefile(temp_dir, os.path.dirname(output_file)):
        logging.info("Cleaning up temporary files")
        shutil.rmtree(temp_dir)

def parse_slice(slice_str):
    """Parse a Python slice notation string into a slice object."""
    parts = slice_str.split(':')
    if len(parts) == 1:
        return slice(int(parts[0]), int(parts[0]) + 1)
    elif len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        return slice(start, stop)
    elif len(parts) == 3:
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        step = int(parts[2]) if parts[2] else None
        return slice(start, stop, step)
    else:
        raise ValueError("Invalid slice format. Use Python slice notation (e.g., '0:16').")

def run_barcode_extraction(args):
    """Main function to run the barcode extraction process."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Validate input files
    valid_fastq_files = []
    for fastq_file in args.fastq_in:
        if not os.path.exists(fastq_file):
            logging.error(f"Input file does not exist: {fastq_file}")
            continue
        valid_fastq_files.append(fastq_file)
    
    if not valid_fastq_files:
        logging.error("No valid input FASTQ files provided.")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.barcodes_out)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")
    
    # Set up temporary directory
    if args.tmp_dir:
        temp_dir = args.tmp_dir
    else:
        temp_dir = tempfile.mkdtemp(prefix="barcode_extractor_")
    
    os.makedirs(temp_dir, exist_ok=True)
    logging.info(f"Using temporary directory: {temp_dir}")
    
    # Set file limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(MAX_FILES, hard), hard))
    
    # Parse barcode slice
    try:
        barcode_slice = parse_slice(args.barcode_slice)
    except ValueError as e:
        logging.error(str(e))
        return 1
    
    # Set number of processes
    num_processes = args.parallel_processes
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    logging.info(f"Using {num_processes} parallel processes")
    logging.info(f"Extracting barcodes using slice {args.barcode_slice}")
    
    try:
        # Extract barcodes
        extract_barcodes(valid_fastq_files, barcode_slice, args.barcodes_out, temp_dir, num_processes, args.verbose)
        logging.info(f"Barcode extraction completed successfully. Output: {args.barcodes_out}")
        return 0
    except Exception as e:
        logging.error(f"Error during barcode extraction: {str(e)}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Extract and sort unique barcodes from FASTQ files using dnaio.")
    parser.add_argument(
        "--fastq-in", 
        nargs="+", 
        required=True,
        help="Input FASTQ file(s)"
    )
    parser.add_argument(
        "--barcodes-out", 
        required=True,
        help="Output file to save unique barcodes"
    )
    parser.add_argument(
        "--barcode-slice", 
        default="0:16", 
        help="Slice position to extract barcode (Python slice notation, e.g., '0:16')"
    )
    parser.add_argument(
        "--parallel-processes", 
        type=int, 
        default=None,
        help="Number of parallel processes to use (default: number of CPU cores)"
    )
    parser.add_argument(
        "--tmp-dir", 
        default=None,
        help="Temporary directory for intermediate files"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    sys.exit(run_barcode_extraction(args))

if __name__ == "__main__":
    main()
import dnaio
import sys
import time
import json
import requests
import logging
from typing import List, Dict, Any, Optional

def read_sequences_file(file_path: str) -> List[str]:
    """Read sequences from a file (supports FASTA format)
    
    Uses dnaio for efficient and robust parsing of FASTA/FASTQ files.
    
    Args:
        file_path: Path to the sequence file (FASTA or FASTQ)
        
    Returns:
        List of sequences without headers
    """
    sequences = []

    with dnaio.open(file_path) as reader:
        for record in reader:
            sequences.append(record.sequence)
    
    return sequences


def check_server_connection(server_url: str) -> bool:
    """Check if the Malva search server is running"""
    try:
        response = requests.get(f"{server_url}/health")
        return response.status_code == 200
    except requests.RequestException:
        return False


def submit_search_job(
    server_url: str,
    query: str,
    dataset_id: Optional[str] = None,
    include_metadata: bool = False
) -> Dict[str, Any]:
    """Submit a search job to the server"""
    # The new server expects a different format
    search_data = {
        "query": query
    }
    
    if dataset_id:
        search_data["dataset_id"] = dataset_id
        
    if include_metadata:
        search_data["include_metadata"] = True
    
    response = requests.post(f"{server_url}/search", json=search_data)
    response.raise_for_status()
    return response.json()


def get_datasets(server_url: str) -> List[Dict[str, Any]]:
    """Get list of available datasets from the server"""
    response = requests.get(f"{server_url}/datasets")
    response.raise_for_status()
    return response.json()


def get_job_status(server_url: str, job_id: str) -> Dict[str, Any]:
    """Get status of a job from the server"""
    response = requests.get(f"{server_url}/search/{job_id}")
    response.raise_for_status()
    return response.json()

def format_results(results: Dict[str, Any], format_type: str = 'text') -> str:
    """Format search results for output"""
    if format_type == 'json':
        return json.dumps(results, indent=2)
    
    # Text format
    output = []
    output.append(f"Search Results (Job ID: {results['job_id']})")
    output.append(f"Dataset: {results['dataset_id']}")
    output.append(f"Status: {results['status']}")
    
    if results['status'] == 'completed':
        if 'completed_at' in results and results['completed_at']:
            execution_time = results['completed_at'] - results['created_at']
            output.append(f"Execution time: {execution_time:.2f} seconds")
        
        output.append(f"\nQuery Type: {results.get('query', {}).get('query_type', 'Unknown')}")
        output.append("\nResults by sequence:")
        
        if 'results' in results:
            for seq, result_data in results['results'].items():
                # Truncate long sequences for display
                display_seq = seq[:50] + '...' if len(seq) > 50 else seq
                
                cells = result_data.get('cells', [])
                counts = result_data.get('counts', [])
                metadata = result_data.get('metadata', {})
                
                output.append(f"\nSequence: {display_seq}")
                output.append(f"Found in {len(cells)} cells:")
                
                # List the first 5 cells and summarize the rest
                for i, cell in enumerate(cells[:5]):
                    count_info = f" with {counts[i]} counts" if i < len(counts) else ""
                    
                    # Add metadata if available
                    meta_info = ""
                    if metadata and cell in metadata:
                        cell_meta = metadata[cell]
                        meta_info = " ["
                        meta_items = []
                        for key, value in cell_meta.items():
                            meta_items.append(f"{key}: {value}")
                        meta_info += ", ".join(meta_items) + "]"
                    
                    output.append(f"  - Cell {cell}{count_info}{meta_info}")
                
                if len(cells) > 5:
                    output.append(f"  - ... and {len(cells) - 5} more cells")
    
    elif results['status'] == 'error':
        output.append(f"Error: {results.get('error', 'Unknown error')}")
    
    return '\n'.join(output)


def wait_for_completion(server_url: str, job_id: str, poll_interval: int = 1) -> Dict[str, Any]:
    """Wait for a job to complete, showing progress"""
    spinner = ['|', '/', '-', '\\']
    spinner_idx = 0
    
    sys.stdout.write("Waiting for job to complete ")
    sys.stdout.flush()
    
    while True:
        job_info = get_job_status(server_url, job_id)
        
        if job_info['status'] in ['completed', 'error']:
            sys.stdout.write("\n")
            return job_info
        
        # Show progress information if available
        progress = job_info.get('progress')
        step_desc = job_info.get('step_description', '')
        
        if progress is not None and step_desc:
            sys.stdout.write(f"\rProgress: {progress}% - {step_desc} {spinner[spinner_idx]} ")
        else:
            sys.stdout.write(f"\rWaiting for job to complete {spinner[spinner_idx]} ")
            
        sys.stdout.flush()
        spinner_idx = (spinner_idx + 1) % len(spinner)
        
        time.sleep(poll_interval)


def _run_list_datasets(args):
    """Handle the list-datasets command"""
    server_url = args.server
    
    if not check_server_connection(server_url):
        logging.error(f"Cannot connect to server at {server_url}")
        return 1
    
    try:
        datasets = get_datasets(server_url)
        
        print(f"Available datasets ({len(datasets)}):")
        print("-" * 80)
        
        for ds in datasets:
            print(f"ID: {ds['dataset_id']}")
            print(f"Name: {ds['name']}")
            print(f"Description: {ds['description']}")
            print(f"Organism: {ds['organism']}")
            print(f"Tissue: {ds.get('tissue', 'N/A')}")
            print(f"Technology: {ds.get('technology', 'N/A')}")
            print(f"Cell count: {ds['cell_count']:,}")
            print(f"K-mer size: {ds['kmer_size']}")
            print("-" * 80)
        
        return 0
    
    except Exception as e:
        logging.error(f"Error listing datasets: {str(e)}")
        return 1


def _run_job_status(args):
    """Handle the job status command"""
    server_url = args.server
    job_id = args.job_id
    
    if not check_server_connection(server_url):
        logging.error(f"Cannot connect to server at {server_url}")
        return 1
    
    try:
        job_info = get_job_status(server_url, job_id)
        
        if args.output:
            # Save to file
            with open(args.output, 'w') as f:
                if args.format == 'json':
                    json.dump(job_info, f, indent=2)
                else:
                    f.write(format_results(job_info, args.format))
            print(f"Results saved to {args.output}")
        else:
            # Print to console
            print(format_results(job_info, args.format))
        
        return 0
    
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logging.error(f"Job {job_id} not found")
        else:
            logging.error(f"Error getting job status: {str(e)}")
        return 1
    except Exception as e:
        logging.error(f"Error getting job status: {str(e)}")
        return 1


def _run_search_data(args):
    """Handle the search command"""
    server_url = args.server
    
    # Check server connection
    if not check_server_connection(server_url):
        logging.error(f"Cannot connect to server at {server_url}")
        return 1
    
    try:
        # Get query content
        query = ""
        
        if args.sequence:
            query = args.sequence
        
        elif args.file:
            try:
                # Try to read as sequences
                file_sequences = read_sequences_file(args.file)
                if file_sequences:
                    # Use the first sequence if multiple are found
                    query = file_sequences[0]
                    logging.info(f"Using first sequence from file ({len(query)} bases)")
                else:
                    # If no sequences found, try as plain text
                    with open(args.file, 'r') as f:
                        query = f.read().strip()
                    logging.info(f"Using file content as raw query ({len(query)} characters)")
            except Exception as e:
                logging.error(f"Error reading file {args.file}: {str(e)}")
                return 1
        
        if not query:
            logging.error("No query provided. Use --sequence or --file")
            return 1
        
        # Submit job
        logging.info(f"Submitting search job to dataset {args.dataset or 'default'}")
        job_info = submit_search_job(
            server_url=server_url,
            query=query,
            dataset_id=args.dataset,
            include_metadata=args.metadata
        )
        
        logging.info(f"Job submitted with ID: {job_info['job_id']}")
        
        # Wait for completion if requested
        if args.wait:
            logging.info("Waiting for job to complete...")
            job_info = wait_for_completion(server_url, job_info['job_id'])
            
            # Output results
            if args.output:
                # Save to file
                with open(args.output, 'w') as f:
                    if args.format == 'json':
                        json.dump(job_info, f, indent=2)
                    else:
                        f.write(format_results(job_info, args.format))
                logging.info(f"Results saved to {args.output}")
            else:
                # Print to console
                print(format_results(job_info, args.format))
        else:
            logging.info(f"Job status can be checked with: malva-search status {job_info['job_id']}")
            
            # If output file is specified but not waiting, just save the job ID
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(f"Job ID: {job_info['job_id']}\n")
                    f.write(f"Check status with: malva-search status {job_info['job_id']}\n")
                logging.info(f"Job ID saved to {args.output}")
        
        return 0
    
    except Exception as e:
        logging.error(f"Error performing search: {str(e)}")
        return 1
    
    
def _run_search(args):
    if args.command == "query":
        return _run_search_data(args)
    elif args.command == "list-datasets":
        return _run_list_datasets(args)
    elif args.command == "status":
        return _run_job_status(args)
    else:
        return 1

if __name__ == "__main__":
    from malva.cli import get_search_parser
    args = get_search_parser().parse_args()
    _run_search()
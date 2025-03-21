import sys
import time
import json
import requests
import logging
from typing import List, Dict, Any, Optional

# TODO: replace with dnaio
def read_sequences_file(file_path: str) -> List[str]:
    """Read sequences from a file (supports FASTA format)"""
    sequences = []
    current_seq = ""
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ""
            else:
                current_seq += line
    
    if current_seq:
        sequences.append(current_seq)
    
    return sequences


def check_server_connection(server_url: str) -> bool:
    """Check if the Malva search server is running"""
    try:
        response = requests.get(f"{server_url}/datasets")
        return response.status_code == 200
    except requests.RequestException:
        return False


def submit_search_job(
    server_url: str,
    sequences: List[str],
    dataset_id: str,
    min_kmer_matches: int = 1,
    k_size: Optional[int] = None
) -> Dict[str, Any]:
    """Submit a search job to the server"""
    search_data = {
        "dataset_id": dataset_id,
        "sequences": sequences,
        "min_kmer_matches": min_kmer_matches,
        "k_size": k_size
    }
    
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
        
        output.append("\nResults by sequence:")
        
        for seq, cells in results['results'].items():
            # Truncate long sequences for display
            display_seq = seq[:50] + '...' if len(seq) > 50 else seq
            output.append(f"\nSequence: {display_seq}")
            output.append(f"Found in {len(cells)} cells:")
            
            # List the first 10 cells and summarize the rest
            for i, cell in enumerate(cells[:10]):
                output.append(f"  - Cell {cell}")
            
            if len(cells) > 10:
                output.append(f"  - ... and {len(cells) - 10} more cells")
    
    elif results['status'] == 'error':
        output.append(f"Error: {results.get('error', 'Unknown error')}")
    
    return '\n'.join(output)


def wait_for_completion(server_url: str, job_id: str, poll_interval: int = 5) -> Dict[str, Any]:
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
        # Get sequences
        sequences = []
        
        if args.sequence:
            sequences.append(args.sequence)
        
        if args.file:
            file_sequences = read_sequences_file(args.file)
            if not file_sequences:
                logging.error(f"No sequences found in file {args.file}")
                return 1
            sequences.extend(file_sequences)
        
        if not sequences:
            logging.error("No sequences provided. Use --sequence or --file")
            return 1
        
        # Get available datasets if not specified
        if not args.dataset:
            datasets = get_datasets(server_url)
            if not datasets:
                logging.error("No datasets available on the server")
                return 1
            
            # Use the first dataset
            dataset_id = datasets[0]['dataset_id']
            logging.info(f"No dataset specified, using {dataset_id}: {datasets[0]['name']}")
        else:
            dataset_id = args.dataset
        
        # Submit job
        logging.info(f"Submitting search job for {len(sequences)} sequences in dataset {dataset_id}")
        job_info = submit_search_job(
            server_url=server_url,
            sequences=sequences,
            dataset_id=dataset_id,
            min_kmer_matches=args.min_matches
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
    if args.command == "search":
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
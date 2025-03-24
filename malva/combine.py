import h5py
import os
import shutil
import logging
import json
import uuid

from rich.progress import track

from malva.index import MalvaIndex
from malva.utils import check_directory_exists, check_file_exists

def combine_indices(combine_dirs, project_uuids=None):
    output_file = os.path.join(combine_dirs, "malva_index.h5")
    project_mapping = {}

    with h5py.File(output_file, 'w', driver='split') as f:
        n_chunks_total = 0
        n_unique_projects = 0

        if project_uuids is None:
            _tracker = os.listdir(combine_dirs)
        else:
            _tracker = [os.path.join(combine_dirs, puuid) for puuid in project_uuids]
    
        for index_folder in track(_tracker, description=f'Processing sub-indices'):
            folder_path = os.path.join(combine_dirs, index_folder)
            index_file = os.path.join(folder_path, "malva_index.h5")

            if not os.path.isdir(folder_path):
                continue
            
            if not os.path.exists(f'{index_file}-r.h5') or not os.path.exists(f'{index_file}-m.h5'):
                raise FileNotFoundError(f"The malva index {folder_path} was not found")
            
            # Create project mapping
            project_id = n_unique_projects
            project_uuid = str(uuid.uuid4()) if project_uuids is None else project_uuids[project_id]

            with h5py.File(index_file, 'r', driver="split") as index_f:
                n_chunks = index_f.attrs['n_chunks']
                for i in range(n_chunks):
                    indices_dataset_name = f"index_{n_chunks_total}_indices"
                    indptr_dataset_name = f"index_{n_chunks_total}_indptr"
                    data_dataset_name = f"index_{n_chunks_total}_data"

                    f[indices_dataset_name] = h5py.ExternalLink(f'{index_file}', f"index_{i}_indices")
                    f[indptr_dataset_name] = h5py.ExternalLink(f'{index_file}', f"index_{i}_indptr")
                    f[data_dataset_name] = h5py.ExternalLink(f'{index_file}', f"index_{i}_data")

                    project_mapping[n_chunks_total] = (project_id, project_uuid)

                    n_chunks_total += 1

                if 'spatial_coord' in index_f and 'spatial_coord' not in f:
                    # TODO: fix so we store the spatial coordinates from various projects. Only works for single cell now!
                    f.create_dataset('spatial_coord', data=index_f['spatial_coord'])
                
                if 'kmer_size' not in f.attrs and 'kmer_size' in index_f.attrs:
                    f.attrs['kmer_size'] = index_f.attrs['kmer_size']

                if 'coord_lims' not in f.attrs and 'coord_lims' in index_f.attrs:
                    f.attrs['coord_lims'] = index_f.attrs['coord_lims']

                if 'n_spatial' not in f.attrs and 'n_spatial' in index_f.attrs:
                    f.attrs['n_spatial'] = index_f.attrs['n_spatial']

            n_unique_projects += 1

        f.attrs['project_mapping'] = json.dumps(project_mapping)
        f.attrs['n_chunks'] = n_chunks_total
        logging.info(f"Created combined index with {n_chunks_total} chunks from {n_unique_projects} projects")
        

def _run_combine(args):
    if not check_directory_exists(args.index_in):
        logging.error("Base directory does not exist")
        return
    
    project_uuids = None
    if args.merge_projects and check_file_exists(args.uuid):
        with open(args.uuid) as file:
            project_uuids = [line.rstrip() for line in file]
    
    index_out = os.path.join(args.index_in, "malva_index.h5")
    
    # we need to check given the 'split' driver
    if not check_file_exists(index_out + "-r.h5") or not check_file_exists(index_out + "-m.h5"):
        combine_indices(args.index_in, project_uuids)
    else:
        logging.warning(f"The combined (non-merged) index exists already at {index_out}")

    if args.merge_chunks:
        kmer_index = MalvaIndex(args.index_in)
        kmer_index.verbose = True

        if kmer_index.n_chunks > 1:
            logging.info(f"Now, {kmer_index.n_chunks} chunks will be merged")
            
            merged_file = f"{kmer_index.index_file}.merged"

            if args.merge_projects:
                logging.info("Merging projects with distinct project IDs")
                
            kmer_index.merge_chunks(merged_file, _merge_projects=args.merge_projects)
            
            kmer_index.verbose = True
            os.remove(f'{kmer_index.index_file}-r.h5')
            os.remove(f'{kmer_index.index_file}-m.h5')
            shutil.move(f'{merged_file}-r.h5', f'{kmer_index.index_file}-r.h5')
            shutil.move(f'{merged_file}-m.h5', f'{kmer_index.index_file}-m.h5')

    logging.info("SUCCESS!")
        

if __name__ == "__main__":
    from malva.cli import get_combine_parser

    args = get_combine_parser().parse_args()
    _run_combine()
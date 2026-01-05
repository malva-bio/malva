import h5py
import os
import shutil
import logging
import json
import uuid
import tempfile

from rich.progress import track
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import current_thread, main_thread

from malva.index import MalvaIndex
from malva.utils import check_directory_exists, check_file_exists

def combine_indices(combine_dir, project_uuids=None, project_id_offset=0):
    output_file = os.path.join(combine_dir, "malva_index.h5")
    project_mapping = {}

    with h5py.File(output_file, 'w', driver='split') as f:
        n_chunks_total = 0
        n_unique_projects = 0

        if project_uuids is None:
            _tracker = sorted(os.listdir(combine_dir))
        else:
            _tracker = [os.path.join(combine_dir, puuid) for puuid in project_uuids]

        iterator = track(_tracker, description='Processing sub-indices') if current_thread() == main_thread() else _tracker
        for index_folder in iterator:
            folder_path = os.path.join(combine_dir, index_folder) if not os.path.isabs(index_folder) else index_folder
            index_file = os.path.join(folder_path, "malva_index.h5")

            if not os.path.isdir(folder_path):
                continue

            if not os.path.exists(f'{index_file}-r.h5') or not os.path.exists(f'{index_file}-m.h5'):
                raise FileNotFoundError(f"The malva index {folder_path} was not found")

            project_id = project_id_offset + n_unique_projects
            project_uuid = os.path.basename(folder_path) if project_uuids is None else project_uuids[project_id]

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

        return project_mapping, n_chunks_total
        

def process_merge_chunks(index_dir, merge_projects=False):
    mindex = MalvaIndex(index_dir)
    mindex.verbose = True
    if mindex.n_chunks > 1:
        logging.info(f"Merging {mindex.n_chunks} chunks in index at {index_dir}")
        merged_file = f"{mindex.index_file}.merged"

        if merge_projects:
            logging.info("Merging projects with distinct project IDs")
        
        mindex.merge_chunks(merged_file, merge_projects=merge_projects)
        
        os.remove(f'{mindex.index_file}-r.h5')
        os.remove(f'{mindex.index_file}-m.h5')
        shutil.move(f'{merged_file}-r.h5', f'{mindex.index_file}-r.h5')
        shutil.move(f'{merged_file}-m.h5', f'{mindex.index_file}-m.h5')

def process_group(group_idx, group, base_dir, project_uuids, merge_chunks, merge_projects, project_id_offset=0):
    # Add project_id_offset parameter with default value of 0
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    temp_group_dir = tempfile.mkdtemp(prefix="malva_group_", dir=parent_dir)

    sorted_group = sorted(group)
    
    for folder in sorted_group:
        src = os.path.join(base_dir, folder)
        dst = os.path.join(temp_group_dir, folder)
        try:
            os.symlink(src, dst)
        except Exception as e:
            logging.warning(f"Symlink failed for {src} with error {e}, using copytree instead.")
            shutil.copytree(src, dst)
    
    # Pass the project_id_offset to combine_indices
    mapping, _ = combine_indices(temp_group_dir, None, project_id_offset=project_id_offset)
    if merge_chunks:
        process_merge_chunks(temp_group_dir, merge_projects)
    
    return group_idx, mapping, temp_group_dir

def hierarchical_combine(base_dir, project_uuids, merge_chunks=False, merge_projects=False, group_size=16, threads=1):
    all_indices = [d for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d)) and 
                      d not in ["malva_index.h5", f"malva_index.h5-r.h5", f"malva_index.h5-m.h5"]]
    
    all_indices.sort()
    
    if len(all_indices) <= group_size:
        logging.info("Number of indices is within direct merge limit. Calling combine_indices directly.")
        combine_indices(base_dir, project_uuids)
        if merge_chunks:
            process_merge_chunks(base_dir, merge_projects)
        return

    logging.info(f"Performing hierarchical merge on {len(all_indices)} indices in groups of {group_size}")

    groups = []
    num_groups = 0
    current_project_offset = 0
    
    for i in range(0, len(all_indices), group_size):
        group = all_indices[i:i + group_size]
        if group:
            groups.append((num_groups, group, current_project_offset))
            current_project_offset += len(group)
            num_groups += 1

    results = []
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(
            process_group, 
            group_idx, 
            group, 
            base_dir, 
            project_uuids, 
            merge_chunks, 
            merge_projects,
            project_id_offset
        ): group_idx for group_idx, group, project_id_offset in groups}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                logging.info(f"Completed processing group {futures[future]}")
            except Exception as e:
                logging.error(f"Error processing group {futures[future]}: {e}")
    
    # Sort results by group index to maintain correct order
    results.sort(key=lambda x: x[0])
    
    group_mappings = []
    intermediate_dirs = []
    
    # We now have proper project IDs from process_group, no need to update them
    for group_idx, mapping, temp_group_dir in results:
        group_mappings.append(mapping)
        intermediate_dirs.append(temp_group_dir)

    # merge the intermediate merged indices into a final combined index
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    final_temp_dir = tempfile.mkdtemp(prefix="malva_final_", dir=parent_dir)
    logging.info(f"Merging {len(intermediate_dirs)} intermediate indices into final index")
    
    for idx, inter_dir in enumerate(intermediate_dirs):
        link_name = os.path.join(final_temp_dir, f"intermediate_{idx}")
        try:
            os.symlink(inter_dir, link_name)
        except Exception as e:
            logging.warning(f"Symlink failed for {inter_dir} with error {e}, using copytree instead.")
            shutil.copytree(inter_dir, link_name)
    
    combine_indices(final_temp_dir, project_uuids=None)
    if merge_chunks:
        process_merge_chunks(final_temp_dir, merge_projects=False)

    # project mappings from intermediate indices (most reliable way...)
    final_mapping = {}
    global_chunk_idx = 0

    logging.info("Reading project mappings from intermediate indices...")
    for idx, inter_dir in enumerate(intermediate_dirs):
        inter_index = os.path.join(inter_dir, "malva_index.h5")
        try:
            with h5py.File(inter_index, 'r', driver='split') as f:
                if 'project_mapping' in f.attrs:
                    sub_mapping = json.loads(f.attrs['project_mapping'])
                    for key in sorted(sub_mapping.keys(), key=lambda x: int(x)):
                        # Original project ID and UUID from the subindex
                        if isinstance(sub_mapping[key], list):
                            project_id, project_uuid = sub_mapping[key]
                        else:
                            project_id, project_uuid = sub_mapping[key]
                        final_mapping[str(global_chunk_idx)] = [project_id, project_uuid]
                        global_chunk_idx += 1
        except Exception as e:
            logging.error(f"Error reading project mapping from {inter_index}: {e}")

    final_index = os.path.join(final_temp_dir, "malva_index.h5")
    with h5py.File(final_index, 'r+', driver='split') as f:
        f.attrs['project_mapping'] = json.dumps(final_mapping)
        logging.info(f"Updated final index with project mapping for {len(final_mapping)} chunks")

    # Ensure file exists before moving
    for suffix in ["", "-r.h5", "-m.h5"]:
        source_file = f"{final_index}{suffix}"
        target_file = os.path.join(base_dir, f"malva_index.h5{suffix}")
        
        if os.path.exists(source_file):
            logging.info(f"Moving {source_file} to {target_file}")
            shutil.copy2(source_file, target_file)
            try:
                os.remove(source_file)
            except Exception as e:
                logging.warning(f"Could not remove source file {source_file}: {e}")
        else:
            logging.warning(f"Source file {source_file} does not exist, skipping move operation")

    # Clean up with error handling
    try:
        for temp_dir in intermediate_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        if os.path.exists(final_temp_dir):
            shutil.rmtree(final_temp_dir)
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
    
    logging.info("Hierarchical merge complete and temporary files cleaned up.")

def _run_combine(args):
    if not check_directory_exists(args.index_in):
        logging.error("Base directory does not exist")
        return

    project_uuids = None
    if args.merge_projects and check_file_exists(args.uuid):
        with open(args.uuid) as file:
            project_uuids = [line.rstrip() for line in file]

    index_out = os.path.join(args.index_in, "malva_index.h5")
    # Check if the combined index already exists.
    if check_file_exists(index_out + "-r.h5") and check_file_exists(index_out + "-m.h5"):
        logging.warning(f"The combined (non-merged) index already exists at {index_out}")
    else:
        # Identify sub-indices in the base directory.
        index_dirs = [d for d in os.listdir(args.index_in)
                      if os.path.isdir(os.path.join(args.index_in, d)) and
                      d not in ["malva_index.h5", f"malva_index.h5-r.h5", f"malva_index.h5-m.h5"]]
        if len(index_dirs) <= 16:
            logging.info("Number of indices is 16 or less, merging directly.")
            combine_indices(args.index_in, project_uuids)
        else:
            logging.info("Large number of indices detected, performing hierarchical merging.")
            hierarchical_combine(args.index_in, project_uuids, merge_chunks=args.merge_chunks,
                                 merge_projects=args.merge_projects, threads=args.threads)
            
            # we re-configure the number of spatial merged coordinates
            mindex = MalvaIndex(args.index_in, verbose=True)
            mindex.open("r+")
            mindex.index.attrs['n_spatial'] = 1_000_000_000
            mindex.close()

    # After combining, if merge_chunks is enabled and there is more than one chunk,
    # perform a final chunk merge on the combined index.
    if args.merge_chunks:
        # Create a MalvaIndex instance for the final merged index.
        mindex = MalvaIndex(args.index_in, verbose=True)
        if mindex.n_chunks > 1:
            logging.info(f"Now, {mindex.n_chunks} chunks will be merged at final level")
            merged_file = f"{mindex.index_file}.merged"

            if args.merge_projects:
                logging.info("Merging projects with distinct project IDs at final level")

            mindex.merge_chunks(merged_file, merge_projects=args.merge_projects)
            os.remove(f'{mindex.index_file}-r.h5')
            os.remove(f'{mindex.index_file}-m.h5')
            shutil.move(f'{merged_file}-r.h5', f'{mindex.index_file}-r.h5')
            shutil.move(f'{merged_file}-m.h5', f'{mindex.index_file}-m.h5')

    logging.info("SUCCESS!")
        

if __name__ == "__main__":
    from malva.cli import get_combine_parser

    args = get_combine_parser().parse_args()
    _run_combine()
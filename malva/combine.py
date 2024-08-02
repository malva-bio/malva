import h5py
import os
import logging

from rich.progress import track

from malva.index import MalvaIndex
from malva.utils import check_directory_exists, check_file_exists

def combine_indices(combine_dirs, index_out):
    output_file = os.path.join(combine_dirs, "malva_index.h5")

    with h5py.File(output_file, 'w', driver='split') as f:
        n_chunks_total = 0
        spatial_coords = []

        for index_folder in track(os.listdir(combine_dirs), description=f'Processing sub-indices'):
            folder_path = os.path.join(combine_dirs, index_folder)
            index_file = os.path.join(folder_path, "malva_index.h5")

            if not os.path.isdir(folder_path):
                continue
            
            if not os.path.exists(f'{index_file}-r.h5') or not os.path.exists(f'{index_file}-m.h5'):
                raise FileNotFoundError(f"The malva index {folder_path} was not found")

            with h5py.File(index_file, 'r', driver="split") as index_f:
                n_chunks = index_f.attrs['n_chunks']
                for i in range(n_chunks):
                    indices_dataset_name = f"index_{n_chunks_total}_indices"
                    indptr_dataset_name = f"index_{n_chunks_total}_indptr"
                    data_dataset_name = f"index_{n_chunks_total}_data"

                    f.create_dataset(indices_dataset_name, data=index_f[f"index_{i}_indices"])
                    f.create_dataset(indptr_dataset_name, data=index_f[f"index_{i}_indptr"])
                    f.create_dataset(data_dataset_name, data=index_f[f"index_{i}_data"])

                    n_chunks_total += 1

                if 'spatial_coord' in index_f and 'spatial_coord' not in f:
                    f.create_dataset('spatial_coord', data=index_f['spatial_coord'])
                
                if 'kmer_size' not in f.attrs:
                    f.attrs['kmer_size'] = index_f.attrs['kmer_size']

                if 'coord_lims' not in f.attrs:
                    f.attrs['coord_lims'] = index_f.attrs['coord_lims']

                if 'n_spatial' not in f.attrs:
                    f.attrs['n_spatial'] = index_f.attrs['n_spatial']

        f.attrs['n_chunks'] = n_chunks_total
        logging.info(f"Created combined index with {n_chunks_total} chunks")
        

def _run_combine(args):
    if not check_directory_exists(args.index_in):
        logging.error("Base directory does not exist")
        return
    
    index_out = os.path.join(args.index_in, "malva_index.h5")
    check_file_exists(index_out, except_when=True)
    combine_indices(args.index_in, index_out)

    if args.merge_chunks:
        kmer_index = MalvaIndex(args.index_in)

        if kmer_index.n_chunks > 1:
            logging.info(f"Now, {kmer_index.n_chunks} chunks will be merged")
            
            merged_file = f"{kmer_index.index_file}.merged"
            kmer_index.merge_chunks(merged_file)
            os.remove(f'{kmer_index.index_file}-r.h5')
            os.remove(f'{kmer_index.index_file}-m.h5')
            shutil.move(f'{merged_file}-r.h5', f'{kmer_index.index_file}-r.h5')
            shutil.move(f'{merged_file}-m.h5', f'{kmer_index.index_file}-m.h5')

    logging.info("SUCCESS!")
        

if __name__ == "__main__":
    from malva.cli import get_combine_parser

    args = get_combine_parser().parse_args()
    _run_combine()




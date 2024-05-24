# TODO: implement multithreading for faster indexing
# TODO: convert iterate_flavor into a class?

import logging
import os
import yaml
import h5py
import shutil
import time

import numpy as np
import pandas as pd

from rich.progress import track
from sortedcontainers import SortedDict
from operator import itemgetter

from katoste.utils import check_file_exists, check_directory_exists, load_pickle, safety_check_eval, binary_search, save_pickle, get_module_path, group_intervals
from katoste.kmer_processing import get_kmers_numeric, get_overlapping_kmers_numeric, encode_kmer

N_CHUNK = 1_000_000_000
N_REPORT = 1_000_000
MAX_PCT = 0.01
MIN_KMER_QUERY = 10
QUERY_JUMP = False


class KatosteIndex:
    def __init__(self, index_dir, rewrite=False, kmer_size_initialize=8):
        self.rewrite = rewrite
        self.index_dir = index_dir
        self.index_file = os.path.join(self.index_dir, 'katoste_index.h5')

        self.kmer_size = None
        self.index = None
        self._index_backed = None
        self.coord_lims = None
        self.n_chunks = 0
        self.data_lengths = None
        self.n_spatial = None
        self.binary_search = self._binary_search_np
        self.get_kmers_numeric = get_kmers_numeric

        self._n_kmers_processed = 0
        self._iter_seqs = []
        self._iter_coords = []

        exists = False
        if rewrite:
            exists = check_directory_exists(self.index_dir)
            if exists:
                shutil.rmtree(self.index_dir, ignore_errors=True)
            
            os.mkdir(self.index_dir)
        else:
            exists = self.index_exists(self)
            if exists:
                logging.info("The index exists. Will load now.")
        
        if not exists:
            logging.info(f"Will create katoste index at `{self.index_file}` with {kmer_size_initialize}-mers")
            self.initialize(kmer_size=kmer_size_initialize)

    @staticmethod
    def index_exists(self):
        exists = check_directory_exists(self.index_dir)
        if exists:
            exists = check_file_exists(self.index_file)

        return exists

    def initialize(self, kmer_size=8):
        if kmer_size < 8 or kmer_size > 32:
            raise ValueError("`kmer_size` must be between 8 and 32 (inclusive)")
        if kmer_size < 24:
            logging.warn(f"For accuracy and speed, we recommend using > 24-mers (currently using {kmer_size}-mers)")

        self.kmer_size = kmer_size

        self.initialize_kmer_index()

    def initialize_kmer_index(self):
        self.open(mode='w')
        self.index.attrs['kmer_size'] = self.kmer_size
        self.index.attrs['n_chunks'] = self.n_chunks
        self.close()

    def append_spatial(self, index):
        self.spatial_index = index
        
        _coord = np.array(list(index.values()))
        self.spatial_coords = _coord[:, 1:]
        
        xmin, xmax = self.spatial_coords[:, 0].min(), self.spatial_coords[:, 0].max()
        ymin, ymax = self.spatial_coords[:, 1].min(), self.spatial_coords[:, 1].max()

        self.coord_lims = (xmin, xmax, ymin, ymax)
        self.n_spatial = _coord[:, 0].max()
        
        self.open(mode='r+')
        self.index.attrs['coord_lims'] = self.coord_lims
        self.index.attrs['n_spatial'] = self.n_spatial
        self.close()

    def open(self, mode='r'):
        self.index = h5py.File(self.index_file, mode)
        if self._index_backed is None or isinstance(self._index_backed, h5py.File):
            self._index_backed = self.index

        if 'kmer_size' in self.index.attrs:
            self.kmer_size = self.index.attrs['kmer_size']

        if 'coord_lims' in self.index.attrs:
            self.coord_lims = self.index.attrs['coord_lims']

        if 'n_spatial' in self.index.attrs:
            self.n_spatial = self.index.attrs['n_spatial']

        if 'n_chunks' in self.index.attrs:
            self.n_chunks = self.index.attrs['n_chunks']
            self.data_lengths = [len(self.index[f'index_{chunk}_data']) for chunk in range(self.n_chunks)]

    def close(self):
        self.index.close()

    def process_kmer(self, kmer, coord_ix):
        if self._n_kmers_processed == 0:
            logging.warn("There are no kmers to process!")
            return

        _chunk = self.n_chunks

        _ix_sorted = np.argsort(kmer)

        kmer = kmer[_ix_sorted]

        k_unique = np.unique(kmer)
        k_change = np.concatenate([np.array([0]), np.where(kmer[:-1] != kmer[1:])[0]])

        if len(kmer) == 0:
            pass

        self.open(mode='r+')
        self.index.create_dataset(f"index_{_chunk}_indices", data=k_unique)
        self.index.create_dataset(f"index_{_chunk}_indptr", data=k_change, dtype=np.uint32)
        self.index.create_dataset(f"index_{_chunk}_data", data=coord_ix[_ix_sorted], dtype=np.uint32)
    
        self.n_chunks += 1
        self.index.attrs['n_chunks'] = self.n_chunks
        self.close()

    def add_sequence(self, sequence, index):
        coords = self.spatial_index.get(encode_kmer(index), None)
        if coords is None:
            return

        kmers = self.get_kmers_numeric(sequence, self.kmer_size)
        _n_kmers = len(kmers)
        self._iter_seqs.extend(kmers)
        self._iter_coords.extend([coords[0]]*_n_kmers)
        self._n_kmers_processed += _n_kmers

    def write(self):
        self.process_kmer(np.array(self._iter_seqs), np.array(self._iter_coords))
        self._n_kmers_processed = 0
        self._iter_seqs = []
        self._iter_coords = []

    def _binary_search_np(self, arr, start, end, v):
        return np.searchsorted(arr, v)

    def _binary_search_katoste(self, arr, start, end, v):
        return binary_search(arr, start, end, v)
    
    def find_kmer(self, kmers):
        vals = {k: set([-1]) for k in kmers}

        # _indptrs = np.zeros(max(self.data_lengths), dtype=np.uint64)

        for ch in track(range(self.n_chunks), description=f"Querying chunks"):
            _data = self.index[f'index_{ch}_data']
            _h5_indices = self._index_backed[f'index_{ch}_indices']
            _h5_indptrs = self._index_backed[f'index_{ch}_indptr']
            _loc = np.searchsorted(_h5_indices, kmers)
            
            # _loc_group = group_intervals(_loc, min_interval=1_000)

            # for _lc in _loc_group:
            #     _indptrs[_lc[0]:_lc[1]+1] = _h5_indptrs[_lc[0]:_lc[1]+1]

            for i, kmer in enumerate(kmers):
                if _loc[i] == -1:
                    continue

                _indptr = _h5_indptrs[_loc[i]:_loc[i]+2]
                if (len(_indptr) != 2):
                    _indptr = [_indptr[0], self.data_lengths[ch]]

                if ((_indptr[1]-_indptr[0])/self.data_lengths[ch] > MAX_PCT) or ((_indptr[1]-_indptr[0]) <= MIN_KMER_QUERY):
                    continue
                
                _res = _data[_indptr[0]:_indptr[1]]

                if len(_res) > 0:
                    vals[kmer].update(set(_res))

        vals = {k: np.array(list(v)) for k, v in vals.items()}
        return vals
    
    def _load_index_to_memory(self):
        # only load from memory when a file or None; skip if already a dict (assumes proper format)
        if isinstance(self._index_backed, h5py.File) or self._index_backed is None:
            self._index_backed = {f'index_{i}_indices': self.index[f'index_{i}_indices'][:] for i in range(self.n_chunks)}
            self._index_backed.update({f'index_{i}_indptr': self.index[f'index_{i}_indptr'][:] for i in range(self.n_chunks)})
            self.binary_search = self._binary_search_katoste

    def where(self, sequence, sliding_size=128, pct_threshold=0.65, lazy_index=True, max_pct=0.01, min_kmer_query=10, query_jump=False):
        global MAX_PCT, MIN_KMER_QUERY, QUERY_JUMP

        # TODO: use as arguments for the find_kmer variable
        MAX_PCT = max_pct
        MIN_KMER_QUERY = min_kmer_query
        QUERY_JUMP = query_jump

        if len(sequence) < self.kmer_size:
            raise ValueError("Query sequence cannot be smaller than kmer size!")

        if pct_threshold < 0 or pct_threshold > 1:
            raise ValueError("`pct_threshold` must be a valid value between 0 and 1")

        # TODO: move into a context manager
        self.open()
        if not lazy_index:
            logging.debug("Will not use lazy loading - the chunk indexes are loaded into memory")
            self._load_index_to_memory()

        # TODO: collapse seq_no and all_seq_no into one
        all_oc = []
        seq_matches = [[0, 1]]
        seq_no = 0
        all_seq_no = 0

        def get_sliding_sequence(string, k):
            n = len(string)
            sliding_seqs = []

            for i in range(n - k + 1):
                slide = string[i:i + k]
                sliding_seqs.append(slide)

            return sliding_seqs
        
        # get data for kmers
        all_kmer_list = []
        _necessary = np.ceil((len(sequence)-self.kmer_size)/np.floor(len(sequence)/self.kmer_size)).astype(int)
        for subseq in get_sliding_sequence(sequence, len(sequence) - _necessary):
            all_kmer_list += [get_kmers_numeric(subseq, self.kmer_size, remove_noncomplex=True)]

        all_kmer_list = np.unique(all_kmer_list)
        # 0 is 'A'*kmer_size but also any low-complexity k-mer
        all_kmer_list = all_kmer_list[all_kmer_list != 0]

        if len(all_kmer_list) == 0:
            return (np.array([0]), np.array([0]), seq_matches)

        all_kmer_dict = self.find_kmer(all_kmer_list)

        sliding_sequences = get_sliding_sequence(sequence, min(len(sequence) - _necessary, sliding_size))

        for subseq in track(sliding_sequences, description='Counting kmers per sequence chunk'):
            if seq_no <= self.kmer_size or seq_no == int(len(subseq)/2) or not QUERY_JUMP:
                # TODO: store from previous and not rerun?
                all_kmer_list = get_kmers_numeric(subseq, self.kmer_size, remove_noncomplex=True)
                kmer_list = [i for i in all_kmer_list if i != 0]

                if len(kmer_list) == 0:
                    seq_matches += [[all_seq_no + k*self.kmer_size, 0] for k in range(len(all_kmer_list))]
                    seq_no += 1
                    all_seq_no += 1
                    continue
    
                all_items = itemgetter(*kmer_list)(all_kmer_dict)

                if len(all_items) == 0:
                    seq_matches += [[all_seq_no + k*self.kmer_size, 0] for k in range(len(all_kmer_list))]
                    seq_no += 1
                    all_seq_no += 1
                    continue
                else:
                    all_items = np.hstack(all_items)

                all_items = all_items[all_items != np.array(-1)]
                all_items, counts = np.unique(all_items, return_counts=True)
                props_ix = np.where(counts / len(all_kmer_list) >= pct_threshold)[0]
                all_oc += [all_items[props_ix]]
                seq_matches += [[all_seq_no + k*self.kmer_size, len(all_items[props_ix])] for k in range(len(all_kmer_list))]

            seq_no += 1
            all_seq_no += 1
            if seq_no == sliding_size:
                seq_no = 0

        logging.info("Counting unique spatial matches")
        if len(all_oc) > 0:
            kmer_locations, kmer_count = np.unique(np.concatenate(all_oc), return_counts=True)
        else:
            kmer_locations, kmer_count = np.array([0]), np.array([0])
        
        self.close()

        return (kmer_locations, kmer_count, seq_matches)


def create_spatial_index(spatial_barcode_file, rescale_coords=1, index_resolution=1, recenter=True):
    check_file_exists(spatial_barcode_file, except_when=False)
    spatial_index = pd.read_csv(spatial_barcode_file, sep='\t')
    spatial_index.drop_duplicates(subset='cell_bc')

    if rescale_coords <= 0 or index_resolution <= 0:
        raise ValueError("`rescale_coords` and `index_resolution` must be greater than 0")
    
    if recenter:
        spatial_index['xcoord'] = spatial_index['xcoord'] - spatial_index['xcoord'].min()
        spatial_index['ycoord'] = spatial_index['ycoord'] - spatial_index['ycoord'].min()

    if rescale_coords != 1:
        spatial_index['xcoord'] *= rescale_coords
        spatial_index['ycoord'] *= rescale_coords

    if index_resolution != 1:
        spatial_index['xcoord'] /= index_resolution
        spatial_index['xcoord'] = spatial_index['xcoord'].astype(int)
        spatial_index['ycoord'] /= index_resolution
        spatial_index['ycoord'] = spatial_index['ycoord'].astype(int)

    if (any(spatial_index['xcoord'] < 0) or any(spatial_index['xcoord'] > 65_535)) \
        or (any(spatial_index['ycoord'] < 0) or any(spatial_index['ycoord'] > 65_535)):
        raise ValueError("Spatial coordinates must be between 0 and 65_535")
    
    xmax = spatial_index['xcoord'].max()
    ymax = spatial_index['ycoord'].max()

    sindex = SortedDict()

    for row in track(spatial_index.itertuples(), total=len(spatial_index), description='Spatial indexing'):
        i = np.ravel_multi_index([int(row.xcoord), int(row.ycoord)], (int(xmax)+1, int(ymax)+1), order='C')
        sindex[encode_kmer(row.cell_bc)] = (i, int(row.xcoord), int(row.ycoord))

    return sindex

def load_flavor(flavor, flavors_config_path):
    with open(flavors_config_path) as stream:
        try:
            flavor_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(exc)
    
    if flavor not in flavor_config['barcode_flavors']:
        raise ValueError(f"Flavor `{flavor}` could not be found")
    
    current_flavor_config = flavor_config['barcode_flavors'][flavor]
    return current_flavor_config

def iterate_flavor(reads_in,
                   bam_tags='CB:{cell}',
                   cell='r1[2:27]'):
    bam_tags = [bt.split(":") for bt in bam_tags.split(",")]
    bam_tags = {what[1:-1]: tag for what, tag in bam_tags}

    if len(reads_in) == 2:
        import dnaio

        # adapted from spacemake
        assert safety_check_eval(cell)
        f_cell = compile(cell, "<string cell>", "eval")
    
        with dnaio.open(reads_in[0], reads_in[1], fileformat='fastq') as f:
            for r1, r2 in f:
                r1 = r1.sequence
                r2 = r2.sequence  
                cell_bc = eval(f_cell)

                yield (r2, cell_bc)

    elif len(reads_in) == 1:
        import pysam

        _cell_tag = bam_tags['cell']
        with pysam.AlignmentFile(reads_in, "rb", until_eof=True) as f:
            for record in f:
                cell_bc = record.get_tag[_cell_tag]
                yield (record.sequence, cell_bc)
    else:
        raise ValueError("`--reads-in` must point to paired FASTQ files or a single BAM file")

def _run_index(args):
    # Validate that input files exist and output files don't
    for _r in args.reads_in:
        check_file_exists(_r, except_when=False)
    
    _out_dir_exists = check_directory_exists(args.index_out)
    if not _out_dir_exists:
        logging.info("Output directory does not exist. Creating...")
        os.mkdir(args.index_out)

    logging.info(f"Configuring flavor `{args.flavor}`")
    _config_path = os.path.join(get_module_path(), 'data', 'config.yaml')
    if check_file_exists("config.yaml"):
        _config_path = "config.yaml"

    flavor_config = load_flavor(args.flavor, _config_path)
    
    _sindex_loc = os.path.join(args.index_out, "sindex.pickle")
    _sindex_exists = check_file_exists(_sindex_loc)

    if _sindex_exists:
        logging.info("Loading previously created `cell (spot) barcode->spatial coordinate` index")
        sindex = load_pickle(_sindex_loc)
    else:
        logging.info("Creating `cell (spot) barcode->spatial coordinate` index")
        sindex = create_spatial_index(args.spatial_bc_in, args.rescale_coords, args.index_resolution, not args.no_recenter)
        logging.info("Saving `cell (spot) barcode->spatial coordinate` index")
        save_pickle(sindex, _sindex_loc)

    _n_locations = len(sindex)

    logging.info(f"Configuring the katoste index")
    kmer_index = KatosteIndex(args.index_out, kmer_size_initialize=args.kmer_length)

    # Set the proper function to generate k-mers
    if args.overlapping:
        logging.info(f"Will index overlapping {args.kmer_length}-mers")
        kmer_index.get_kmers_numeric = get_overlapping_kmers_numeric
    
    logging.info("Adding spatial index to katoste index")
    kmer_index.append_spatial(sindex)

    logging.info(f"Indexing sequence {args.kmer_length}-mers in space from {args.reads_in} with flavor {args.flavor}")
    logging.info(f"Will write to disk every {args.chunksize:,} sequences, and once at the end (remaining sequences)")
    _bam_tags = flavor_config['bam_tags']
    _cell = flavor_config['cell']

    _n_sequences = 0
    _t0 = time.time()
    for r2, cell_bc in iterate_flavor(args.reads_in, bam_tags=_bam_tags, cell=_cell):
        kmer_index.add_sequence(r2, cell_bc)
        _n_sequences += 1
        if (_n_sequences) % args.chunksize == 0:
            kmer_index.write()
        if (_n_sequences) % N_REPORT == 0:
            _t1 = time.time()
            _elapsed = _t1 - _t0
            logging.info(f"Processed {_n_sequences:,} sequences in {round(_elapsed, 2)} s ({round(N_REPORT/_elapsed, 2):,} reads/s)")
            _t0 = time.time()

    logging.info("Writing all remaining chunks into index")
    kmer_index.write()
    logging.info(f"Indexed {_n_sequences:,} sequences into {_n_locations:,} spatial locations")


if __name__ == "__main__":
    from katoste.cli import get_index_parser
    args = get_index_parser().parse_args()
    _run_index()
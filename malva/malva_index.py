# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

import json
import logging
import os
import shutil
import struct
import time
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class _IndexCompat:
    """Minimal shim that mimics HDF5 file for 'key in index' and index['key'] access."""
    def __init__(self, keys, parent):
        self._keys = keys
        self._parent = parent
    def __contains__(self, key):
        return key in self._keys
    def __getitem__(self, key):
        if key == 'spatial_coord':
            return self._parent.spatial_coord
        raise KeyError(key)
    @property
    def attrs(self):
        return {'n_spatial': self._parent.n_spatial}

class MalvaIndex:
    """
    High-performance prefix-bucketed k-mer index for spatial / single-cell data.

    Provides fast k-mer lookup for spatial and single-cell queries using prefix-bucketed
    compressed storage (pi.bin, suffixes.bin, data.bin) on disk.

    Attributes:
        index_dir (str): Directory where index files are stored
        kmer_size (int): Length of k-mers used in the index
        l_prefix (int): Number of bases used as prefix (default 12)
        l_suffix (int): Number of bases used as suffix
        verbose (bool): Whether to print detailed logging information
    """

    def __init__(self, index_dir, rewrite=False, kmer_size_initialize=24,
                 l_prefix=12, verbose=False, jump_amount=0,
                 max_project_capacity=512):
        self.index_dir = index_dir
        self.kmer_size = kmer_size_initialize
        self.l_prefix = l_prefix
        self.l_suffix = kmer_size_initialize - l_prefix
        self.jump_amount = kmer_size_initialize if jump_amount == 0 else jump_amount
        self.verbose = verbose
        self.n_spatial = 0
        self.coord_lims = None
        self.spatial_coord = None
        self.spatial_index = None  # SpatialIndex cdef class from Malva

        # Project merging support
        project_bits = int(np.ceil(np.log2(max_project_capacity)))
        self.PROJECT_ID_SHIFT = 32 - project_bits
        self.CELL_ID_MASK = (1 << (32 - project_bits)) - 1
        self.HAS_MERGED_PROJECTS = False
        self.project_mapping = None

        # The underlying Cython PrefixIndex for queries
        self._prefix_index = None

        # Chunk tracking for per-sample indexing
        self._chunk_paths = []
        self._builder = None

        self._background_model = None

        self.meta_path = os.path.join(index_dir, 'meta.json')

        if rewrite:
            if os.path.exists(index_dir):
                shutil.rmtree(index_dir)
            os.makedirs(index_dir, exist_ok=True)
        elif self._index_exists():
            logging.info("Prefix index exists. Loading metadata.")
            self._load_metadata()
        else:
            logging.info(f"Will create prefix index at `{index_dir}` with {kmer_size_initialize}-mers")
            os.makedirs(index_dir, exist_ok=True)

    def _index_exists(self):
        return (os.path.exists(os.path.join(self.index_dir, 'meta.json')) and
                os.path.exists(os.path.join(self.index_dir, 'pi.bin')))

    def _load_metadata(self):
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        self.kmer_size = meta['kmer_size']
        self.l_prefix = meta['l_prefix']
        self.l_suffix = meta['l_suffix']
        self.n_spatial = meta.get('n_cells', 0)
        self.HAS_MERGED_PROJECTS = meta.get('merge_projects', False)
        if 'project_id_shift' in meta:
            self.PROJECT_ID_SHIFT = meta['project_id_shift']
        if 'cell_id_mask' in meta:
            self.CELL_ID_MASK = meta['cell_id_mask']
        if 'project_mapping' in meta:
            self.project_mapping = {int(k): v for k, v in meta['project_mapping'].items()}

    def set_spatial_index(self, sindex):
        """Set a spatial index (with coordinates)."""
        self.spatial_index = sindex
        self.set_spatial_coords(sindex.get_coords())

    def set_barcode_index(self, sindex):
        """Set a barcode-only index (single-cell, no spatial coords)."""
        self.spatial_index = sindex
        self.n_spatial = sindex.num_items()
        self._save_metadata()

    def set_spatial_coords(self, coords):
        """Store spatial coordinates."""
        xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
        ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
        self.coord_lims = (xmin, xmax, ymin, ymax)
        self.n_spatial = len(coords)
        np.save(os.path.join(self.index_dir, 'spatial_coord.npy'), coords)
        self._save_metadata()

    def _save_metadata(self):
        """Save/update metadata to meta.json, preserving fields from the index build."""
        meta = {}
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'r') as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        meta['kmer_size'] = self.kmer_size
        meta['l_prefix'] = self.l_prefix
        meta['l_suffix'] = self.l_suffix
        meta['n_cells'] = int(self.n_spatial)
        meta['n_prefixes'] = int(1 << (2 * self.l_prefix))
        meta['merge_projects'] = self.HAS_MERGED_PROJECTS
        meta['project_id_shift'] = self.PROJECT_ID_SHIFT
        meta['cell_id_mask'] = int(self.CELL_ID_MASK)
        if self.coord_lims:
            meta['coord_lims'] = [float(x) for x in self.coord_lims]
        if self.project_mapping:
            meta['project_mapping'] = {str(k): v for k, v in self.project_mapping.items()}
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def add_reads(self, reads_in, bam_tags='CB:{cell}', cell='r1[2:27]',
                  n_report=10_000_000, chunksize=100_000_000, threads=1):
        """
        Add reads from FASTQ files and build the index.

        """
        from malva.utils import check_cell_string
        from malva.indexes import process_fastq_reads, build_from_sorted_chunks

        read_group, start, end = check_cell_string(cell)
        is_bulk = isinstance(reads_in[0], int)

        chunk_paths = process_fastq_reads(
            reads_in=reads_in,
            output_dir=self.index_dir,
            spatial_index=self.spatial_index,
            kmer_size=self.kmer_size,
            l_prefix=self.l_prefix,
            jump_amount=self.jump_amount,
            trim_start=start,
            trim_end=end,
            chunksize=chunksize,
            n_report=n_report,
            threads=threads,
            is_bulk=is_bulk,
        )

        self._chunk_paths = chunk_paths
        self._build_from_chunks()
        
        self._save_metadata()

    def _build_from_chunks(self):
        """Build the index from sorted chunk files."""
        from malva.indexes import build_from_sorted_chunks

        if len(self._chunk_paths) == 0:
            logging.info("Index built directly (no chunk merge needed)")
            chunk_dir = os.path.join(self.index_dir, '_chunks')
            if os.path.exists(chunk_dir):
                shutil.rmtree(chunk_dir)
            return

        logging.info(f"Merging {len(self._chunk_paths)} chunks with n_cells={self.n_spatial}")
        build_from_sorted_chunks(
            self._chunk_paths,
            self.index_dir,
            kmer_size=self.kmer_size,
            l_prefix=self.l_prefix,
            n_cells=self.n_spatial,
            verbose=self.verbose,
        )

        meta_check_path = os.path.join(self.index_dir, 'meta.json')
        if os.path.exists(meta_check_path):
            import json as _json
            with open(meta_check_path) as _f:
                _m = _json.load(_f)
            logging.info(f"Post-merge meta.json: n_kmers={_m.get('n_kmers',0):,}, n_cells={_m.get('n_cells',0):,}")
        
        pi_check = os.path.join(self.index_dir, 'pi.bin')
        suf_check = os.path.join(self.index_dir, 'suffixes.bin')
        dat_check = os.path.join(self.index_dir, 'data.bin')
        logging.info(f"Post-merge files: PI={os.path.getsize(pi_check)/1e6:.1f}MB "
                     f"Suf={os.path.getsize(suf_check)/1e6:.1f}MB "
                     f"Data={os.path.getsize(dat_check)/1e6:.1f}MB")

        chunk_dir = os.path.join(self.index_dir, '_chunks')
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)

        self._chunk_paths = []

    def merge_chunks(self, file_out, merge_projects=False, sample_percentage=0.05):
        """
        Merge this index with itself (multi-chunk scenario) or merge
        multiple sample indices. 
        
        """
        if len(self._chunk_paths) > 0:
            self._build_from_chunks()
        else:
            logging.info("Index already built, nothing to merge")

    def open(self, mode='r', blosc_load_to_memory=False):
        """Open the index for querying."""
        if self._index_exists():
            self._load_metadata()

            coord_path = os.path.join(self.index_dir, 'spatial_coord.npy')
            if os.path.exists(coord_path):
                self.spatial_coord = np.load(coord_path)

        else:
            logging.warning("Index does not exist yet")

    def close(self):
        """Close the index."""
        if self._prefix_index is not None:
            self._prefix_index.close()
            self._prefix_index = None

    def _ensure_index_open(self):
        """Ensure the PrefixIndex is open for queries."""
        if self._prefix_index is None or not self._prefix_index.is_loaded:
            from malva.indexes import PrefixIndex as CyPrefixIndex
            self._prefix_index = CyPrefixIndex()
            self._prefix_index.open(self.index_dir)

    def find_kmer(self, kmers, count_at_most=10000, count_at_least=10,
                  chunk_id=0, use_batched=False):
        """
        Find kmers in the index. Returns dict of kmer -> np.array(uint32 cell IDs).
        """
        self._ensure_index_open()

        if not isinstance(kmers, np.ndarray):
            kmers = np.array(kmers, dtype=np.uint64)
        
        if len(kmers) == 0:
            return {}

        result = self._prefix_index.query(
            kmers,
            count_at_most=count_at_most,
            count_at_least=count_at_least,
        )
        
        return result

    def load_index_to_memory(self, chunk_id=0, chunk_size=50_000_000,
                             max_mem=None, force=False,
                             count_at_most=10000, count_at_least=10):
        """
        """
        self._ensure_index_open()

    def set_background_model(self, background_model):
        self._background_model = background_model

    def where(self, sequence, sliding_size=128, pct_threshold=0.65,
              count_at_most=10_000, count_at_least=10, chunk_id=0,
              single_count=False, max_mem='1M', force_reload=False,
              use_background_model=True, use_batched=False, *args, **kwargs):
        """
        Locate spatial positions where a sequence or set of sequences appear.
        
        """
        from malva.indexes import quantify_where

        if pct_threshold < 0 or pct_threshold > 1:
            raise ValueError("`pct_threshold` must be between 0 and 1")

        if isinstance(sequence, str):
            sequence_groups = [[sequence]]
        elif isinstance(sequence, list) and all(isinstance(s, str) for s in sequence):
            sequence_groups = [sequence]
        elif isinstance(sequence, list) and all(isinstance(s, list) for s in sequence):
            sequence_groups = sequence
        else:
            raise ValueError("sequence must be str, List[str], or List[List[str]]")

        self._ensure_index_open()

        logging.info(f"quantify_where: n_spatial={self.n_spatial}, kmer_size={self.kmer_size}, "
                     f"sliding_size={sliding_size}, n_groups={len(sequence_groups)}")

        return quantify_where(
            self._prefix_index,
            sequence_groups,
            self.kmer_size,
            sliding_size=sliding_size,
            pct_threshold=pct_threshold,
            count_at_most=count_at_most,
            count_at_least=count_at_least,
            max_cell_id=self.n_spatial,
            single_count=single_count,
            background_model=self._background_model,
            use_background_model=use_background_model,
            verbose=self.verbose,
        )

    def get_project_id(self, cell_id):
        if self.HAS_MERGED_PROJECTS:
            return cell_id >> self.PROJECT_ID_SHIFT
        return cell_id

    def get_cell_id(self, cell_id):
        if self.HAS_MERGED_PROJECTS:
            return cell_id & self.CELL_ID_MASK
        return cell_id

    @property
    def n_chunks(self):
        """Number of index chunks (always 1 for a built index)."""
        return 1 if self._index_exists() else 0

    @property
    def index_file(self):
        """Path to the index metadata file."""
        return self.meta_path

    @property
    def index(self):
        """Dict-like accessor for index metadata keys."""
        keys = set()
        if self.spatial_coord is not None:
            keys.add('spatial_coord')
        return _IndexCompat(keys, self)

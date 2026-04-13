# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

"""Legacy HDF5-based MalvaIndex — read-only wrapper for index conversion.

This module re-implements the minimal interface of the original Cython
``MalvaIndex`` (``indexes.pyx`` before the prefix-bucketed refactor) as
plain Python so that :mod:`malva.convert` can read old on-disk indices
without requiring the old compiled extension.

Only the attributes and methods used by ``convert_malva_to_prefix`` and
``verify_indices`` are implemented.  Write / indexing operations are **not**
supported.
"""

import json
import logging
import math
import os

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class MalvaIndex:
    """Read-only wrapper around a legacy HDF5-split MalvaIndex directory.

    Parameters
    ----------
    index_dir:
        Directory that contains ``malva_index.h5-m.h5`` and
        ``malva_index.h5-r.h5`` (the HDF5 split-driver pair produced by
        the old indexing pipeline).
    """

    def __init__(self, index_dir: str, *, max_project_capacity: int = 512):
        self.index_dir = index_dir
        self.index_file = os.path.join(index_dir, "malva_index.h5")

        project_bits = int(math.ceil(math.log2(max_project_capacity)))
        self.PROJECT_ID_SHIFT: int = 32 - project_bits
        self.CELL_ID_MASK: int = (1 << (32 - project_bits)) - 1
        self.HAS_MERGED_PROJECTS: bool = False

        self.kmer_size: int = 24
        self.n_spatial: int = 0
        self.n_chunks: int = 0
        self.coord_lims = None
        self.spatial_coord = None
        self.project_mapping = None
        self.data_lengths = []

        # The live h5py.File handle — set by open(), cleared by close().
        self.index = None

    # ------------------------------------------------------------------
    # File lifecycle
    # ------------------------------------------------------------------

    def open(self, mode: str = "r"):
        """Open the underlying HDF5 file.

        Parameters
        ----------
        mode:
            Passed to :func:`h5py.File`.  For conversion use ``'r'``.
        """
        self.index = h5py.File(self.index_file, mode, driver="split")

        attrs = self.index.attrs

        if "kmer_size" in attrs:
            self.kmer_size = int(attrs["kmer_size"])

        if "coord_lims" in attrs:
            self.coord_lims = tuple(attrs["coord_lims"])

        if "n_spatial" in attrs:
            self.n_spatial = int(attrs["n_spatial"])

        if "n_chunks" in attrs:
            self.n_chunks = int(attrs["n_chunks"])
            self.data_lengths = [
                len(self.index[f"index_{c}_data"]) for c in range(self.n_chunks)
            ]

        if "project_id_shift" in attrs:
            self.PROJECT_ID_SHIFT = int(attrs["project_id_shift"])

        if "cell_id_mask" in attrs:
            self.CELL_ID_MASK = int(attrs["cell_id_mask"])

        if "has_merged_projects" in attrs:
            self.HAS_MERGED_PROJECTS = bool(attrs["has_merged_projects"])

        if "spatial_coord" in self.index:
            self.spatial_coord = self.index["spatial_coord"]

        if "project_mapping" in attrs:
            raw = attrs["project_mapping"]
            self.project_mapping = {int(k): v for k, v in json.loads(raw).items()}

        logger.debug(
            "Opened legacy index at %s  "
            "(kmer_size=%d, n_spatial=%d, n_chunks=%d)",
            self.index_dir,
            self.kmer_size,
            self.n_spatial,
            self.n_chunks,
        )

    def close(self):
        """Flush and close the HDF5 file."""
        if self.index is not None:
            self.index.flush()
            self.index.close()
            self.index = None

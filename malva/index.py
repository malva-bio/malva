# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

import logging
import os
import yaml

from malva.barcodes import SpatialIndex, create_spatial_index, create_singlecell_index
from malva.malva_index import MalvaIndex
from malva.utils import check_directory_exists, check_file_exists, get_module_path

N_REPORT = 1_000_000


def load_flavor(flavor, flavors_config_path):
    """Load a barcode flavor configuration."""
    if flavor.lower().endswith(".yaml"):
        if not os.path.isfile(flavor):
            raise FileNotFoundError(f"Custom flavor file '{flavor}' not found.")
        with open(flavor, "r") as stream:
            return yaml.safe_load(stream)
    else:
        with open(flavors_config_path, "r") as stream:
            flavor_config = yaml.safe_load(stream)

        if flavor not in flavor_config["barcode_flavors"]:
            raise ValueError(f"Flavor `{flavor}` could not be found")

        return flavor_config["barcode_flavors"][flavor]


def _run_index(args):
    """Build a prefix-bucketed index from FASTQ files."""
    if args.flavor != 'bulk':
        for _r in args.reads_in:
            check_file_exists(_r, except_when=False)
    else:
        check_file_exists(args.reads_in[1], except_when=False)

    if not check_directory_exists(args.index_out):
        logging.info("Output directory does not exist. Creating...")
        os.makedirs(args.index_out, exist_ok=True)

    logging.info(f"Configuring flavor `{args.flavor}`")
    _config_path = os.path.join(get_module_path(), "data", "config.yaml")
    flavor_config = load_flavor(args.flavor, _config_path)
    _bam_tags = flavor_config["bam_tags"]
    _cell = flavor_config["cell"]

    _sindex_loc = os.path.join(args.index_out, "sindex.bin")
    _sindex_exists = check_file_exists(_sindex_loc)

    if _sindex_exists:
        logging.info("Loading previously created barcode->coordinate index")
        sindex = SpatialIndex()
        if args.flavor == "stereo_seq":
            sindex.load_binary_stomics(_sindex_loc)
        else:
            sindex.load_binary(_sindex_loc)
    elif args.flavor.startswith("sc_") or args.flavor == 'bulk':
        logging.info("Creating barcode single-cell index")
        sindex = create_singlecell_index(args.spatial_bc_in)
    else:
        if args.flavor == "stereo_seq":
            raise ValueError("STOmics indices must be provided in .bin format!")
        logging.info("Creating spatial barcode->coordinate index")
        sindex = create_spatial_index(args.spatial_bc_in)
        logging.info("Saving spatial index")
        sindex.save_binary(_sindex_loc)

    l_prefix = min(args.kmer_length // 2, 12)
    if hasattr(args, 'l_prefix') and args.l_prefix > 0:
        l_prefix = args.l_prefix

    jump_amount = 1 if getattr(args, 'overlapping', False) else args.kmer_length
    kmer_index = MalvaIndex(
        args.index_out,
        kmer_size_initialize=args.kmer_length,
        l_prefix=l_prefix,
        jump_amount=jump_amount,
        verbose=True,
    )

    if args.flavor == "stereo_seq":
        kmer_index.set_barcode_index(sindex)
        kmer_index.set_spatial_coords(sindex.get_coords_stomics())
    elif args.flavor.startswith("sc_") or args.flavor == 'bulk':
        kmer_index.set_barcode_index(sindex)
    else:
        kmer_index.set_spatial_index(sindex)

    if args.flavor == 'bulk':
        try:
            args.reads_in[0] = int(args.bulk_id)
        except Exception:
            logging.error("Could not set the bulk identifier to a number")
            exit(1)

    logging.info(f"Indexing {args.kmer_length}-mers from {args.reads_in}")
    logging.info(f"  l_prefix={l_prefix}, l_suffix={args.kmer_length - l_prefix}")
    logging.info(f"  Writing chunks every {args.chunksize:,} sequences")

    kmer_index.add_reads(
        args.reads_in,
        _bam_tags,
        _cell,
        chunksize=args.chunksize,
        threads=args.threads,
    )

    logging.info("SUCCESS!")


if __name__ == "__main__":
    from malva.cli import get_index_parser
    args = get_index_parser().parse_args()
    _run_index(args)

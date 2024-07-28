import logging
import os
import yaml

from malva.utils import check_file_exists, check_directory_exists, get_module_path
from malva.indexes import MalvaIndex, SpatialIndex, create_spatial_index

N_REPORT = 1_000_000


def load_flavor(flavor, flavors_config_path):
    if flavor.lower().endswith('.yaml'):
        if not os.path.isfile(flavor):
            raise FileNotFoundError(f"Custom flavor file '{flavor}' not found.")
        with open(flavor, 'r') as stream:
            try:
                custom_flavor_config = yaml.safe_load(stream)
                return custom_flavor_config
            except yaml.YAMLError as exc:
                raise yaml.YAMLError(f"Error parsing custom flavor file: {exc}")
    else:
        with open(flavors_config_path, 'r') as stream:
            try:
                flavor_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise yaml.YAMLError(f"Error parsing flavors config file: {exc}")
        
        if flavor not in flavor_config['barcode_flavors']:
            raise ValueError(f"Flavor `{flavor}` could not be found")
        
        current_flavor_config = flavor_config['barcode_flavors'][flavor]
        return current_flavor_config


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
    _bam_tags = flavor_config['bam_tags']
    _cell = flavor_config['cell']
    
    _sindex_loc = os.path.join(args.index_out, "sindex.bin")
    _sindex_exists = check_file_exists(_sindex_loc)

    if _sindex_exists:
        logging.info("Loading previously created `cell (spot) barcode->spatial coordinate` index")
        sindex = SpatialIndex()
        sindex.load_binary(_sindex_loc)
    else:
        logging.info("Creating `cell (spot) barcode->spatial coordinate` index")
        sindex = create_spatial_index(args.spatial_bc_in, args.rescale_coords, args.index_resolution, not args.no_recenter)
        logging.info("Saving `cell (spot) barcode->spatial coordinate` index")
        sindex.save_binary(_sindex_loc)

    logging.info(f"Configuring the malva index")
    kmer_index = MalvaIndex(args.index_out, kmer_size_initialize=args.kmer_length)
    
    logging.info("Adding spatial index to malva index")
    kmer_index.append_spatial(sindex)

    logging.info(f"Indexing sequence {args.kmer_length}-mers in space from {args.reads_in} with flavor {args.flavor}")
    logging.info(f"Will write to disk every {args.chunksize:,} sequences, and once at the end (remaining sequences)")
    kmer_index.add_reads(args.reads_in, _bam_tags, _cell, chunksize=args.chunksize, threads=args.threads)
    
    logging.info("SUCCESS!")


if __name__ == "__main__":
    from malva.cli import get_index_parser
    args = get_index_parser().parse_args()
    _run_index()
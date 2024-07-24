import argparse
import logging

INDEX_HELP = "Build a katoste index from spatial transcriptomic sequencing reads"
def get_index_parser():
    parser = argparse.ArgumentParser(
        description=INDEX_HELP,
        allow_abbrev=False,
        add_help=False,
    )

    parser.add_argument(
        "--reads-in",
        type=str,
        required=True,
        nargs="+",
        help="""FASTQ(s) or BAM file containing the transcriptomic information, 
        UMI and cell (spatial) barcode (in R1/R2 structure, or in a single file)""",
    )
    parser.add_argument(
        "--spatial-bc-in",
        type=str,
        required=True,
        help="""Tabular file containing columns BC,X,Y:
        BC: the cell (spatial) barcode sequence
        X: x spatial coordinate (any units)
        Y: y spatial coordinate (any units)""",
    )
    parser.add_argument(
        "--index-out",
        type=str,
        required=True,
        help="""Valid directory where the katoste index (and metadata) will be written into.
        If the directory exists, it must not contain files called `index.kst` and `index_metadata.json`.
        Otherwise, an exception will be thrown.""",
    )
    parser.add_argument(
        "--flavor",
        type=str,
        default="openst",
        choices=['openst', 'stereo_seq', 'slide_seq', 'visium', 'seq_scope_v1', '*.yaml'], # TODO: formatting of the .yaml is acceptable, or move validation to the function
        help="""Spatial transcriptomics technology. 
        These are default configurations to read from the paired FASTQ (or BAM) files.
        Other configurations can be provided as a properly formatted `.yaml` file - see
        documentation.""",
    )
    parser.add_argument(
        "--kmer-length",
        type=int,
        default=24,
        help="Length (in nucleotides) of indexed k-mers, non-overlapping.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000_000,
        help="""Consecutive chunk that will be accumulated into RAM before writing.
        Consider reducing this number to reduce RAM usage (indexing might be slower).""",
    )
    parser.add_argument(
        "--overlapping",
        action="store_true",
        help="""By default, the index stores non-overlapping k-mers.
        With this option, overlapping k-mers are indexed, increasing
        sensitivity against mutation events during query time, but also
        increases time to build the index and its size.""",
    )
    parser.add_argument(
        "--rescale-coords",
        type=float,
        default=1,
        help="Factor to convert from provided coordinate units into microns",
    )
    parser.add_argument(
        "--index-resolution",
        type=float,
        default=10,
        help="Spatial resolution of the katoste index, in microns.",
    )
    parser.add_argument(
        "--no-recenter",
        action="store_true",
        help="""Spatial coordinates will be kept 'as is'. 
        The x,y coordinate values must be in the range 0-65,535.
        Otherwise, an exception will be thrown.""",
    )
    parser.add_argument(
        "--threads",
        type=float,
        default=1,
        help="""Number of threads used for parallel processing""",
    )
    return parser


def setup_index_parser(parent_parser):
    parser = parent_parser.add_parser(
        "index",
        help=INDEX_HELP,
        parents=[get_index_parser()],
    )
    parser.set_defaults(func=cmd_run_index)

    return parser


def cmd_run_index(args):
    from katoste.index import _run_index

    _run_index(args)


SHOW_HELP = "Query a DNA/RNA sequence against a katoste index and visualize spatial distribution"
def get_show_parser():
    parser = argparse.ArgumentParser(
        description=SHOW_HELP,
        allow_abbrev=False,
        add_help=False,
    )

    parser.add_argument(
        "--index-in",
        type=str,
        help="""Valid directory where the katoste index (and metadata) is located.
        The directory must contain the file `index.kst` and `index_metadata.json`.
        Otherwise, an exception will be thrown.""",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="""FASTA file containing the query sequences.""",
    )
    parser.add_argument(
        "--image-out",
        type=str,
        required=True,
        help="""Directory where the image results will be saved into. 
        
        One image in TIFF format will be created per query sequence, under the directory specified in 
        Filenames are generated from the FASTA header per sequence.""",
    )
    parser.add_argument(
        "--multichannel",
        action="store_true",
        help="""Will save a single image where channels are the individual query sequences (named)"""
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="""Additionally to TIFF images, the coordinates of spots and the amount of 
        signal is stored as a N_SPOTS-by-(X, Y, INTENSITY) pickled numpy array.""",
    )
    parser.add_argument(
        "--scalebar",
        action="store_true",
        help="""A scalebar is automatically displayed.
        The size is by default 25/100 of the image width.""",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=1,
        help="What is the scale, respect to the original index spatial dimensions per unit, used for rendering.",
    )
    parser.add_argument(
        "--render-smoothing",
        type=float,
        default=1.5,
        help="Sigma value for gaussian smoothing of pseudoimages (for rendering purposes)",
    )
    return parser


def setup_show_parser(parent_parser):
    parser = parent_parser.add_parser(
        "show",
        help=SHOW_HELP,
        parents=[get_show_parser()],
    )
    parser.set_defaults(func=cmd_run_show)

    return parser


def cmd_run_show(args):
    from katoste.show import _run_show

    _run_show(args)


SERVE_HELP = "Webserver for interactive spatial querying of katoste indexes"
def get_serve_parser():
    parser = argparse.ArgumentParser(
        description=SERVE_HELP,
        allow_abbrev=False,
        add_help=False,
    )

    parser.add_argument(
        "--index-in",
        type=str,
        help="""Valid directory where the katoste index (and metadata) is located.
        The directory must contain the file `index.kst` and `index_metadata.json`.
        Otherwise, an exception will be thrown.""",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="""Port where the webserver will be listening to""",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="127.0.0.1",
        help="""Address where the webserver will be available at""",
    )
    parser.add_argument(
        "--lazy-index",
        action="store_true",
        help="""No part of the index will be loaded to main memory (full-disk queries)""",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1000,
        help="""Maximum allowed length for DNA/RNA queries""",
    )
    return parser


def setup_serve_parser(parent_parser):
    parser = parent_parser.add_parser(
        "serve",
        help=SERVE_HELP,
        parents=[get_serve_parser()],
    )
    parser.set_defaults(func=cmd_run_serve)

    return parser


def cmd_run_serve(args):
    from katoste.serve.serve import _run_serve

    _run_serve(args)


def cmdline_args():
    parent_parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="katoste: fast indexing and querying of genomic sequences from spatial transcriptomics data",
    )

    parent_parser_subparsers = parent_parser.add_subparsers(title="commands", dest="subcommand")
    parent_parser.add_argument(
    '--version',
    action = 'store_true')

    setup_index_parser(parent_parser_subparsers)
    setup_show_parser(parent_parser_subparsers)
    setup_serve_parser(parent_parser_subparsers)

    parsed_args = parent_parser.parse_args()

    return parent_parser, parsed_args


def cmdline_main():
    import importlib.metadata
    import setproctitle
    import sys

    setproctitle.setproctitle('katoste ' + " ".join(sys.argv[1:]))

    parser, args = cmdline_args()

    if args.version and args.subcommand is None:
        print(importlib.metadata.version('katoste'))
        return 0
    else:
        del args.version

    if "func" in args:
        logging.info(f"katoste {args.subcommand} - running with the following parameters:")
        logging.info(args.__dict__)
        args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    cmdline_main()

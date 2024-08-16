import argparse
import logging

INDEX_HELP = "Build a malva index from spatial transcriptomic sequencing reads"


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
        nargs=2,
        help="""Pair of FASTQ files containing the transcriptomic information, 
        UMI and cell (spatial) barcode (in R1/R2 structure, paired-end)""",
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
        help="""Valid directory where the malva index (and metadata) will be written into.
        If the directory exists, it must not contain files called `malva_index.h5`.
        Otherwise, an exception will be thrown.""",
    )
    parser.add_argument(
        "--flavor",
        type=str,
        default="openst",
        help="""Spatial transcriptomics technology. 
        These are default configurations to read from the paired FASTQ (or BAM) files.
        Other configurations can be provided as a properly formatted `.yaml` file - see
        documentation.
        
        Currently, flavors 'openst', 'stereo_seq', 'slide_seq', 'visium', 
        'seq_scope_v1', 'sc_10x_v1', 'sc_10x_v3' or a path to a .yaml file, are supported""",
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
        default=100_000_000,
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
        "--merge-chunks",
        action="store_true",
        help="""When the chunk size is less than the number of total reads, there will be
        several separate chunks in the index file. When this option is provided, the different
        chunks are merged into a single one, which will reduce index size and improve query speed.
        This adds a bit of time to the overall processing.""",
    )
    parser.add_argument(
        "--threads",
        type=int,
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
    from malva.index import _run_index

    _run_index(args)


SHOW_HELP = "Query a DNA/RNA sequence against a malva index and visualize spatial distribution"


def get_show_parser():
    parser = argparse.ArgumentParser(
        description=SHOW_HELP,
        allow_abbrev=False,
        add_help=False,
    )

    parser.add_argument(
        "--index-in",
        type=str,
        help="""Valid directory where the malva index (and metadata) is located.
        The directory must contain the file `malva_index.h5`.
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
        help="""Will save a single image where channels are the individual query sequences (named)""",
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
    from malva.show import _run_show

    _run_show(args)


QUANT_HELP = "Pseudo-quantification of gene expression profiles"


def get_quant_parser():
    parser = argparse.ArgumentParser(
        description=QUANT_HELP,
        allow_abbrev=False,
        add_help=False,
    )

    parser.add_argument(
        "--index-in",
        type=str,
        help="""Valid directory where the malva index (and metadata) is located.
        The directory must contain the file `malva_index.h5`.
        Otherwise, an exception will be thrown.""",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=False,
        default="human_utr",
        choices=["human_utr", "mouse_utr", "mouse_cdna", "mouse_utr_ncrna", "mouse_cdna_ncrna"],
        help="""Reference used for pseudoquantification. Options available: 'human_utr', 'mouse_utr', 'mouse_cdna', 'mouse_utr_ncrna', 'mouse_cdna_ncrna'. Default: 'human_utr'""",
    )
    parser.add_argument(
        "--background-model",
        type=str,
        required=False,
        default=None,
        help="""Path to background model (*.bmodel.bin) of k-mer abundance (those too abundant will be ignored during query, e.g., multimappers). Default: None""",
    )
    parser.add_argument(
        "--folder-out",
        type=str,
        required=True,
        help="""Directory where the gene expression pseudoquantification.
        
        Three files will be created, similar to cellranger output: 
        barcodes.txt.gz, features.txt.gz, and matrix.mtx""",
    )
    parser.add_argument(
        "--h5ad",
        action="store_true",
        help="""When specified, Resaves matrix.mtx to AnnData format, at --folder-out.""",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        required=False,
        default=0,
        help="""Aggregates spatial units from the AnnData file into bins (when --h5ad is specified). Default: 0 (no binning)""",
    )
    parser.add_argument(
        "--sliding-size",
        type=int,
        required=False,
        default=128,
        help="""Quantification sliding window size; should match average indexed read length. Default: 128""",
    )
    parser.add_argument(
        "--pct-threshold",
        type=float,
        required=False,
        default=0.65,
        help="""Percentage of indexed k-mers (0-1 range) that should match per coordinates for considering match. Default: 65 (percent)""",
    )
    parser.add_argument(
        "--kmer-min",
        type=int,
        required=False,
        default=10,
        help="""k-mers occurring less than --kmer-min times are ignored""",
    )
    parser.add_argument(
        "--kmer-max",
        type=int,
        required=False,
        default=10_000,
        help="""k-mers occurring more than --kmer-max times are ignored""",
    )
    parser.add_argument(
        "--single-count",
        action="store_true",
        help="""When specified, malva counts whether the query sequence was found or not
        for a specific sequence.""",
    )
    return parser


def setup_quant_parser(parent_parser):
    parser = parent_parser.add_parser(
        "quant",
        help=QUANT_HELP,
        parents=[get_quant_parser()],
    )
    parser.set_defaults(func=cmd_run_quant)

    return parser


def cmd_run_quant(args):
    from malva.quant import _run_quant

    _run_quant(args)


CELLXMER_HELP = "Convert the malva index to a cell-by-mer AnnData file that can be used for clustering and sequence assembly"


def get_cellxmer_parser():
    parser = argparse.ArgumentParser(
        description=CELLXMER_HELP,
        allow_abbrev=False,
        add_help=False,
    )

    parser.add_argument(
        "--index-in",
        type=str,
        help="""Valid directory where the malva index (and metadata) is located.
        The directory must contain the file `malva_index.h5`.
        Otherwise, an exception will be thrown.""",
    )
    parser.add_argument(
        "--h5ad-out",
        type=str,
        required=True,
        help="""Directory where the cell-by-mer object will be stored.""",
    )
    parser.add_argument(
        "--kmer-min",
        type=int,
        required=False,
        default=10,
        help="""k-mers occurring less than --kmer-min times are ignored""",
    )
    parser.add_argument(
        "--kmer-max",
        type=int,
        required=False,
        default=10_000,
        help="""k-mers occurring more than --kmer-max times are ignored""",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        required=False,
        default=0,
        help="""Aggregates spatial units at the final AnnData file to reduce dimensions of final cell-by-kmer matrix""",
    )
    return parser


def setup_cellxmer_parser(parent_parser):
    parser = parent_parser.add_parser(
        "cellxmer",
        help=CELLXMER_HELP,
        parents=[get_cellxmer_parser()],
    )
    parser.set_defaults(func=cmd_run_cellxmer)

    return parser


def cmd_run_cellxmer(args):
    from malva.cellxmer import _run_cellxmer

    _run_cellxmer(args)


COMBINE_HELP = "Combine various sub-indexes from the same sample (e.g., processed in parallel) into a single index"


def get_combine_parser():
    parser = argparse.ArgumentParser(
        description=COMBINE_HELP,
        allow_abbrev=False,
        add_help=False,
    )

    parser.add_argument(
        "--index-in",
        type=str,
        help="""Valid directory where the malva indices are located as subdirectories.
        The new combined index (and metadata) will be stored at this folder.""",
    )
    parser.add_argument(
        "--merge-chunks",
        action="store_true",
        help="""When the chunk size is less than the number of total reads, there will be
        several separate chunks in the index file. When this option is provided, the different
        chunks are merged into a single one, which will reduce index size and improve query speed.
        This adds a bit of time to the overall processing.""",
    )
    return parser


def setup_combine_parser(parent_parser):
    parser = parent_parser.add_parser(
        "combine",
        help=COMBINE_HELP,
        parents=[get_combine_parser()],
    )
    parser.set_defaults(func=cmd_run_combine)

    return parser


def cmd_run_combine(args):
    from malva.combine import _run_combine

    _run_combine(args)


SERVE_HELP = "Webserver for interactive spatial querying of malva indexes"


def get_serve_parser():
    parser = argparse.ArgumentParser(
        description=SERVE_HELP,
        allow_abbrev=False,
        add_help=False,
    )

    parser.add_argument(
        "--index-in",
        type=str,
        help="""Valid directory where the malva index (and metadata) is located.
        The directory must contain the file `malva_index.h5`.
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
    parser.add_argument(
        "--rescale-coords",
        type=float,
        default=1,
        help="Provided coordinate units (from the index) are rescaled by this factor",
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
    from malva.serve.serve import _run_serve

    _run_serve(args)


def cmdline_args():
    parent_parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="malva: fast indexing and querying of genomic sequences from spatial transcriptomics data",
    )

    parent_parser_subparsers = parent_parser.add_subparsers(title="commands", dest="subcommand")
    parent_parser.add_argument("--version", action="store_true")

    setup_index_parser(parent_parser_subparsers)
    setup_show_parser(parent_parser_subparsers)
    setup_serve_parser(parent_parser_subparsers)
    setup_quant_parser(parent_parser_subparsers)
    setup_cellxmer_parser(parent_parser_subparsers)
    setup_combine_parser(parent_parser_subparsers)

    parsed_args = parent_parser.parse_args()

    return parent_parser, parsed_args


def cmdline_main():
    import importlib.metadata
    import sys

    import setproctitle

    setproctitle.setproctitle("malva " + " ".join(sys.argv[1:]))

    parser, args = cmdline_args()

    if args.version and args.subcommand is None:
        print(importlib.metadata.version("malva"))
        return 0
    else:
        del args.version

    if "func" in args:
        logging.info(f"malva {args.subcommand} - running with the following parameters:")
        logging.info(args.__dict__)
        args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    cmdline_main()

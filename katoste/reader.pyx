# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import pysam

from xopen import xopen
from katoste.utils import safety_check_eval
from katoste.fastq_processing import SequenceFastqParser, KmerFastqParser
from cython cimport boundscheck, wraparound

ctypedef unsigned int uint32_t

def iterate_flavor(list reads_in,
                   str bam_tags='CB:{cell}',
                   str cell='r1[2:27]'):
    cdef dict bam_tags_dict
    cdef int num_reads = len(reads_in)

    # Parse BAM tags into a dictionary
    bam_tags_dict = {bt.split(":")[1][1:-1]: bt.split(":")[0] for bt in bam_tags.split(",")}

    if num_reads == 2:
        # For paired FASTQ files
        assert safety_check_eval(cell)
        f_cell = compile(cell, "<string cell>", "eval")

        # TODO: update trim_start and trim_end so it uses the config provided by the user
        # TODO: provide kmer size
        iter_r1 = SequenceFastqParser(xopen(reads_in[0], "rb"), 512 * 1024, trim_start = 2, trim_end = 27)
        iter_r2 = KmerFastqParser(xopen(reads_in[1], "rb"), 512 * 1024, kmer_size = 24)
        
        for r1, r2 in zip(iter_r1, iter_r2):
            yield r1, r2

    elif num_reads == 1:
        # For a single BAM file
        _cell_tag = bam_tags_dict['cell']
        with pysam.AlignmentFile(reads_in[0], "rb") as f:
            for record in f.fetch(until_eof=True):
                cell_bc = record.get_tag(_cell_tag)
                yield (record.query_sequence, cell_bc)
    else:
        raise ValueError("`--reads-in` must point to paired FASTQ files or a single BAM file")
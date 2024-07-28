# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import io

from xopen import xopen
from malva.utils import safety_check_eval
from malva.fastq_processing import SequenceFastqParser, KmerFastqParser
from malva.utils import check_cell_string

ctypedef unsigned int uint32_t

cdef int BUFFER_SIZE = max(io.DEFAULT_BUFFER_SIZE, 128 * 1024)

def iterate_flavor(list reads_in,
                   str bam_tags='CB:{cell}',
                   str cell='r1[2:27]',
                   int kmer_size=24,
                   int threads=1):
    cdef dict bam_tags_dict
    cdef int num_reads = len(reads_in)
    read_group, start, end = check_cell_string(cell)
    
    if read_group != 'r1':
            raise NotImplementedError("Only 'r1' is implemented")

    # Parse BAM tags into a dictionary
    bam_tags_dict = {bt.split(":")[1][1:-1]: bt.split(":")[0] for bt in bam_tags.split(",")}

    if num_reads == 2:
        # For paired FASTQ files
        assert safety_check_eval(cell)
        f_cell = compile(cell, "<string cell>", "eval")

        iter_r1 = SequenceFastqParser(xopen(reads_in[0], "rb", threads=max(threads//2, 1)), BUFFER_SIZE, trim_start = start, trim_end = end)
        iter_r2 = KmerFastqParser(xopen(reads_in[1], "rb", threads=max(threads//2, 1)), BUFFER_SIZE, kmer_size = kmer_size)
        
        for r1, r2 in zip(iter_r1, iter_r2):
            yield r1, r2

    else:
        raise ValueError("`--reads-in` must point to paired FASTQ files")


def iterate_fasta(filename):
    try:
        import dnaio
    except ImportError:
        raise ImportError("Please install dnaio: `pip install dnaio`")

    with dnaio.open(filename) as reader:
        for record in reader:
            yield record.name, record.sequence
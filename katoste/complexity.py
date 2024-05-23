from math import log
import numpy as np

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def overlapping_windows(sequence, L):
    """
    Returns overlapping windows of size `L` from sequence `sequence`
    :param sequence: the nucleotide or protein sequence to scan over
    :param L: the length of the windows to yield
    """
    windows = []

    for index, residue in enumerate(sequence):
        if (index + L) < (len(sequence) + 1):
            window = sequence[index:L+index]
            windows.append(window)

    return windows

def compute_rep_vector(sequence, N):
    """
    Computes the repetition vector (as seen in Wooton, 1993) from a
    given sequence of a biopolymer with `N` possible residues.

    :param sequence: the nucleotide or protein sequence to generate a repetition vector for.
    :param N: the total number of possible residues in the biopolymer `sequence` belongs to.
    """
    encountered_residues = set()
    repvec = []

    for residue in sequence:
        if residue not in encountered_residues:
            residue_count = sequence.count(residue)
            repvec.append(residue_count)
            encountered_residues.add(residue)

        if len(encountered_residues) == N:
            break

    while len(repvec) < N:
        repvec.append(0)

    return sorted(repvec, reverse=True)



def complexity(sequence, N):
    """
    Computes the Shannon Entropy of a given sequence of a
    biopolymer with `N` possible residues. See (Wooton, 1993)
    for more.

    :param sequence: the nucleotide or protein sequence whose Shannon Entropy is to calculated.
    :param N: the total number of possible residues in the biopolymer `sequence` belongs to.
    """
    repvec = compute_rep_vector(sequence, N)
    L = len(sequence)
    entropy = sum([-1*(n/L)*log((n/L), N) for n in repvec if n != 0])

    return entropy



def mask_low_complexity(seq_rec, maskchar="N", N=20, L=12):
    """
    Masks low-complexity nucleic/amino acid sequences with
    a given mask character.

    :param seq_rec: a string
    :param maskchar: Character to mask low-complexity residues with.
    :param N: Number of residues to expect in the sequence. (20 for AA, 4 for DNA)
    :param L: Length of sliding window that reads the sequence.
    """

    windows = overlapping_windows(seq_rec, L)

    rep_vectors = [(window, compute_rep_vector(window, N)) for window in windows]

    window_complexity_pairs = [(rep_vector[0], complexity(rep_vector[1], N)) for rep_vector in rep_vectors]

    complexities = np.array([complexity(rep_vector[1], N) for rep_vector in rep_vectors])

    avg_complexity = complexities.mean()
    std_complexity = complexities.std()

    k1_cutoff = min([avg_complexity + std_complexity,
                 avg_complexity - std_complexity])

    alignment = [[] for i in range(0, len(seq_rec))]

    for window_offset, window_complexity_pair in enumerate(window_complexity_pairs):
        if window_complexity_pair[1] < k1_cutoff:
            window = "".join([maskchar for i in range(0, L)])
        else:
            window = window_complexity_pair[0]
    
        for residue_offset, residue in enumerate(window):
            i = window_offset+residue_offset
            alignment[i].append(residue)

    new_seq = []

    for residue_array in alignment:
        if residue_array.count(maskchar) > 3:
            new_seq.append(maskchar)
        else:
            new_seq.append(residue_array[0])

    new_seq = "".join(new_seq)

    # avoid that there are isolated N's
    for i in range(1, len(new_seq)-1):
        if new_seq[i] == 'N' and new_seq[i] != 'N' and new_seq[i] != 'N':
            new_seq[i] = seq_rec[i]

    return new_seq
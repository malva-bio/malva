import requests
import re

ENSEMBL_REST = "https://rest.ensembl.org"

def process_dna_string(sequence):
    """
    Validates and parses a DNA sequence or a FASTA-like format sequence.
    
    Parameters:
    sequence (str): The input DNA sequence or FASTA-like format sequence.
    
    Returns:
    str: A single line DNA sequence with only ATCG characters, other nucleotides replaced by A, and U replaced by T.
    """
    sequence = sequence.strip()
    
    if sequence.startswith('>'):
        # Remove the header line
        sequence = '\n'.join(sequence.split('\n')[1:])
    
    sequence = re.sub(r'\s+', '', sequence)
    valid_dna_chars = set("ATCG")
    result = []
    
    for char in sequence.upper():
        if char == 'U':
            result.append('T')
        elif char in valid_dna_chars:
            result.append(char)
        else:
            result.append('A')
    
    return {'sequence': ''.join(result)}

def handle_sequence(input_string):
    """
    Checks the input string for specific conditions and routes it accordingly.
    
    Parameters:
    input_string (str): The input string to check and handle.
    
    Returns:
    str: the parsed DNA sequence for the input_string feature
    """
    input_string = input_string.strip()
    seq_out = ""

    if input_string.startswith('gene:'):
        _input = process_gene_string(input_string)
        seq_out = get_from_gene(_input['gene_id'], _input['species'], seqtype=_input['seqtype'])[_input['split'][0]:_input['split'][1]]
    
    elif input_string.startswith('ensembl:'):
        _input = process_ensembl_string(input_string)
        seq_out = get_from_ensembl(_input['ensembl_id'])
    
    else:
        _input = process_dna_string(input_string)
        seq_out = _input['sequence']
    
    if seq_out == "":
        raise ValueError("DNA sequence not valid or not found")
    
    return seq_out
    
    
def process_gene_string(gene_string):
    """
    Processes a string that starts with 'gene:' and extracts the gene ID, species, and split parameter.
    
    Parameters:
    gene_string (str): The input string starting with 'gene:'.
    
    Returns:
    dict: A dictionary with keys 'gene_id', 'species', and 'split'.
    """
    gene_string = gene_string.strip()
    
    if not gene_string.startswith('gene:'):
        raise ValueError("Input string must start with 'gene:'")
    
    gene_info = gene_string[len('gene:'):].strip()
    
    species = 'homo sapiens'
    split = [0, None]
    seqtype = 'genomic'

    parts = gene_info.split(';')
    
    gene_id = None
    for part in parts:
        if part.startswith('species:'):
            species = part[len('species:'):].strip()
        if part.startswith('type:'):
            seqtype = part[len('type:'):].strip()
        elif part.startswith('split:'):
            split_str = part[len('split:'):].strip()
            split = split_str.split(',')
            if len(split) != 2:
                raise ValueError("The 'split' parameter must have exactly two elements")
            try:
                split = [int(s.strip()) for s in split]
            except ValueError:
                raise ValueError("Both elements of the 'split' parameter must be integers")
        else:
            if gene_id is not None:
                raise ValueError("Multiple gene IDs found in input string")
            gene_id = part.strip()
    
    if gene_id is None:
        raise ValueError("Gene ID is missing in the input string")
    
    return {'gene_id': gene_id, 'species': species, 'split': split, 'seqtype': seqtype}


def process_ensembl_string(ensembl_string):
    """
    Processes a string that starts with 'ensembl:' and extracts the Ensembl ID.
    
    Parameters:
    ensembl_string (str): The input string starting with 'ensembl:'.
    
    Returns:
    dict: A dictionary with the key 'ensembl_id'.
    """
    ensembl_string = ensembl_string.strip()
    
    if not ensembl_string.startswith('ensembl:'):
        raise ValueError("Input string must start with 'ensembl:'")
    
    ensembl_id = ensembl_string[len('ensembl:'):].strip()
    
    return {'ensembl_id': ensembl_id}

def get_from_gene(gene_id: str, species: str = "homo_sapiens", seqtype: str = "genomic"):
    if seqtype not in ['genomic', 'cdna']:
        raise ValueError("'type' must be 'genomic' or 'cdna'")
    
    ext = f"/xrefs/symbol/{species}/{gene_id}?content-type=text/plain"
    r = requests.get(ENSEMBL_REST+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
    decoded = r.json()
    ensembl_id = decoded[0]['id']

    return get_from_ensembl(ensembl_id=ensembl_id, seqtype=seqtype)

def get_from_ensembl(ensembl_id: str, seqtype: str = "genomic"):
    if seqtype not in ['genomic', 'cdna']:
        raise ValueError("'type' must be 'genomic' or 'cdna'")
    
    ext = f"/sequence/id/{ensembl_id}?type={seqtype}"
    if seqtype == 'cdna':
        ext += ';multiple_sequences=1'

    r = requests.get(ENSEMBL_REST+ext, headers={ "Content-Type" : "text/x-fasta"})
    if not r.ok:
        r.raise_for_status()

    return r.text
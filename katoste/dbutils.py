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
        seq_out = get_from_gene(_input['gene_id'], _input['species'])
    
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
    Processes a string that starts with 'gene:' and extracts the gene ID and species.
    
    Parameters:
    gene_string (str): The input string starting with 'gene:'.
    
    Returns:
    dict: A dictionary with keys 'gene_id' and 'species'.
    """
    # Remove leading and trailing whitespace
    gene_string = gene_string.strip()
    
    # Ensure the string starts with 'gene:'
    if not gene_string.startswith('gene:'):
        raise ValueError("Input string must start with 'gene:'")
    
    # Remove the 'gene:' prefix
    gene_info = gene_string[len('gene:'):].strip()
    
    # Initialize default species
    species = 'homo sapiens'
    
    # Check if the species is specified
    if ';species:' in gene_info:
        gene_id, species = gene_info.split(';species:', 1)
    else:
        gene_id = gene_info
    
    # Strip any additional whitespace
    gene_id = gene_id.strip()
    species = species.strip()
    
    # Return the parsed gene ID and species as a dictionary
    return {'gene_id': gene_id, 'species': species}


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

def get_from_gene(gene_id: str, species: str = "homo_sapiens"):
    ext = f"/xrefs/symbol/{species}/{gene_id}?content-type=text/plain"
    r = requests.get(ENSEMBL_REST+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
    decoded = r.json()
    ensembl_id = decoded[0]['id']

    return get_from_ensembl(ensembl_id=ensembl_id)

def get_from_ensembl(ensembl_id: str):
    ext = f"/sequence/id/{ensembl_id}?content-type=text/plain"

    r = requests.get(ENSEMBL_REST+ext, headers={ "Content-Type" : "text/plain"})
    if not r.ok:
        r.raise_for_status()

    return r.text
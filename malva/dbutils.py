import os
import pathlib
import pickle
import re
import requests
import sqlite3

from rich.progress import track

from malva.reader import iterate_fasta
from malva.utils import download_url_to_file

ENSEMBL_REST = "https://rest.ensembl.org"

# implement pre-download ensembl data
revcomp = str.maketrans("ACGTU", "TGCAA")

# ensembl fasta
ENSEMBL_FASTA = {"mus_musculus": "Mus_musculus.GRCm39.cdna.all.fa.gz",
                 "homo_sapiens": "Homo_sapiens.GRCh38.cdna.all.fa.gz"}

SEQDB_LOCAL_DIR = pathlib.Path.home().joinpath(".malva", "ensembl_data")

class EnsemblLocalDB:
    """
    A class to manage local storage and retrieval of Ensembl cDNA sequences.

    This class provides functionality to download, index, and query cDNA sequences
    from Ensembl for specified species.

    Attributes:
        data_dir (str): Directory to store downloaded and indexed data.
        index (dict): Dictionary mapping Ensembl IDs to sequences.
        gene_to_ensembl (dict): Dictionary mapping gene names to Ensembl IDs.
    """
    def __init__(self, data_dir=SEQDB_LOCAL_DIR, species='mus_musculus'):
        f"""
        Initialize the EnsemblLocalDB object.

        Args:
            data_dir (str, optional): Directory to store data. Defaults to '{SEQDB_LOCAL_DIR}'.
        """
        self.species = species
        self.data_dir = os.path.join(data_dir, self.species)
        self.db_path = os.path.join(self.data_dir, 'ensembl.db')
        pathlib.Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _create_tables(self):
        """
        Create the necessary tables in the SQLite database if they don't exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # Use Write-Ahead Logging
            conn.execute("PRAGMA synchronous=NORMAL")  # Reduce synchronous IOPS
            conn.execute("PRAGMA cache_size=1000000")  # Increase cache size (in pages)
            conn.execute("PRAGMA mmap_size=30000000000")  # Memory-mapped I/O

            cursor = conn.cursor()

            # Single table for all data with a composite primary key
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gene_data (
                    gene_name TEXT COLLATE NOCASE,
                    ensembl_id TEXT,
                    sequence TEXT,
                    PRIMARY KEY (gene_name, ensembl_id)
                ) WITHOUT ROWID
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_gene_name ON gene_data (gene_name COLLATE NOCASE)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ensembl_id ON gene_data (ensembl_id)')
            conn.commit()

    def database_exists(self):
        """
        Check if the local database exists, is properly set up, and contains data.

        Returns:
            bool: True if the database exists, is set up, and contains data; False otherwise.
        """
        if not os.path.exists(self.db_path):
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if the gene_data table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gene_data'")
                if not cursor.fetchone():
                    return False
                
                # Check if the gene_data table contains any rows
                cursor.execute("SELECT COUNT(*) FROM gene_data")
                count = cursor.fetchone()[0]
                
                return count > 0
        except sqlite3.Error:
            # If there's any database error, consider the database as not properly set up
            return False

    def download_cdna_fasta(self):
        """
        Download the cDNA FASTA file for a given species from Ensembl.

        Returns:
            str: The local path to the downloaded file.
        """
        base_url = "http://ftp.ensembl.org/pub/current_fasta"
        filename = ENSEMBL_FASTA[self.species]
        url = f"{base_url}/{self.species}/cdna/{filename}"
        local_path = os.path.join(self.data_dir, filename)

        print(url)

        if not os.path.exists(local_path):
            download_url_to_file(url, local_path)

        return local_path

    def index_fasta(self, fasta_file):
        """
        Parse the FASTA file and store the data in the SQLite database efficiently.

        Args:
            fasta_file (str): Path to the FASTA file to be indexed.
        """
        print(f"Indexing {fasta_file}...")
        data_to_insert = []
        for record in track(iterate_fasta(fasta_file), "Indexing EnsemblLocalDB"):
            ensembl_id = record[0].split('.')[0]
            sequence = str(record[1])
            gene_name = record[0].split('gene_symbol:')[1].split()[0] if 'gene_symbol:' in record[0] else None
            
            if gene_name:
                data_to_insert.append((gene_name, ensembl_id, sequence))
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany('INSERT OR REPLACE INTO gene_data (gene_name, ensembl_id, sequence) VALUES (?, ?, ?)', data_to_insert)
            conn.commit()


    def save_index(self):
        """
        Save the indexed data to disk using pickle.
        """
        with open(os.path.join(self.data_dir, 'index.pkl'), 'wb') as f:
            pickle.dump((self.index, self.gene_to_ensembl), f)

    def load_index(self):
        """
        Load the indexed data from disk.

        Returns:
            bool: True if the index was successfully loaded, False otherwise.
        """
        index_path = os.path.join(self.data_dir, 'index.pkl')
        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                self.index, self.gene_to_ensembl = pickle.load(f)
            return True
        return False

    def get_from_gene(self, gene_id, seq_type='cdna'):
        """
        Retrieve cDNA sequences for a given gene ID or Ensembl ID.

        This method first checks if the database exists. If not, it sets up the database
        by downloading and indexing the necessary files. The search is case-insensitive.

        Args:
            gene_id (str): The gene ID or Ensembl ID to query.
            seq_type (str, optional): The type of sequence to retrieve. Currently only 'cdna' is supported. Defaults to 'cdna'.

        Returns:
            list: A list of cDNA sequences associated with the given gene ID.

        Raises:
            ValueError: If seq_type is not 'cdna'.
        """
        if seq_type != 'cdna':
            raise ValueError("Only 'cdna' sequence type is currently supported")

        if not self.database_exists():
            print("Local database not found. Setting up the database...")
            fasta_file = self.download_cdna_fasta()
            self.index_fasta(fasta_file)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Use a single query to fetch all relevant sequences
            cursor.execute('''
                SELECT sequence FROM gene_data
                WHERE gene_name = ? COLLATE NOCASE OR ensembl_id = ? COLLATE NOCASE
            ''', (gene_id, gene_id))
            
            sequences = [row[0] for row in cursor.fetchall()]

        return sequences

# the databases that have been loaded into memory
_localdbs = {}

def process_dna_string(sequence):
    """
    Validates and parses a DNA sequence or a FASTA-like format sequence.

    Args:
        sequence (str): The input DNA sequence or FASTA-like format sequence.

    Returns:
        str: A single line DNA sequence with only ATCG characters, other nucleotides replaced by A, and U replaced by T.
    """
    sequence = sequence.strip()

    if sequence.startswith(">"):
        # Remove the header line
        return {"sequence": parse_multifasta(sequence)}

    sequence = re.sub(r"\s+", "", sequence)
    valid_dna_chars = set("ATCG")
    result = []

    for char in sequence.upper():
        if char == "U":
            result.append("T")
        elif char in valid_dna_chars:
            result.append(char)
        else:
            result.append("A")

    return {"sequence": "".join(result)}

def parse_multifasta(fasta_string):
    """
    Validates and parses a FASTA-formatted string.

    Args:
        fasta_string (str): FASTA-formatted sequence.
    Returns:
        list: A list of single line DNA sequences with only ATCG characters, other nucleotides replaced by A, and U replaced by T.
    """
    sequences = []

    for fasta_entry in fasta_string.split(">")[1:]:
        lines = fasta_entry.split("\n")
        dna_sequence = "".join(line.strip() for line in lines[1:])
        dna_sequence = re.sub(r"\s+", "", dna_sequence)
        valid_dna_chars = set("ATCG")
        result = []
        for char in dna_sequence.upper():
            if char == "U":
                result.append("T")
            elif char in valid_dna_chars:
                result.append(char)
            else:
                result.append("A")
        sequences.append("".join(result))

    return sequences


def handle_sequence(input_string, recursion=True):
    """
    Checks the input string for specific conditions and routes it accordingly.

    Args:
        input_string (str): The input string to check and handle.

    Returns:
        str: the parsed DNA sequence for the input_string feature
    """
    input_string = input_string.strip()
    seq_out = ""

    if input_string.startswith("gene:"):
        _input = process_gene_string(input_string)
        _seq_out = get_from_gene(_input["gene_id"], _input["species"], seqtype=_input["seqtype"])

        seq_out = []

        for s in _seq_out:
            if _input["split"][1] is not None and _input["split"][0] > _input["split"][1]:
                s = s.translate(revcomp)[::-1][_input["split"][1] : _input["split"][0]]
            else:
                s = s[_input["split"][0] : _input["split"][1]]

            seq_out.append(s)

        return seq_out

    elif input_string.startswith("ensembl:"):
        _input = process_ensembl_string(input_string)
        _seq_out = get_from_ensembl(_input["ensembl_id"], _input["seqtype"])

        seq_out = []

        for s in _seq_out:
            if _input["split"][1] is not None and _input["split"][0] > _input["split"][1]:
                s = s.translate(revcomp)[::-1][_input["split"][1] : _input["split"][0]]
            else:
                s = s[_input["split"][0] : _input["split"][1]]

            seq_out.append(s)

        return seq_out

    elif input_string.startswith(">"):
        seq_out = parse_multifasta(input_string)
        return seq_out["sequence"]
    else:
        input_string = validate_and_infer_query(input_string)
        if recursion:
            seq_out = handle_sequence(input_string, recursion=False)
        else:
            seq_out = process_dna_string(input_string)["sequence"]

    if seq_out == "":
        raise ValueError("DNA sequence not valid or not found")

    # we apply again in case ensembl is parsed as fasta
    return process_dna_string(seq_out)["sequence"]


def process_gene_string(gene_string):
    """
    Processes a string that starts with 'gene:' and extracts the gene ID, species, and split parameter.

    Args:
        gene_string (str): The input string starting with 'gene:'.

    Returns:
        dict: A dictionary with keys 'gene_id', 'species', and 'split'.
    """
    gene_info = gene_string.strip()

    if not gene_string.startswith("gene:"):
        raise ValueError("Input string must start with 'gene:'")

    species = "homo_sapiens"
    split = [0, None]
    seqtype = "genomic"

    parts = gene_info.split(";")

    gene_id = None
    for part in parts:
        if part.startswith("species:"):
            species = part[len("species:") :].strip()
        elif part.startswith("type:"):
            seqtype = part[len("type:") :].strip()
        elif part.startswith("split:"):
            split_str = part[len("split:") :].strip()
            split = split_str.split(",")
            if len(split) != 2:
                raise ValueError("The 'split' parameter must have exactly two elements")
            try:
                split = [int(s.strip()) for s in split]
            except ValueError:
                raise ValueError("Both elements of the 'split' parameter must be integers")
        elif part.startswith("gene:"):
            if gene_id is not None:
                raise ValueError("Multiple gene IDs found in input string")
            gene_id = part[len("gene:") :].strip()

    if gene_id is None:
        raise ValueError("Gene ID is missing in the input string")

    return {"gene_id": gene_id, "species": species, "split": split, "seqtype": seqtype}


def process_ensembl_string(ensembl_string):
    """
    Processes a string that starts with 'ensembl:' and extracts the Ensembl ID.

    Args:
        ensembl_string (str): The input string starting with 'ensembl:'.

    Returns:
        dict: A dictionary with the key 'ensembl_id'.
    """
    ensembl_string = ensembl_string.strip()

    if not ensembl_string.startswith("ensembl:"):
        raise ValueError("Input string must start with 'ensembl:'")

    split = [0, None]
    seqtype = "genomic"

    parts = ensembl_string.split(";")

    ensembl_id = None
    for part in parts:
        if part.startswith("type:"):
            seqtype = part[len("type:") :].strip()
        elif part.startswith("split:"):
            split_str = part[len("split:") :].strip()
            split = split_str.split(",")
            if len(split) != 2:
                raise ValueError("The 'split' parameter must have exactly two elements")
            try:
                split = [int(s.strip()) for s in split]
            except ValueError:
                raise ValueError("Both elements of the 'split' parameter must be integers")
        elif part.startswith("ensembl:"):
            if ensembl_id is not None:
                raise ValueError("Multiple ensembl IDs found in input string")
            ensembl_id = part[len("ensembl:") :].strip()

    if ensembl_id is None:
        raise ValueError("Ensembl ID is missing in the input string")

    return {"ensembl_id": ensembl_id, "split": split, "seqtype": seqtype}


def get_from_gene(gene_id: str, species: str = "homo_sapiens", seqtype: str = "genomic"):
    if species not in _localdbs:
        _localdbs[species] = EnsemblLocalDB(species=species)
    
    _localdb = _localdbs[species]

    if _localdb.database_exists():
        return _localdb.get_from_gene(gene_id, seqtype)

    if seqtype not in ["genomic", "cdna"]:
        raise ValueError("'type' must be 'genomic' or 'cdna'")

    ext = f"/xrefs/symbol/{species}/{gene_id}?content-type=text/plain"
    r = requests.get(ENSEMBL_REST + ext, headers={"Content-Type": "application/json"})
    if not r.ok:
        r.raise_for_status()
    decoded = r.json()

    if len(decoded) < 1:
        raise ValueError(f"Gene '{gene_id}' for species '{species}' was not found")

    ensembl_id = decoded[0]["id"]

    return get_from_ensembl(ensembl_id=ensembl_id, seqtype=seqtype)


def get_from_ensembl(ensembl_id: str, seqtype: str = "genomic"):
    if seqtype not in ["genomic", "cdna", "transcript"]:
        raise ValueError("'type' must be 'genomic' or 'cdna'")

    ext = f"/sequence/id/{ensembl_id}?type={seqtype}"
    if seqtype == "transcript":
        ext = f"/sequence/id/{ensembl_id}"
    elif seqtype == "cdna":
        ext += ";multiple_sequences=1"

    r = requests.get(ENSEMBL_REST + ext, headers={"Content-Type": "text/x-fasta"})
    if not r.ok:
        r.raise_for_status()

    return parse_multifasta(r.text)


def validate_and_infer_query(input_string):
    """
    Validate and infer whether the input is gene IDs or DNA sequences.

    Args:
        input_string (str): The user input string.

    Returns:
        str: Corrected query string or raises an exception if validation fails.
    """
    # Split the input into lines and remove empty lines
    lines = [line.strip() for line in input_string.splitlines() if line.strip()]
    lines = lines[:1]

    def is_dna_sequence(seq):
        return bool(re.fullmatch(r"[ACGTNacgtn]+", seq))

    def is_gene_id(gene):
        return bool(re.fullmatch(r"[a-zA-Z0-9._-]+", gene))

    def is_ensembl_id(gene):
        return bool(re.fullmatch(r"ENS[GTPE][a-zA-Z0-9._-]+", gene, re.IGNORECASE))

    inferred_genes = []
    inferred_sequences = []
    inferred_ensembl = []

    input_types = set()

    for line in lines:
        if is_dna_sequence(line):
            inferred_sequences.append(line)
            input_types.add("sequence")
        elif is_gene_id(line):
            inferred_genes.append(line)
            input_types.add("gene")
        elif is_ensembl_id(line):
            inferred_ensembl.append(line)
            input_types.add("ensembl")
        else:
            raise ValueError(f"Invalid input: '{line}'. Please enter valid gene IDs or DNA sequences.")

    if len(input_types) > 1:
        raise ValueError(
            "Mixed input detected: Please provide either gene IDs, Ensembl IDs, or DNA sequences, not multiple types."
        )

    if inferred_genes:
        return "gene:" + ",".join(inferred_genes)
    elif inferred_ensembl:
        return "ensembl:" + ",".join(inferred_ensembl)
    elif inferred_sequences:
        return "".join(inferred_sequences)
    else:
        raise ValueError("No valid gene IDs, Ensembl IDs, or DNA sequences detected. Please check your input.")

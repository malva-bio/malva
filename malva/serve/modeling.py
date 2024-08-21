import ollama
import re
import ast
import logging

from malva.dbutils import handle_sequence

SYSTEM_PROMPT = """
You are an AI assistant specialized in generating gene lists for spatial transcriptomics analysis. Your task is to create Python-formatted lists of genes that are highly relevant to specific biological phenotypes or systems described by the user.

Requirements:
1. Generate a single Python list containing gene symbols (not gene names or descriptions).
2. Include only genes that are highly specific and relevant to the described phenotype or biological system.
3. Base your selections on in-depth knowledge from scientific literature.
4. Provide between 10 and 50 genes, prioritizing the most informative markers.
5. Focus on genes that would be useful for calculating a gene activity score in spatial transcriptomics data.
6. Consider the specific developmental stage, tissue type, or biological context provided in the user's query.
7. Order the genes by relevance or importance, with the most crucial genes listed first.

Output format:
- Provide only the Python-formatted list of gene symbols.
- Do not include any explanatory text, comments, or additional information.
- Use standard Python list syntax: ["GENE1", "GENE2", "GENE3", ...]

Example input: "Identify genes for neural progenitor cells in the developing mouse cortex at E14.5"
Example output: ["Nes", "Sox2", "Pax6", "Hes1", "Hes5", "Notch1", "Fgfr2", "Emx2", "Otx1", "Ascl1"]

Respond to the user's query by providing a single, focused list of genes that best represents the described biological system or phenotype.
""".replace('\n', ' ')

SPECIES = "homo_sapiens"

def setup_model(model_name: str = 'malva', system_prompt: str = SYSTEM_PROMPT):
    modelfile=f'''
    FROM llama3.1
    SYSTEM {system_prompt}
    '''
    ollama.create(model=model_name, modelfile=modelfile)

def parse_gene_lists(input_string):
    # Define the regular expression to match one or two lists with optional newlines in between
    pattern = r"\[(.*?)\](?:\s*|\n|\n\n)(?:\[(.*?)\])?"
    
    # Search for the lists in the input string
    match = re.search(pattern, input_string, re.DOTALL)
    
    if match:
        # Extract and convert the first list
        list1 = ast.literal_eval(f"[{match.group(1)}]")
        
        # Check if there's a second list
        if match.group(2):
            list2 = ast.literal_eval(f"[{match.group(2)}]")
            return list1, list2
        else:
            return list1, None
    else:
        raise ValueError("No valid gene list found in the input string")

def handle_natural_query(query: str, model: str = 'malva', verbose: bool = True):
    model_response = ollama.generate(model=model, prompt=query)
    genelists_str = model_response['response']
    if verbose:
        logging.info(f"Query: {query}, reponse {genelists_str}")

    positive_markers, _ = parse_gene_lists(genelists_str)

    positive_markers_sequences = []
    for pm in positive_markers:
        try:
            positive_markers_sequences += handle_sequence(f"gene:{pm};species:{'_'.join(SPECIES.lower().split(' '))};type:cdna") 
        except KeyError:
            logging.error(f"Could not find gene {pm}")

    return positive_markers, positive_markers_sequences
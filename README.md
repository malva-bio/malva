# Malva

**[Documentation](https://malva.readthedocs.io)** | **[Web Platform](https://malva.bio)** 

Malva is a nucleotide sequence indexer that enables sequence search at single-cell and spatial resolution: query any sequence across millions of single cells, in seconds.


Malva powers **Malva Index**, a very large collection of datasets spanning 60+ million single cells.

You can use the [Malva Client API](https://github.com/malva-bio/malva_client) to connect to and query the Malva Index.


**Binaries are freely available for academic non-profit use**

## What Malva Can Do

- Search any nucleotide sequence across single cells from your datasets
- *In silico* probe-based detection of viral/bacterial transcripts, circular RNAs, splice variants, and mutations

## Get started

```bash
# first, get "malva_binaries" upon request (see below)
cd malva_binaries
pip install .
```

```bash
# Build an index from your data
malva index --reads-in R1.fastq.gz R2.fastq.gz --spatial-bc-in openst_barcodes.tsv --index-out my_index --flavor openst

# Quantify gene expression
malva quant --index-in my_index --reference human_utr --folder-out output --h5ad
```

See the [Malva Tools documentation](https://malva-bio.github.io/malva) for detailed usage.

## System Requirements

- Python 3.11+
- Linux, macOS, or Windows
- Internet connection (for Malva Index)
- For local analysis: `htslib` recommended for better performance

## Citation

If you use Malva in your research, please cite:

> [TBA]

## Contact

- Issues and questions: [GitHub Issues](https://github.com/malva-bio/malva/issues)
- General inquiries: [Rajewsky Lab @ MDC Berlin](https://rajewsky-lab.github.io)
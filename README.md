# katoste: fast indexing and querying of genomic sequences from spatial transcriptomics data

### [🌐 website](https://rajewsky-lab.github.io/katoste/latest) | [📜 preprint](https://www.biorxiv.org/content/10.1101/2023.12.22.572554v1) | [🐁 datasets](https://rajewsky-lab.github.io/katoste/latest/examples/datasets/)

TODO: include an image of the workflow

`katoste` is an [inverted index](https://en.wikipedia.org/wiki/Inverted_index) that enables rapid retrieval of sequences from spatial transcriptomics data.

`katoste` operates by constructing an inverse index from the raw sequence data and corresponding spatial coordinates. This facilitates fast retrieval of sequences based on user queries, such as k-mers or custom sequences.

## Quick start
TODO: explain basic concepts

### 1. Required data
TODO: describe
You must have a tsv file with the following columns:

```bash
cell_bc  xcoord   ycoord
ATTAATTA 1  2
ATTCCCTA 1  3
...   ...   ...
```

### 2. Create an index
TODO describe files required and basic concepts
```bash
katoste index \
   --reads-in R1.fastq.gz R2.fastq.gz \
   --spatial-bc-in spatial_barcodes.tsv \
   --index-out example_index
```

This will create a spatial index in the folder `example_index`

### 3. Visualize data
```bash
katoste show \
   --reads-in R1.fastq.gz R2.fastq.gz \
   --spatial-bc-in spatial_barcodes.tsv \
   --index-out example_index
```

## Documentation
All the detail to `katoste` are available in [our documentation website](https://rajewsky-lab.github.io/katoste/).

We love to have an open approach to documentation. We decided to use [mkdocs](https://github.com/mkdocs/mkdocs) as our documentation backend 
to make your life easier. So, feel free to suggest changes by opening a 
[documentation-related issue](https://github.com/rajewsky-lab/katoste/issues/new?assignees=&labels=docs&template=&title=)!

You can build the documentation locally by following these steps:
1. Clone this repository
   ```sh
   git clone https://github.com/rajewsky-lab/katoste
   ```
2. Install the dependencies for building the documentation:
   ```sh
   pip install mkdocs-material mkdocs-autorefs mknotebooks
   ```
3. Serve mkdocs with the following command:
   ```sh
   mkdocs serve -f katoste/mkdocs.yml
   ```

## Contributing
`katoste` is an open-source project mostly maintained by the [Rajewsky lab @ MDC Berlin](https://www.mdc-berlin.de/n-rajewsky) - so, your involvement is warmly welcome! 
If you're excited to join us, we recommend the following steps:

- Looking for ideas? See our [Volunteer Project Board](https://github.com/orgs/rajewsky-lab/projects/1) to see what we may need help with.
- Found a bug? Contact an admin in the form of an [issue](https://github.com/rajewsky-lab/katoste/issues/new?assignees=&labels=&template=bug-report.md&title=).
- Implement your idea following guidelines set by the [official contributing guide](CONTRIBUTING.md)
- Wait for admin approval; approval is iterative, but if accepted will belong to the main repository.

In general, you can always refer to the [contribution guidelines](CONTRIBUTING.md) for more details!
Currently, only [admins](https://github.com/orgs/rajewsky-lab/people) will be merging all accepted changes.

## Code of Conduct
Everyone interacting in the `katoste` project's codebases, issue trackers, and discussion forums is expected to follow the [PSF Code of Conduct](https://www.python.org/psf/conduct/).

## License
The software tools of this project are under the GNU License - see the [LICENSE](LICENSE) file for details.

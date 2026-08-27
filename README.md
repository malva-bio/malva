# Malva

**[Documentation](https://malva.readthedocs.io)** | **[Web Platform](https://malva.bio)**

Malva is a nucleotide sequence indexer that enables sequence search at single-cell and spatial resolution: query any sequence across millions of single cells, in seconds.

Malva powers **Malva Index**, a large collection of datasets spanning 60+ million single cells.
You can use the [Malva Client API](https://github.com/malva-bio/malva_client) to connect to and query the Malva Index.

> **Code and binaries are freely available for academic non-profit use.** See [LICENSE](LICENSE) for details.

---

## What Malva Can Do

- Search any nucleotide sequence across single cells from your own datasets
- *In silico* probe-based detection of viral/bacterial transcripts, circular RNAs, splice variants, and mutations

---

## Installation

`Malva` can be installed from pip.

```bash
pip install malva
```

**Linux note:** installing `liburing` is recommended for optimal I/O performance.
```bash
sudo apt-get install liburing2      # Debian/Ubuntu
sudo dnf install liburing           # RHEL/Fedora
```
Malva will build and run without it, but I/O will fall back to standard `pread`.

---

## Building from Source

Requires Python ≥ 3.9, a C++17 compiler, and Cython ≥ 3.0.

```bash
pip install poetry cython>=3.0 numpy setuptools wheel pkgconfig
pip install .
```

See the [documentation](https://malva.readthedocs.io) for full build instructions, usage, and examples.

---

## Citation

If you use Malva in your research, please cite our [paper in Nature](https://www.nature.com/articles/s41586-026-10975-w):

```latex
@article{LenPerin2026,
  title = {Ultrafast and reference-free sequence discovery in single-cell data},
  ISSN = {1476-4687},
  url = {http://dx.doi.org/10.1038/s41586-026-10975-w},
  DOI = {10.1038/s41586-026-10975-w},
  journal = {Nature},
  publisher = {Springer Science and Business Media LLC},
  author = {León-Periñán,  Daniel and Karaiskos,  Nikos and Rajewsky,  Nikolaus},
  year = {2026},
  month = Aug 
}
```
---

## Contact

- Issues: [GitHub Issues](https://github.com/malva-bio/malva/issues)
- General inquiries: [Rajewsky Lab @ MDC Berlin](https://rajewsky-lab.github.io)

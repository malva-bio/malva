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

If you use Malva in your research, please cite:

> [TBA]

---

## Contact

- Issues: [GitHub Issues](https://github.com/malva-bio/malva/issues)
- General inquiries: [Rajewsky Lab @ MDC Berlin](https://rajewsky-lab.github.io)

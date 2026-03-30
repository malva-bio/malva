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

### Option 1 — pre-built wheel (recommended)

Pre-built wheels for Linux (manylinux 2.28, x86_64) are published on the
[GitHub Releases page](https://github.com/malva-bio/malva/releases).

```bash
# Download the wheel for your Python version and install
pip install malva-<version>-cp311-cp311-manylinux_2_28_x86_64.whl
```

> **Linux requirement:** `liburing` must be installed on the host.
> ```bash
> # Debian/Ubuntu
> sudo apt-get install liburing2
> # RHEL/Fedora
> sudo dnf install liburing
> ```

### Option 2 — container bundle (no root, HPC-friendly)

Each release also provides a self-contained `malva_dist.tar.gz` that bundles
malva and all dependencies inside an Apptainer container image.
No Python or root access required on the host.

```bash
tar -xf malva_dist.tar.gz
./malva_dist/malva --version
./malva_dist/malva index --help
```

---

## Quick Start

```bash
# Build an index from spatial transcriptomics reads
malva index \
  --reads-in R1.fastq.gz R2.fastq.gz \
  --spatial-bc-in openst_barcodes.tsv \
  --index-out my_index \
  --flavor openst

# Quantify against a reference
malva quant \
  --index-in my_index \
  --reference human_utr \
  --folder-out output \
  --h5ad
```

See the [documentation](https://malva.readthedocs.io) for detailed usage of all commands.

---

## Building from Source (Advanced)

Building malva from source requires a C++17 compiler, Cython ≥ 3.0, and
`liburing` (Linux only).

```bash
git clone https://github.com/malva-bio/malva.git
cd malva

# Install Python build dependencies
pip install poetry cython>=3.0 numpy setuptools wheel pkgconfig

# Build and install (production mode)
pip install .

# Or, to keep profiling/tracing hooks for development:
MALVA_DEBUG_BUILD=1 pip install -e .
```

### Building distributable wheels

Use the scripts in [`build_scripts/`](build_scripts/README.md) to produce
manylinux-compliant wheels:

```bash
# Recommended: Apptainer + manylinux_2_28 (produces portable Linux wheels)
bash build_scripts/build_with_apptainer.sh

# Alternative: Docker + cibuildwheel
bash build_scripts/build_with_docker.sh
```

Wheels are written to `wheelhouse/`. See [`build_scripts/README.md`](build_scripts/README.md)
for full instructions including the container distribution option.

#### Build environment variable

| Variable | Default | Description |
|---|---|---|
| `MALVA_DEBUG_BUILD` | `0` | Set to `1` for debug builds with Cython tracing/profiling |

---

## Verbose logging

Set `MALVA_DEBUG=1` to enable verbose debug logging for any command:

```bash
MALVA_DEBUG=1 malva index ...
```

---

## System Requirements

| | Minimum |
|---|---|
| Python | 3.9 – 3.12 |
| OS | Linux (x86_64), macOS (x86_64 / arm64) |
| Linux | `liburing` ≥ 2.0 |
| Compiler (source builds) | GCC ≥ 11 or Clang ≥ 14, C++17 |

---

## Citation

If you use Malva in your research, please cite:

> [TBA]

---

## Contact

- Issues and questions: [GitHub Issues](https://github.com/malva-bio/malva/issues)
- General inquiries: [Rajewsky Lab @ MDC Berlin](https://rajewsky-lab.github.io)

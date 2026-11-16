# -*- coding: utf-8 -*-
# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

from setuptools import setup

packages = \
['malva', 'malva.serve', 'malva.serve.templates']

package_data = \
{'': ['*'], 'malva': ['data/*'], 'malva.serve': ['static/css/*']}

install_requires = \
['anndata>=0.10.0',
 'datashader>0.15.0',
 'dnaio>1.0.0',
 'flask>2.3.0',
 'flask_cors>=6.0.2',
 'flask_session>=0.8.0',
 'h5py>=3',
 'matplotlib>=3.5',
 'numpy>=1.17.0',
 'ollama>=0.3.1',
 'pandas>=1.0',
 'pillow>=9.0',
 'pyyaml>5.4',
 'requests>=2.26',
 'rich>13.0',
 'scanpy>=1.9',
 'scikit-image>=0.24.0',
 'scikit-learn>=1.0',
 'scipy>=1.7',
 'setproctitle>1.0',
 'tifffile>2022.4.8',
 'tqdm>=4.66.6',
 'werkzeug>=2.0',
 'xopen>=1.7.0']

extras_require = \
{'all': ['sra-reader>=0.1.0', 'pysam>=0.22.0'],
 'bam': ['pysam>=0.22.0'],
 'sra': ['sra-reader>=0.1.0']}

entry_points = \
{'console_scripts': ['malva = malva.__main__:run_malva']}

setup_kwargs = {
    'name': 'malva',
    'version': '1.1.4',
    'description': 'malva: fast indexing and querying of genomic sequences from single-cell omics data',
    'long_description': '# Malva\n\n[![PyPI](https://img.shields.io/pypi/v/malva)](https://pypi.org/project/malva/)\n[![Documentation](https://readthedocs.org/projects/malva/badge/?version=latest)](https://malva.readthedocs.io)\n[![GitHub](https://img.shields.io/badge/GitHub-malva--bio%2Fmalva-blue)](https://github.com/malva-bio/malva)\n\n**[Documentation](https://malva.readthedocs.io)** | **[Web Platform](https://malva.bio)**\n\nMalva is a nucleotide sequence indexer that enables sequence search at single-cell and spatial resolution: query any sequence across millions of single cells, in seconds.\n\nMalva powers [**Malva Index**](https://malva.mdc-berlin.de), a large collection of datasets spanning 75+ million human single cells.\nYou can use the [Malva Client API](https://github.com/malva-bio/malva_client) to connect to and query the Malva Index.\n\n> **Code and binaries are freely available for academic non-profit use.** See [LICENSE](LICENSE) for details.\n\n---\n\n## What Malva Can Do\n\n- Search any nucleotide sequence across single cells from your own datasets\n- *In silico* probe-based detection of viral/bacterial transcripts, circular RNAs, splice variants, and mutations\n\n---\n\n## Installation\n\n`Malva` can be installed from pip.\n\n```bash\npip install malva\n```\n\n**Linux note:** installing `liburing` is recommended for optimal I/O performance.\n```bash\nsudo apt-get install liburing2      # Debian/Ubuntu\nsudo dnf install liburing           # RHEL/Fedora\n```\nMalva will build and run without it, but I/O will fall back to standard `pread`.\n\n---\n\n## Building from Source\n\nRequires Python ≥ 3.9, a C++17 compiler, and Cython ≥ 3.0.\n\n```bash\npip install poetry cython>=3.0 numpy setuptools wheel pkgconfig\npip install .\n```\n\nSee the [documentation](https://malva.readthedocs.io) for full build instructions, usage, and examples.\n\n---\n\n## Citation\n\nIf you use Malva in your research, please cite our [paper in Nature](https://www.nature.com/articles/s41586-026-10975-w):\n\n```latex\n@article{LenPerin2026,\n  title = {Ultrafast and reference-free sequence discovery in single-cell data},\n  ISSN = {1476-4687},\n  url = {http://dx.doi.org/10.1038/s41586-026-10975-w},\n  DOI = {10.1038/s41586-026-10975-w},\n  journal = {Nature},\n  publisher = {Springer Science and Business Media LLC},\n  author = {León-Periñán,  Daniel and Karaiskos,  Nikos and Rajewsky,  Nikolaus},\n  year = {2026},\n  month = Aug \n}\n```\n---\n\n## Contact\n\n- Issues: [GitHub Issues](https://github.com/malva-bio/malva/issues)\n- General inquiries: [Rajewsky Lab @ MDC Berlin](https://rajewsky-lab.github.io)\n',
    'author': 'Daniel León-Periñán',
    'author_email': 'daniel.leonperinan@mdc-berlin.de',
    'maintainer': 'Daniel León-Periñán',
    'maintainer_email': 'daniel.leonperinan@mdc-berlin.de',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'entry_points': entry_points,
    'python_requires': '>=3.9,<3.15',
}
from build_extensions import *
build(setup_kwargs)

setup(**setup_kwargs)

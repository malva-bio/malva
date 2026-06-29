# -*- coding: utf-8 -*-
# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

from setuptools import setup

packages = \
['malva', 'malva.serve']

package_data = \
{'': ['*'], 'malva': ['data/*'], 'malva.serve': ['static/*']}

install_requires = \
['numpy>=1.17.0',
 'pandas>=1.0',
 'dnaio>1.0.0',
 'xopen>=1.7.0',
 'h5py>=3',
 'rich>13.0',
 'setproctitle>1.0',
 'tifffile>2022.4.8',
 'pyyaml>5.4',
 'flask>2.3.0',
 'flask_session>=0.8.0',
 'flask_cors>=6.0.2',
 'tqdm>=4.66.6']

entry_points = \
{'console_scripts': ['malva = malva.__main__:run_malva']}

setup_kwargs = {
    'name': 'malva',
    'version': '1.0.1',
    'description': 'malva: fast indexing and querying of genomic sequences from spatial transcriptomics data',
    'long_description': "# malva: fast indexing and querying of genomic sequences from spatial transcriptomics data\n\n### [🌐 website](https://rajewsky-lab.github.io/malva/latest) | [📜 preprint](https://www.biorxiv.org/content/10.1101/2023.12.22.572554v1) | [🐁 datasets](https://rajewsky-lab.github.io/malva/latest/examples/datasets/)\n\nTODO: include an image of the workflow\n\n`malva` is an [inverted index](https://en.wikipedia.org/wiki/Inverted_index) that enables rapid retrieval of sequences from spatial transcriptomics data.\n\n`malva` operates by constructing an inverse index from the raw sequence data and corresponding spatial coordinates. This facilitates fast retrieval of sequences based on user queries, such as k-mers or custom sequences.\n\n## Quick start\nTODO: include here how to install and use with an Open-ST dataset\nTODO: include an image\n\n## Documentation\nAll the detail to `malva` are available in [our documentation website](https://rajewsky-lab.github.io/malva/).\n\nWe love to have an open approach to documentation. We decided to use [mkdocs](https://github.com/mkdocs/mkdocs) as our documentation backend \nto make your life easier. So, feel free to suggest changes by opening a \n[documentation-related issue](https://github.com/rajewsky-lab/malva/issues/new?assignees=&labels=docs&template=&title=)!\n\nYou can build the documentation locally by following these steps:\n1. Clone this repository\n   ```sh\n   git clone https://github.com/rajewsky-lab/malva\n   ```\n2. Install the dependencies for building the documentation:\n   ```sh\n   pip install mkdocs-material mkdocs-autorefs mknotebooks\n   ```\n3. Serve mkdocs with the following command:\n   ```sh\n   mkdocs serve -f malva/mkdocs.yml\n   ```\n\n## Contributing\n`malva` is an open-source project mostly maintained by the [Rajewsky lab @ MDC Berlin](https://www.mdc-berlin.de/n-rajewsky) - so, your involvement is warmly welcome! \nIf you're excited to join us, we recommend the following steps:\n\n- Looking for ideas? See our [Volunteer Project Board](https://github.com/orgs/rajewsky-lab/projects/1) to see what we may need help with.\n- Found a bug? Contact an admin in the form of an [issue](https://github.com/rajewsky-lab/malva/issues/new?assignees=&labels=&template=bug-report.md&title=).\n- Implement your idea following guidelines set by the [official contributing guide](CONTRIBUTING.md)\n- Wait for admin approval; approval is iterative, but if accepted will belong to the main repository.\n\nIn general, you can always refer to the [contribution guidelines](CONTRIBUTING.md) for more details!\nCurrently, only [admins](https://github.com/orgs/rajewsky-lab/people) will be merging all accepted changes.\n\n## Code of Conduct\nEveryone interacting in the `malva` project's codebases, issue trackers, and discussion forums is expected to follow the [PSF Code of Conduct](https://www.python.org/psf/conduct/).\n\n## License\nThe software tools of this project are under the GNU License - see the [LICENSE](LICENSE) file for details.\n",
    'author': 'Daniel León-Periñán',
    'author_email': 'daniel.leonperinan@mdc-berlin.de',
    'maintainer': 'Daniel León-Periñán',
    'maintainer_email': 'daniel.leonperinan@mdc-berlin.de',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.9,<3.13',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)

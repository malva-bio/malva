Installation
============

Malva Tools are distributed in two formats:

- **Python wheel**: For using Malva as a Python library in notebooks and scripts
- **Apptainer container**: For running CLI commands on HPC systems

Choose the installation method that fits your use case, or install both.

Availability
------------

Malva Tools are provided free of charge for academic non-profit research.

**Download**

Pre-built binaries are available upon request.

The distribution includes both the Python wheel and the Apptainer container.

**Access Request**

For access, contact daniel.leonperinan@mdc-berlin.de with your name, institution, and intended use case.

System Requirements
-------------------

**Supported Platform**
    Linux (Ubuntu 22.04+, CentOS 7+, or similar)

**Python Version**
    Python 3.11 (for wheel installation)

**Hardware**
    - Minimum: 2 GB RAM, 1 CPU core
    - Recommended: 8 GB RAM, 2+ CPU cores for large datasets
    - Storage: Sufficient space for raw data and indices

Option 1: Python Wheel (Recommended)
------------------------------------

Install the Python wheel to use Malva in notebooks and Python scripts. This is the recommended method for most users.

1. Download the wheel file (upon request)

2. Install with pip:

.. code-block:: bash

   pip install malva-0.2.0-cp311-cp311-linux_x86_64.whl

3. Verify the installation:

.. code-block:: python

   import malva
   from malva.indexes import MalvaIndex

This method enables:

- Importing Malva in Python scripts and Jupyter notebooks
- Using the Python API for programmatic queries
- Running CLI commands via ``malva``

Option 2: Apptainer Container
-----------------------------

Use the Apptainer container for running CLI commands, especially on HPC systems where you may not have control over the Python environment.

**Prerequisites**

Apptainer (formerly Singularity) must be available. Check with:

.. code-block:: bash

   apptainer --version

If not available, ask your system administrator or install following the `official guide <https://apptainer.org/docs/admin/main/installation.html>`_.

**Setup**

1. Download and extract the distribution:

.. code-block:: bash

   wget https://bimsbstatic.mdc-berlin.de/rajewsky/malva/releases/malva_dist.tar.gz
   tar -xzvf malva_dist.tar.gz
   cd malva_dist

2. Verify the installation:

.. code-block:: bash

   ./malva --version
   ./malva --help

**Distribution Contents**

.. code-block:: text

   malva_dist/
   ├── python.sif        # Apptainer container with Python runtime
   ├── site-packages/    # Malva package and dependencies
   ├── malva             # CLI wrapper script
   └── malva_runner.py   # Helper for Python/Jupyter integration

HPC Configuration
-----------------

On HPC systems, you may need to configure Apptainer to access data directories outside the default bind paths.

**Binding additional paths**

If Malva cannot find your data files (``FileNotFoundError``), edit the ``malva`` wrapper script to bind your data directories:

.. code-block:: bash

   #!/bin/bash
   DIST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   exec apptainer exec \
       --bind "$(pwd):$(pwd)" \
       --bind "/data:/data" \
       --bind "$DIST:$DIST" \
       --env PYTHONPATH="$DIST/site-packages" \
       --pwd "$(pwd)" \
       "$DIST/python.sif" \
       python -m malva "$@"

Replace ``/data:/data`` with your institution's data path (e.g., ``/scratch:/scratch`` or ``/home:/home``).

**Symlinks**

If your input files are symlinks, ensure the symlink target directory is also bound. Alternatively, use the resolved path (``readlink -f <symlink>``).

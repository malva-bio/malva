Installation
============

Malva Tools are distributed as pre-built binaries with an Apptainer container for dependency management. This ensures reproducibility across different systems.

Availability
------------

Malva Tools are provided free of charge for academic non-profit research.

**Download**

Pre-built binaries are available upon request.

**Access Request**

For access, contact daniel.leonperinan@mdc-berlin.de with your name, institution, and intended use case.

System Requirements
-------------------

**Supported Platform**
    Linux (Ubuntu 22.04+, CentOS 7+, or similar)

**Required Software**
    Apptainer (formerly Singularity), available on most HPC systems

**Hardware**
    - Minimum: 2 GB RAM, 1 CPU core
    - Recommended: 8 GB RAM, 2+ CPU cores for large datasets
    - Storage: Sufficient space for raw data and indices

Installing Apptainer
--------------------

Apptainer is pre-installed on most HPC clusters. Check availability with:

.. code-block:: bash

   apptainer --version

If not available, ask your system administrator or install following the `official guide <https://apptainer.org/docs/admin/main/installation.html>`_.

For local installation on Ubuntu/Debian:

.. code-block:: bash

   sudo apt update
   sudo apt install -y apptainer

Distribution Contents
---------------------

After receiving access, you will download a distribution package containing:

.. code-block:: text

   malva_dist/
   ├── python.sif        # Apptainer container with Python runtime
   ├── site-packages/    # Malva package and dependencies
   ├── malva             # CLI wrapper script
   └── malva_runner.py   # Helper for Python/Jupyter integration

Setup
-----

1. Download and extract the distribution:

.. code-block:: bash

   tar -xzvf malva_dist.tar.gz
   cd malva_dist

2. Verify the installation:

.. code-block:: bash

   ./malva --version
   ./malva --help

Additional Configuration
-----------------

You may need to configure Apptainer to access data directories outside the default bind paths.

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
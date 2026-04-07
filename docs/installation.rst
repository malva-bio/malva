Installation
============

Malva is distributed in two formats. Choose the one that fits your use case, or install both.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: :octicon:`package;1.5em` Python Wheel
        :class-card: sd-border-0 sd-shadow-sm

        **Recommended for most users**

        Use Malva in notebooks and Python scripts. Full API access for programmatic queries.

    .. grid-item-card:: :octicon:`container;1.5em` Apptainer Container
        :class-card: sd-border-0 sd-shadow-sm

        **Best for HPC systems**

        Run CLI commands without managing Python environments. Works on shared clusters.

----

Availability
------------

.. note::

   Malva is provided **free of charge** for academic non-profit research.

   Code and pre-built binaries (wheel + container) are available via pip and in our [GitHub repository](https://github.com/malva-bio/malva)

**To request access:** Contact daniel.leonperinan@mdc-berlin.de with your name, institution, and intended use case.

----

System Requirements
-------------------

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: :octicon:`device-desktop;1em` Platform
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Linux (Ubuntu 22.04+, CentOS 7+)

    .. grid-item-card:: :octicon:`code;1em` Python
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Python 3.11 (for wheel)

    .. grid-item-card:: :octicon:`cpu;1em` Hardware
        :class-card: sd-border-0 sd-shadow-sm sd-text-center

        Min: 2 GB RAM, 1 CPU

----

Option 1: Python Wheel
----------------------

.. tip::

   This is the **recommended** method for most users. It enables full Python API access.

**Step 1: Download the wheel file** (provided upon request)

**Step 2: Install with pip**

.. code-block:: bash

   pip install malva-0.2.0-cp311-cp311-linux_x86_64.whl

**Step 3: Verify the installation**

.. code-block:: python

   import malva
   from malva.malva_index import MalvaIndex

**What you can do:**

- Import Malva in Python scripts and Jupyter notebooks
- Use the Python API for programmatic queries
- Run CLI commands via ``malva`` or ``python -m malva``

----

Option 2: Apptainer Container
-----------------------------

Use this option on HPC systems where you may not have control over the Python environment.

**Prerequisites**

Apptainer (formerly Singularity) must be available:

.. code-block:: bash

   apptainer --version

If not available, ask your system administrator or see the `Apptainer installation guide <https://apptainer.org/docs/admin/main/installation.html>`_.

**Setup**

1. Download and extract the distribution (provided upon request):

.. code-block:: bash

   tar -xzvf malva_dist.tar.gz
   cd malva_dist

2. Verify the installation:

.. code-block:: bash

   ./malva --version
   ./malva --help

**Distribution contents:**

.. code-block:: text

   malva_dist/
   ├── python.sif        # Apptainer container with Python runtime
   ├── site-packages/    # Malva package and dependencies
   ├── malva             # CLI wrapper script
   └── malva_runner.py   # Helper for Python/Jupyter integration

----

HPC Configuration
-----------------

.. warning::

   On HPC systems, Apptainer may not have access to all data directories by default.

**Binding additional paths**

If you get ``FileNotFoundError`` for files that exist, edit the ``malva`` wrapper script to bind your data directories:

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

Replace ``/data:/data`` with your institution's data path (e.g., ``/scratch:/scratch``).

**Symlinks**

If your input files are symlinks, ensure the symlink target directory is also bound, or use the resolved path:

.. code-block:: bash

   readlink -f /path/to/symlink

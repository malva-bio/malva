#!/bin/bash

# Clean previous build
rm -rf _build/
rm -rf api/

# Generate API documentation
sphinx-apidoc -o api/ ../malva --force --separate --no-toc

# Write the top-level API reference page, including the algorithmic framework
cat > api/modules.rst << 'EOF'
API Reference
=============

This section documents the Malva Python API and the full algorithmic framework underlying index
construction and querying. The algorithmic documentation is intended to enable re-implementation
and reproducibility; the Python API exposes the components available via the open-source
distribution.

----

Algorithmic Framework
---------------------

The indexing and query algorithms are fully documented below, including data structures, pseudocode,
and default parameters.

.. toctree::
   :maxdepth: 2

   ../indexing_framework

----

Python Modules
--------------

The following modules are available in the open-source Python distribution:

.. toctree::
   :maxdepth: 4

   malva.indexes
   malva.fastq_processing
   malva.dbutils
   malva.spacemake
   malva.utils
EOF

# Build HTML documentation
sphinx-build -b html . _build/html

echo "Documentation built in _build/html/"
echo "Open _build/html/index.html in your browser"
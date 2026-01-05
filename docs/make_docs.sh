#!/bin/bash

# Clean previous build
rm -rf _build/
rm -rf api/

# Generate API documentation
sphinx-apidoc -o api/ ../malva --force --separate --no-toc

# Ensure api/modules.rst exists with correct content
cat > api/modules.rst << 'EOF'
API Reference
=============

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
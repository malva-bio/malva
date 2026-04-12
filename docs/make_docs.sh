#!/bin/bash

# Clean previous build
rm -rf _build/
rm -rf api/

# Generate API documentation
sphinx-apidoc -o api/ ../malva --force --separate --no-toc

# Write the top-level API reference page
cat > api/modules.rst << 'EOF'
Python Modules
==============

The following modules are available in the open-source Python distribution.

.. toctree::
   :maxdepth: 1

   malva
EOF

# Build HTML documentation
sphinx-build -b html . _build/html

echo "Documentation built in _build/html/"
echo "Open _build/html/index.html in your browser"

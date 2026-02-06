#!/bin/bash
# Build wheels locally for testing

set -e

# Clean previous builds
rm -rf build/ dist/ wheelhouse/ *.egg-info
find . -name "*.so" -delete
find . -name "*.cpp" -path "*/malva/*" -delete

# Install cibuildwheel
pip install cibuildwheel

# Build for current platform only (fast testing)
export CIBW_BUILD="cp311-manylinux_x86_64"
export CIBW_MANYLINUX_X86_64_IMAGE="manylinux_2_28"
export CIBW_BEFORE_BUILD="pip install numpy cython"
export CIBW_ENVIRONMENT="MALVA_DEBUG_BUILD=0"

# Build
python -m cibuildwheel --platform linux --output-dir wheelhouse

echo "Wheels built in ./wheelhouse/"
ls -la wheelhouse/
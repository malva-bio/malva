#!/bin/bash
# build_with_apptainer.sh

set -e

PROJECT_DIR="$(pwd)"
OUTPUT_DIR="${PROJECT_DIR}/wheelhouse"
MANYLINUX_IMAGE="quay.io/pypa/manylinux_2_28_x86_64:2025.12.13-1"

mkdir -p "$OUTPUT_DIR"

# Pull the manylinux image if needed
if [ ! -f manylinux_2_28.sif ]; then
    echo "Pulling manylinux image..."
    apptainer pull manylinux_2_28.sif docker://${MANYLINUX_IMAGE}
fi

echo "Building wheel inside container..."
apptainer exec \
    --cleanenv \
    --no-home \
    --bind "${PROJECT_DIR}:/project" \
    --bind "${OUTPUT_DIR}:/output" \
    --pwd /project \
    manylinux_2_28.sif \
    /bin/bash -c '
        set -e
        
        PYTHON=/opt/python/cp311-cp311/bin/python
        PIP=/opt/python/cp311-cp311/bin/pip
        
        # Ensure we use container paths only
        export HOME=/tmp
        export PYTHONUSERBASE=/tmp/python-user
        export PATH="/opt/python/cp311-cp311/bin:$PATH"
        
        echo "=== Python version ==="
        $PYTHON --version
        
        echo "=== Installing build dependencies ==="
        rm -rf /tmp/python-user/*
        $PIP install --upgrade "pip<26.0" "wheel<0.46.2" "setuptools<80.10.1"
        $PIP install "cython>=3.0" "numpy>=1.21"
        
        echo "=== Cleaning previous builds ==="
        rm -rf build/ dist/ *.egg-info
        find . -name "*.so" -delete
        find . -name "*.cpp" -path "*/malva/*" -delete
        
        echo "=== Setting environment ==="
        export MALVA_DEBUG_BUILD=0
        
        echo "=== Building extensions in-place first (to verify compilation) ==="
        $PYTHON setup.py build_ext --inplace
        
        echo "=== Checking if .so files were created ==="
        find malva -name "*.so" -ls
        
        echo "=== Now building the wheel ==="
        $PYTHON setup.py bdist_wheel
        
        echo "=== Contents of dist/ ==="
        ls -la dist/
        
        echo "=== Inspecting wheel contents ==="
        for whl in dist/*.whl; do
            echo "Contents of $whl:"
            $PYTHON -m zipfile -l "$whl" | grep -E "\.(so|py)$" | head -50
        done
        
        echo "=== Repairing wheel for manylinux compliance ==="
        for whl in dist/*.whl; do
            auditwheel show "$whl" || true
            auditwheel repair "$whl" -w /output/
        done
        
        echo "=== Final wheel contents ==="
        for whl in /output/*.whl; do
            echo "Contents of $whl:"
            $PYTHON -m zipfile -l "$whl" | grep -E "\.(so|py)$" | head -50
        done
        
        echo "=== Testing import in isolated environment ==="
        # Create a fresh virtual environment for testing
        $PYTHON -m venv /tmp/test_env
        source /tmp/test_env/bin/activate
        
        # Install the wheel and its dependencies
        pip install /output/*.whl
        
        # Test imports
        python -c "import malva; print(\"malva imported OK\")"
        python -c "from malva.kmer_processing import encode_kmer; print(\"kmer_processing OK\")"
        python -c "from malva.indexes import MalvaIndex; print(\"indexes OK\")" || echo "indexes import failed (may need runtime deps)"
        
        deactivate
        
        echo ""
        echo "Build complete!"
    '

echo ""
echo "============================================"
echo "Wheels built successfully in ${OUTPUT_DIR}/"
echo "============================================"
ls -la "${OUTPUT_DIR}/"

echo ""
echo "To test locally, run:"
echo "  pip install ${OUTPUT_DIR}/malva-*.whl"
#!/bin/bash
#
# build_malva_distribution.sh
#
# Creates a malva distribution using a pre-built Python container.
# No root required. Works on shared servers.
#
# Output:
#   malva_dist/
#   ├── python.sif      # Base Python container
#   ├── site-packages/  # Malva + dependencies
#   └── malva           # Wrapper script
#
# Usage:
#   ./build_malva_distribution.sh [/path/to/wheel.whl]
#

set -e

WHEEL_PATH="$1"

if [ -z "$WHEEL_PATH" ]; then
    WHEEL_PATH=$(find ./wheelhouse ./dist . -maxdepth 1 -name "malva-*.whl" 2>/dev/null | head -1)
fi

if [ ! -f "$WHEEL_PATH" ]; then
    echo "Error: No malva wheel found"
    echo "Usage: $0 [/path/to/malva-*.whl]"
    exit 1
fi

WHEEL_PATH=$(realpath "$WHEEL_PATH")

echo "Building malva distribution"
echo "  Wheel: $WHEEL_PATH"

# Setup
DIST_DIR="$(pwd)/malva_dist"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/site-packages"

# Pull base image
echo ""
echo "Pulling Python container..."
apptainer pull "$DIST_DIR/python.sif" docker://python:3.11-bookworm

# Install packages
echo ""
echo "Installing malva..."
WHEEL_DIR=$(dirname "$WHEEL_PATH")
apptainer exec \
    --bind "$DIST_DIR:$DIST_DIR" \
    --bind "$WHEEL_DIR:$WHEEL_DIR" \
    "$DIST_DIR/python.sif" \
    pip install --target "$DIST_DIR/site-packages" --no-cache-dir "$WHEEL_PATH"

# Verify
echo ""
echo "Verifying..."
apptainer exec \
    --bind "$DIST_DIR:$DIST_DIR" \
    --env PYTHONPATH="$DIST_DIR/site-packages" \
    "$DIST_DIR/python.sif" \
    python -c "import malva; print('malva:', malva.__file__)"

# Create wrapper
cat > "$DIST_DIR/malva" << 'EOF'
#!/bin/bash
DIST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec apptainer exec \
    --bind "$(pwd):$(pwd)" \
    --bind "$DIST:$DIST" \
    --env PYTHONPATH="$DIST/site-packages" \
    --pwd "$(pwd)" \
    "$DIST/python.sif" \
    python -m malva "$@"
EOF
chmod +x "$DIST_DIR/malva"

# Create Python helper
cat > "$DIST_DIR/malva_runner.py" << 'EOF'
"""
Use malva from Python/Jupyter:

    from malva_runner import malva, malva_python
    
    malva("--version", capture_output=True)
    malva_python("from malva.kmer_processing import encode_kmer; print(encode_kmer('ACGT'))")
"""
import subprocess, os
from pathlib import Path

DIST = Path(__file__).parent.resolve()

def _cmd_apptainer(extra):
    return [
        "apptainer", "exec",
        "--bind", f"{os.getcwd()}:{os.getcwd()}",
        "--bind", f"{DIST}:{DIST}",
        "--env", f"PYTHONPATH={DIST}/site-packages",
        "--pwd", os.getcwd(),
        str(DIST / "python.sif")
    ] + extra

def _cmd_docker(extra):
    return [
        "docker", "run", "--rm",
        "-v", f"{os.getcwd()}:{os.getcwd()}",
        "-v", f"{DIST}:{DIST}",
        "-w", os.getcwd(),
        "-e", f"PYTHONPATH={DIST}/site-packages",
        "python:3.11",
    ] + extra

def _get_cmd(extra):
    # Use apptainer if available, else docker
    import shutil
    if shutil.which("apptainer") and (DIST / "python.sif").exists():
        return _cmd_apptainer(extra)
    else:
        return _cmd_docker(extra)

def malva(*args, capture_output=False):
    return subprocess.run(_get_cmd(["python", "-m", "malva"] + list(args)), capture_output=capture_output, text=True)

def malva_python(code, capture_output=True):
    return subprocess.run(_get_cmd(["python", "-c", code]), capture_output=capture_output, text=True)

def malva_script(path, *args, capture_output=False):
    p = Path(path).resolve()
    cmd = _get_cmd(["python", str(p)] + list(args))
    # Add bind for script directory
    if "apptainer" in cmd[0]:
        cmd.insert(4, "--bind")
        cmd.insert(5, f"{p.parent}:{p.parent}")
    else:
        cmd.insert(4, "-v")
        cmd.insert(5, f"{p.parent}:{p.parent}")
    return subprocess.run(cmd, capture_output=capture_output, text=True)
EOF

# Create macOS/Docker wrapper
cat > "$DIST_DIR/malva-docker" << 'EOF'
#!/bin/bash
DIST="$(cd "$(dirname "$0")" && pwd)"
docker run --rm \
    -v "$(pwd):$(pwd)" \
    -v "$DIST:$DIST" \
    -w "$(pwd)" \
    -e PYTHONPATH="$DIST/site-packages" \
    python:3.11 python -m malva "$@"
EOF
chmod +x "$DIST_DIR/malva-docker"

# Test
echo ""
"$DIST_DIR/malva" --version

echo ""
echo "====================================="
echo "Done!"
echo "====================================="
echo ""
ls -lh "$DIST_DIR/"
echo ""
echo "Usage:"
echo "  ./malva_dist/malva --help"
echo "  ./malva_dist/malva --version"
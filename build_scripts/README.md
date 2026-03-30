# Building malva wheels

Pre-built wheels for Linux (manylinux) and macOS are distributed via the
[GitHub Releases page](https://github.com/malva-bio/malva/releases).
Use the scripts in this directory only when you need to produce wheels locally
(e.g. to cut a new release or test a platform).

## Requirements

- Ubuntu ≥ 22.04 (for Linux wheels)
- Apptainer **or** Docker
- Python ≥ 3.9

## Build modes

The `build.py` at the project root reads one environment variable:

| Variable | Value | Effect |
|---|---|---|
| `MALVA_DEBUG_BUILD` | `0` (default) | `-O3`, no tracing, symbol stripping — use for distribution |
| `MALVA_DEBUG_BUILD` | `1` | `-O0`, Cython tracing/profiling enabled — use for development |

## Scripts

### `build_with_apptainer.sh` — recommended for releases

Builds a manylinux-compliant wheel inside an Apptainer container.
Runs `auditwheel repair` to bundle required shared libraries and produces a
portable wheel in `wheelhouse/`.

```bash
cd /path/to/malva_core
bash build_scripts/build_with_apptainer.sh
# → wheelhouse/malva-*.whl
```

### `build_with_docker.sh` — quick local test via cibuildwheel

Uses [cibuildwheel](https://cibuildwheel.readthedocs.io) and Docker.
Faster for iteration; requires Docker to be running.

```bash
cd /path/to/malva_core
bash build_scripts/build_with_docker.sh
# → wheelhouse/malva-*.whl
```

### `distribute_container.sh` — container-based distribution

For HPC environments without pip access. Installs a wheel into an Apptainer
container and produces a self-contained `malva_dist/` bundle with a `./malva`
wrapper script that requires no Python on the host.

```bash
bash build_scripts/distribute_container.sh wheelhouse/malva-*.whl
# → malva_dist/malva  (run with ./malva_dist/malva --help)
```

## Testing the wheel

```bash
python -m venv /tmp/malva_test && source /tmp/malva_test/bin/activate
pip install wheelhouse/malva-*.whl
malva --version
deactivate
```

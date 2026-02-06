Requirements: 
- A machine running Ubuntu >=22.04
- Apptainer (or Docker)

To build the binaries, we need to run `build_with_*`, depending on the system

Use `build_with_apptainer.sh` because this has the dependencies all there,
we don't depend on the environment in the `malva` server

To generate the container, run `distribute_container.sh`
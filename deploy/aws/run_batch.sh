#!/bin/bash
# Batch entrypoint wrapper — sources OpenFOAM and runs the Python entrypoint
source /opt/openfoam12/etc/bashrc
exec python /app/deploy/aws/batch_entrypoint.py "$@"

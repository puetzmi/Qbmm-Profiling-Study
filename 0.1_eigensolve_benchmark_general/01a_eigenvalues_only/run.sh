#!/usr/bin/env bash
# This script runs the general eigensolver benchmark application using the run-script in the parent directory with `EIGVECS=false`, i.e. computing only eigenvalues.
# It must be ensured that the current working directory contains the proper executable (or a link to it) and that environment variables such as library paths are set properly.


export EIGVECS=false            # compute eigenvalues only
$(dirname $0)/../run.sh         # run case
unset EIGVECS

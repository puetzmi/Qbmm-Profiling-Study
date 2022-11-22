#!/usr/bin/bash
# This script runs the general eigensolver benchmark application using the parameters below.
# It must be ensured that the current working directory contains the proper executable (or a link to it) and that environment variables such as library paths are set properly.

## PARAMETERS
NMIN=2                  # minumum matrix size
NMAX=100                # maximum matrix size
NMAT=1000               # number of matrices per size
NEXEC=1000              # number of executions per matrix
OUTFILE=cpu_times.out   # output filename
EIGVECS=false           # compute eigenvalues only


## DISABLE MULTI-THREADING
export MKL_DYNAMIC="FALSE"
export MKL_NUM_THREADS=1
export OMP_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1


## EXECUTE
./eigensolver_benchmark_general n_min=$NMIN n_max=$NMAX n_matrices=$NMAT n_exec=$NEXEC outfile=$OUTFILE random_seed=$RANDSEED compute_eigenvectors=$EIGVECS

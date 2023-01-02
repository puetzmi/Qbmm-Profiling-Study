#!/usr/bin/env bash

# This script executes the 'core_inversion_benchmark' application in the current working directory using the setup file 'setup.inp' (also in the working directory) as well as the parameters below. Besides the presence of the test file 'setup.inp' and the executable 'core_inversion_benchmark' in the working directory, it must be ensured that the data directory (see parameter `DATA_DIR`) exists and that it contains all the necessary input files. The only command line parameter required is the number of moments, which must be an even positive integer.

## PARAMETERS
N_MOMSETS=100000
N_EXEC=10000
JACOBI_DIAG_FILE_PREFIX=alpha-coeffs_nmom
JACOBI_DIAG_FILE_SUFFIX=.dat
JACOBI_SDIAG_FILE_PREFIX=gamma-coeffs_nmom
JACOBI_SDIAG_FILE_SUFFIX=.dat
MOMENTS_FILE_PREFIX=moments_nmom
MOMENTS_FILE_SUFFIX=.dat
OUTFILE_PREFIX=""
SETUP_FILE=setup.inp
DATA_DIR=$(dirname -- $0)/../data

N_MOMS=$1
MORE_ARGS=${@:2}

## DISABLE MULTI-THREADING
export MKL_DYNAMIC="FALSE"
export MKL_NUM_THREADS=1
export OMP_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1

## CONCATENATE PARTS OF FILENAMES
JACOBI_DIAG_FILE="${DATA_DIR}/${JACOBI_DIAG_FILE_PREFIX}${N_MOMS}${JACOBI_DIAG_FILE_SUFFIX}"
JACOBI_SDIAG_FILE="${DATA_DIR}/${JACOBI_SDIAG_FILE_PREFIX}${N_MOMS}${JACOBI_SDIAG_FILE_SUFFIX}"
MOMENTS_FILE="${DATA_DIR}/${MOMENTS_FILE_PREFIX}${N_MOMS}${MOMENTS_FILE_SUFFIX}"

## EXECUTE
./qmom_core_inversion n_exec=$N_EXEC n_moms=$N_MOMS n_momsets=$N_MOMSETS setup_file=$SETUP_FILE \
    moments_file=$MOMENTS_FILE jacobi_diagonal_file=$JACOBI_DIAG_FILE jacobi_superdiagonal_file=$JACOBI_SDIAG_FILE \
    outfile_prefix=$OUTFILE_PREFIX \
    $MORE_ARGS

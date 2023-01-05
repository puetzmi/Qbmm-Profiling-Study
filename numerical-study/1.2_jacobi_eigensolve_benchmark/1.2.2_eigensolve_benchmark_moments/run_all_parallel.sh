#!/usr/bin/env bash

# This script starts a parallel run using the `run.sh` script in this directory and command line arguments, where the first is the number of processors used and the second a range specifying the number of moments.

# Example:
# --------
# Presuming that this script is in the parent directory of the run directory (containing the `qmom_eigensolver` executable and the setup file `setup.inp`), the command
#   `../run_all_parallel 4 4:20`
# runs the application with 4,6,..,18,20 moments using 4 cores simulataneously.
#
# It must be ensured by the user that the used processor has at least the number of cores passed as a parameter.
#

## PARAMETERS
N_PROCS=$1
MOM_RANGE=(${2//:/ })
N_MOM_MIN=${MOM_RANGE[0]}
N_MOM_MAX=${MOM_RANGE[1]}
SCRIPT_DIR=$(dirname -- $0)
MORE_ARGS=${@:3}

N_JOBS="\j"


## PARALLEL RUN ON $N_PROCS CORES
for ((N_MOM=$N_MOM_MIN; N_MOM<=$N_MOM_MAX; N_MOM=N_MOM+2)); do
    while (( ${N_JOBS@P} >= N_PROCS )); do
        wait -n
    done
    $SCRIPT_DIR/run.sh $N_MOM $MORE_ARGS >& nmom_$N_MOM.log &
    sleep 1
done
wait

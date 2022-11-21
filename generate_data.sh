#!/usr/bin/env bash

# This script generates the data directory with data of moment sets of 4 to 20 moments (100000 each) using the parameters below.

## PARAMETERS
NMOM_MIN=4
NMOM_MAX=20
NSETS=100000
OUTFILE_EXT=.dat
GAMMA=0.5
DELTA=1
OUTPUT_DIR=data
SEED=125125

## EXECUTE
for ((NMOM=$NMOM_MIN; NMOM<=$NMOM_MAX; NMOM=NMOM+2)); do
    python3 ../python_tools/generate_random_moments.py nmom=$NMOM nsets=$NSETS outfile-suffix=_nmom$NMOM$OUTFILE_EXT gamma=$GAMMA delta=$DELTA output-dir=$OUTPUT_DIR random-seed=$SEED &
done

#!/bin/bash
#BSUB -nnodes 1
#BSUB -W 10
#BSUB -P CFD126
#BSUB -alloc_flags "smt4"
#BSUB -J zefr
#BSUB -o out_test.%J
#BSUB -e out_test.%J
#BSUB -q batch

# Load modules
module load gcc cuda essl hdf5 metis

# Launch job
jsrun --smpiargs="-gpu" -n 1 -g 6 -c 42 -a 6 ./zefr input_file

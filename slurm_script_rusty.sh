#!/bin/bash
#SBATCH -N1 --ntasks-per-node=96
#SBATCH -t 00-22:00:00
##SBATCH -p ccb
#SBATCH -C genoa
#SBATCH -o cyst_log.out

# Start from an "empty" module collection.
module purge
# Load in what we need to execute mpirun.
module load julia/1.11.2

# We assume this executable is in the directory from which you ran sbatch.
snakemake -c2 -j 48 --keep-going

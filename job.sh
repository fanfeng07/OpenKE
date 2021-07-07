#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=kraken_fast

WORKDIR=/home/user/feng/$SLURM_JOB_ID
mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit -1 

touch test.txt

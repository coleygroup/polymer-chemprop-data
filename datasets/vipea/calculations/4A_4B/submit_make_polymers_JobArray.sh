#!/bin/bash

#SBATCH --job-name=mkpoly
#SBATCH --output=slurm-%j-%x-%a.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --array 0-681

# Loading the required module
source /etc/profile
module load anaconda/2022a
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/etc/profile.d/conda.sh
source activate rdkit

python ../make-polymers.py --acids ../../acids.csv --bromides ../../bromides.csv --iAs 0 --iBs $SLURM_ARRAY_TASK_ID --nconfs 8 --nrandom 32 --ncpu $SLURM_CPUS_PER_TASK

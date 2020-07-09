#!/bin/bash
#
#SBATCH --job-name=target_no_view
#SBATCH --output=target_no_view.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1000
#
#SBATCH --array=0-7

FILES=(/path/to/data/*)

srun ./my_program.exe ${FILES[$SLURM_ARRAY_TASK_ID]}
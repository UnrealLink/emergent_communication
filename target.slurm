#!/bin/bash
#
#SBATCH --job-name=target_no_view
#SBATCH --output=target_no_view.txt
# SBATCH --chdir=/gpfs/home/vvillecroze/emergent_communication
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1000
#
#SBATCH --array=0-4

# useful informations to print
echo "#############################" 
echo "User:" $USER
echo "Date:" `date`
echo "Host:" `hostname`
echo "Directory:" `pwd`
echo "SLURM_JOBID:" $SLURM_JOBID
echo "SLURM_SUBMIT_DIR:" $SLURM_SUBMIT_DIR
echo "SLURM_JOB_NODELIST:" $SLURM_JOB_NODELIST
echo "#############################" 

#############################

gpfs/home/vvillecroze/miniconda3/envs/custom_ssd/bin/python train/train_a3c.py --env target --lr 0.00001 --agents 2 \
    --save target_easy$SLURM_ARRAY_TASK_ID --processes 16 --view-size 0 --hidden 64 --communication --seed $SLURM_ARRAY_TASK_ID \
    --horizon 20000000
 
echo "Job finished"

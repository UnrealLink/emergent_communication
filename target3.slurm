#!/bin/bash
#
#SBATCH --job-name=target3
#SBATCH --output=outputs/target3.txt
#SBATCH --chdir=/gpfs/home/vvillecroze/emergent_communication
#
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=16
#
#SBATCH --array=0-9

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

/gpfs/home/vvillecroze/miniconda3/envs/custom_ssd/bin/python train/train_a3c.py \
    --env target --save easy_speaker_bias_view_3_$SLURM_ARRAY_TASK_ID --view-size 3 \
    --seed $SLURM_ARRAY_TASK_ID --horizon 200000000 --cpu-only

echo "Job finished"
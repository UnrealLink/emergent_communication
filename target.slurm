#!/bin/bash
#
#SBATCH --job-name=target
#SBATCH --output=outputs/target.txt
#SBATCH --chdir=/gpfs/home/vvillecroze/emergent_communication
#
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
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

/gpfs/home/vvillecroze/miniconda3/envs/custom_ssd/bin/python train/train_a3c.py --env target --save target_easy_speaker_view_0_noise_0_$SLURM_ARRAY_TASK_ID --view-size 0 --communication --seed $SLURM_ARRAY_TASK_ID --horizon 3000000 --cpu-only --noise 0

echo "Job finished"

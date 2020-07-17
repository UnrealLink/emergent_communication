#!/bin/bash
#
#SBATCH --job-name=target0
#SBATCH --output=outputs/target0.txt
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

/gpfs/home/vvillecroze/miniconda3/envs/custom_ssd/bin/python train/train_a3c.py --env target \
    --save new_target_view_0_noise_0_$SLURM_ARRAY_TASK_ID --view-size 0 --communication \
    --seed $SLURM_ARRAY_TASK_ID --horizon 20000000 --cpu-only --noise 0
 
/gpfs/home/vvillecroze/miniconda3/envs/custom_ssd/bin/python train/train_a3c.py --env target \
    --save new_target_view_0_noise_3_$SLURM_ARRAY_TASK_ID --view-size 0 --communication \
    --seed $SLURM_ARRAY_TASK_ID --horizon 20000000 --cpu-only --noise 0.3

/gpfs/home/vvillecroze/miniconda3/envs/custom_ssd/bin/python train/train_a3c.py --env target \
    --save new_target_view_0_noise_5_$SLURM_ARRAY_TASK_ID --view-size 0 --communication \
    --seed $SLURM_ARRAY_TASK_ID --horizon 20000000 --cpu-only --noise 0.5

/gpfs/home/vvillecroze/miniconda3/envs/custom_ssd/bin/python train/train_a3c.py --env target \
    --save new_target_view_0_noise_7_$SLURM_ARRAY_TASK_ID --view-size 0 --communication \
    --seed $SLURM_ARRAY_TASK_ID --horizon 20000000 --cpu-only --noise 0.7

/gpfs/home/vvillecroze/miniconda3/envs/custom_ssd/bin/python train/train_a3c.py --env target \
    --save new_target_view_0_noise_9_$SLURM_ARRAY_TASK_ID --view-size 0 --communication \
    --seed $SLURM_ARRAY_TASK_ID --horizon 20000000 --cpu-only --noise 0.9

echo "Job finished"

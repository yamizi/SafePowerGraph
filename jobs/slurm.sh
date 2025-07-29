#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --mail-user=sghamizi@uni-koeln.de
#SBATCH --mail-type=END
#SBATCH --output=/scratch/sghamizi/logs/logfile-array-%j.out
#SBATCH --error=/scratch/sghamizi/logs/error-array-%j.out
#SBATCH --mem=100gb
#SBATCH --partition=smp
#SBATCH --array=0-2

# srun --pty --mem 150gb -c 8 -p interactive -t 20:30:00 -N 1 -G a30:1 -n 1   /bin/bash

module load lang/Python/3.9.5-GCCcore-10.3.0
source /scratch/sghamizi/robustgraph/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

EPSILON=0
RANDOMRESTART=0

sh jobs/grad_loop_fast.sh 0 0 "bus_vm#${SLURM_ARRAY_TASK_ID}" "attack_modelsV5_subset"
#sh jobs/grad_loop_fast.sh 0 0 "bus_vm#0" "attack_modelsV5_subset"


#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@gmail.com
#SBATCH -J "Case118_SAGE"
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -C volta
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module'command"
# You probably want to use more recent gpu/CUDA-optimized software builds
#module use /opt/apps/resif/iris/2019b/gpu/modules/all

python3 -m venv ~/venv/panda
source ~/venv/panda/bin/activate
module load lang/Python/3.8.6-GCCcore-10.2.0

cd ~/pandapower/
pip install -r requirements-all.txt
pip install protobuf==3.20.1

#export CUDA_VISIBLE_DEVICES=0
sh jobs/journal_extension/case118_gpu.sh "gat" 202406200
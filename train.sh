#!/bin/bash
#SBATCH --job-name=dqn_highway
#SBATCH --output=logs/slurm_%j_seed%a.out   # stdout per job
#SBATCH --error=logs/slurm_%j_seed%a.err    # stderr per job
#SBATCH --time=24:00:00                      # adjust based on your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0,1,2                        # one job per seed

source ~/.bashrc
conda activate your_env_name                 # replace with your env

mkdir -p logs checkpoints

python train.py --seed $SLURM_ARRAY_TASK_ID
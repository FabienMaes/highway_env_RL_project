#!/bin/bash
#SBATCH --partition=gpu_inter
#SBATCH --job-name=sb3_highway
#SBATCH --output=rlogs/slurm_sb3_%j.out
#SBATCH --error=rlogs/slurm_sb3_%j.err
#SBATCH --time=02:00:00

# 2. Activer l'environnement avec le chemin absolu
source /usr/users/rl_course_26/rl_course_26_27/my_rl_venv/bin/activate

mkdir -p checkpoints
python train_sb3.py

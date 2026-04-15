#!/bin/bash
#SBATCH --job-name=RL_Project
#SBATCH --partition=gpu_prod_long
#SBATCH --output=logs/sb3gamma%x_%A_%a.out
#SBATCH --error=logs/sb3gamma%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

# Load environment
source /usr/users/rl_course_26/rl_course_26_27/my_rl_venv/bin/activate

# 3. Créer les dossiers
mkdir -p logs checkpoints

# Run custom DQN (1500 episodes)
python dce_training.py --agent dqn --seed $SLURM_ARRAY_TASK_ID

# Run custom DDQN (1500 episodes)
python dce_training.py --agent ddqn --seed $SLURM_ARRAY_TASK_ID

# Run SB3 Baseline (35000 steps)
python dce_training.py --agent sb3 --seed $SLURM_ARRAY_TASK_ID
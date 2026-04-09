#!/bin/bash
#SBATCH --job-name=dqn_highway
#SBATCH --output=rlogs/all3_%j_seed%a.out
#SBATCH --error=rlogs/all3_%j_seed%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_prod_night
#SBATCH --cpus-per-task=4
#SBATCH --array=0

# Warning: you need a rlogs folder before running

# 2. Activer l'environnement avec le chemin absolu
source /usr/users/rl_course_26/rl_course_26_27/my_rl_venv/bin/activate

# 3. Créer les dossiers
mkdir -p logs checkpoints

# 4. Lancer le script
python train_ddqnsb3.py --seed $SLURM_ARRAY_TASK_ID --total_steps 60000 --sb3_steps 60000
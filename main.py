import os
import csv
import random
import argparse

import torch
import highway_env
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN as SB3_DQN
from stable_baselines3.common.logger import configure


from DQN import HighwayDQN
from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from utils import eval_agent, train, visualize_episode, plot_training_curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent on Highway-env")
    parser.add_argument("--model", type=str, choices=["baseline", "DQN", "DDQN"], default="DDQN",
                        help="Choose the model to train: 'baseline' (Stable-Baselines3), 'DQN', or 'DDQN'")
    args = parser.parse_args()
    model_choice = args.model

    train_seeds = [42, 123, 456]
    test_seeds = [3, 33, 333, 3333, 33333]
    summary_results = []
    agent = None

    print(f"\n{'='*80}")
    print(f"RUNNING PIPELINE FOR MODEL: {model_choice}")
    print(f"{'='*80}")

    for train_seed in train_seeds:
        print(f"\n{'='*40}")
        print(f"Starting Training for Seed: {train_seed}")
        print(f"{'='*40}")

        torch.manual_seed(train_seed)
        np.random.seed(train_seed)
        random.seed(train_seed)

        env = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG, render_mode='rgb_array')
        env.reset(seed=train_seed)

        action_space = env.action_space
        observation_space = env.observation_space

        if model_choice == "baseline":
            print(f"Training Stable-Baselines3 Baseline (Seed {train_seed})...")
            log_dir = f"logs_{model_choice}_seed_{train_seed}"
            os.makedirs(log_dir, exist_ok=True)
            sb3_logger = configure(log_dir, ["csv"])
            agent = SB3_DQN(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=5e-4,
                buffer_size=15_000,
                learning_starts=200,
                seed=train_seed
            )
            agent.set_logger(sb3_logger)
            agent.learn(total_timesteps=20_000, progress_bar=True)

            losses = []
            episode_rewards = []
            csv_file = os.path.join(log_dir, "progress.csv")

            if os.path.exists(csv_file):
                with open(csv_file, mode='r') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if 'train/loss' in row and row['train/loss']:
                            losses.append(float(row['train/loss']))
                        if 'rollout/ep_rew_mean' in row and row['rollout/ep_rew_mean']:
                            episode_rewards.append(float(row['rollout/ep_rew_mean']))

            plot_training_curves(episode_rewards, losses, train_seed, model_type=model_choice)
        else:
            gamma = 0.99
            batch_size = 32
            buffer_capacity = 15000
            update_target_every = 200

            epsilon_start = 1.0
            decrease_epsilon_factor = 10000
            epsilon_min = 0.05
            learning_rate = 5e-4

            agent = HighwayDQN(
                env,
                gamma,
                batch_size,
                buffer_capacity,
                update_target_every,
                epsilon_start,
                decrease_epsilon_factor,
                epsilon_min,
                learning_rate,
                model_type=model_choice
            )

            N_episodes = 200
            print(f"Training Custom {model_choice} (Seed {train_seed})...")

            losses, episode_rewards = train(env, agent, N_episodes, eval_every=10)
            plot_training_curves(episode_rewards, losses, train_seed, model_type=model_choice)

        save_path = f"models/{model_choice.lower()}_highway_seed_{train_seed}.pth"
        print(f"Saving the custom model to {save_path}...")
        agent.save(save_path)

        seeds_to_evaluate = [train_seed] + test_seeds
        eval_results = eval_agent(env, agent, num_episodes=50, seeds=seeds_to_evaluate, show_progress=True)

        for eval_seed, mean_r, std_r, crash_r in eval_results:
            summary_results.append((train_seed, eval_seed, mean_r, std_r, crash_r))

        visualize_episode(agent, seed=train_seed, record=True, output_dir=f"videos/{model_choice.lower()}")
        env.close()


    print(f"\n{'=' * 80}")
    print(f"FINAL EVALUATION TABLE: {model_choice} (Across Training and Test Seeds)")
    print(f"{'=' * 80}")
    print(f"{'Train Seed':<12} | {'Eval Seed':<15} | {'Reward':<15}")
    print("-" * 80)
    for train_s, eval_s, mean_r, std_r, crash_r in summary_results:
        marker = "(Train)" if train_s == eval_s else "(Test)"
        eval_str = f"{eval_s} {marker}"
        reward_str = f"{mean_r:.2f} +/- {std_r:.2f}"
        print(f"{train_s:<12} | {eval_str:<15} | {reward_str:<15} | {crash_r:<10.0%}")

    csv_path = f"evaluation_results_{model_choice.lower()}.csv"
    print(f"\nSaving numerical results to {csv_path}...")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Train Seed", "Eval Seed", "Evaluation Type", "Mean Reward", "Std Dev", "Crash Rate"])
        for train_s, eval_s, mean_r, std_r, crash_r in summary_results:
            eval_type = "Train" if train_s == eval_s else "Test"
            writer.writerow([train_s, eval_s, eval_type, round(mean_r, 2), round(std_r, 2), round(crash_r, 2)])

    if agent is not None:
        visualize_episode(agent)

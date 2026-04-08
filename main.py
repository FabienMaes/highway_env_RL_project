import csv
import random

import torch
import highway_env
import numpy as np
import gymnasium as gym

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID

from DQN import HighwayDQN
from utils import eval_agent, train, visualize_episode, plot_training_curves


if __name__ == "__main__":
    train_seeds = [42, 123, 456]
    test_seeds = [3, 33, 333, 3333, 33333]
    summary_results = []
    agent = None

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

        gamma = 0.99
        batch_size = 32
        buffer_capacity = 15000
        update_target_every = 200

        epsilon_start = 1.0
        decrease_epsilon_factor = 10000
        epsilon_min = 0.05
        learning_rate = 5e-4

        agent = HighwayDQN(
            action_space,       # type: ignore
            observation_space,  # type: ignore
            gamma,
            batch_size,
            buffer_capacity,
            update_target_every,
            epsilon_start,
            decrease_epsilon_factor,
            epsilon_min,
            learning_rate
        )

        N_episodes = 200
        losses, rewards = train(env, agent, N_episodes, eval_every=10)

        save_path = f"models/custom_dqn_highway_{train_seed}.pth"
        print(f"Saving the custom model to {save_path}...")
        agent.save(save_path)

        seeds_to_evaluate = [train_seed] + test_seeds
        eval_results = eval_agent(env, agent, num_episodes=50, seeds=seeds_to_evaluate, show_progress=True)

        for eval_seed, mean_r, std_r in eval_results:
            summary_results.append((train_seed, eval_seed, mean_r, std_r))

        plot_training_curves(rewards, losses, train_seed)

        visualize_episode(agent, seed=train_seed, record=True)

        env.close()


    print(f"\n{'='*55}")
    print("FINAL EVALUATION TABLE (Across Training and Test Seeds)")
    print(f"{'='*55}")
    print(f"{'Train Seed':<12} | {'Eval Seed':<15} | {'Reward':<15}")
    print("-" * 55)
    for train_s, eval_s, mean_r, std_r in summary_results:
        marker = "(Train)" if train_s == eval_s else "(Test)"
        eval_str = f"{eval_s} {marker}"
        reward_str = f"{mean_r:.2f} +/- {std_r:.2f}"
        print(f"{train_s:<12} | {eval_str:<15} | {reward_str:<15}")

    csv_path = "evaluation_results.csv"
    print(f"\nSaving numerical results to {csv_path}...")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Train Seed", "Eval Seed", "Evaluation Type", "Mean Reward", "Std Dev"])
        for train_s, eval_s, mean_r, std_r in summary_results:
            eval_type = "Train" if train_s == eval_s else "Test"
            writer.writerow([train_s, eval_s, eval_type, round(mean_r, 2), round(std_r, 2)])

    if agent is not None:
        visualize_episode(agent)


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
    test_seeds = [3, 33, 333]
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

    print(f"\n{'='*65}")
    print("FINAL EVALUATION TABLE (Across Training and Test Seeds)")
    print(f"{'='*65}")
    print(f"{'Train Seed':<12} | {'Eval Seed':<15} | {'Mean Reward':<15} | {'Std Dev':<10}")
    print("-" * 65)
    for train_s, eval_s, mean_r, std_r in summary_results:
        # Add a marker to clearly denote whether the agent had seen this seed before
        marker = "(Train)" if train_s == eval_s else "(Test)"
        eval_str = f"{eval_s} {marker}"
        print(f"{train_s:<12} | {eval_str:<15} | {mean_r:<15.2f} | {std_r:<10.2f}")

    if agent is not None:
        visualize_episode(agent)


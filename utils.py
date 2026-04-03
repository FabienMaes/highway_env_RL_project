from typing import Optional

import highway_env
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt


from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from DQN import HighwayDQN


def eval_agent(env: gym.Env, agent: HighwayDQN, num_episodes=5, seeds: Optional[list[int]] = None, show_progress=True) -> list[tuple[Optional[int], float, float]]:
    all_seed_results = []
    seeds_to_run = seeds if seeds is not None else [None]

    for seed in seeds_to_run:
        episode_rewards = []

        desc = f"Evaluating Seed {seed}" if seed is not None else "Evaluation"
        iterator = tqdm(range(num_episodes), desc=desc) if show_progress else range(num_episodes)

        for episode in iterator:
            reset_seed = seed + episode if seed is not None else None
            state, _= env.reset(seed=reset_seed)
            done = False
            total_reward = 0
            while not done:
                action = agent.get_action(state, epsilon=0.0)

                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward    # type: ignore
                done = terminated or truncated
            episode_rewards.append(total_reward)

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        if show_progress:
            seed_str = str(seed) if seed is not None else "N/A"
            print(f"Seed {seed_str} | Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        all_seed_results.append((seed, mean_reward, std_reward))

    return all_seed_results


def train(env: gym.Env, agent: HighwayDQN, N_episodes: int, eval_every=10) -> tuple[list[float], list[float]]:
    losses = []
    rewards_history = []

    pbar = tqdm(range(N_episodes), desc="Training DQN")
    for ep in pbar:
        done = False
        state, _ = env.reset()
        episode_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            is_done = terminated or truncated
            loss_val = agent.update(state, action, reward, is_done, next_state) # type: ignore

            state = next_state
            episode_reward += reward # type: ignore
            losses.append(loss_val)

            done = is_done

        rewards_history.append(episode_reward)
        pbar.set_postfix({
            "Reward": f"{episode_reward:.2f}",
            "Epsilon": f"{agent.epsilon:.2f}",
        })

        if ((ep + 1) % eval_every == 0):
            eval_results = eval_agent(env, agent, num_episodes=5, seeds=None, show_progress=False)
            mean_reward = eval_results[0][1]
            tqdm.write(f"Episode {ep + 1:04d} | Eval Mean Reward: {mean_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

    return losses, rewards_history


def visualize_episode(agent: HighwayDQN, seed: Optional[int] = None, record = False, output_dir = "videos") -> None:
    """Runs one greedy episode to visualize the agent's behavior, optionally saving it as an MP4 and returning it for display."""
    render_mode = "rgb_array" if record else "human"
    env = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG, render_mode=render_mode)

    if record:
        env = RecordVideo(env, video_folder=output_dir, name_prefix=f"agent_rollout_{seed}", disable_logger=True)

    state, _ = env.reset(seed=seed)
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        if not record:
            env.render()

        # Use epsilon=0.0 for pure exploitation (greedy policy)
        action = agent.get_action(state, epsilon=0.0)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward # type: ignore
        done = terminated or truncated
        step_count += 1

    print(f"Visualization Episode Reward: {total_reward:.2f}")
    env.close()
    if record:
        print(f"Saved video to {output_dir}/")


def plot_training_curves(episode_rewards: list[float], episode_losses: list[float], seed: int) -> None:
    """Plots and saves the rewards and losses in a side-by-side subplot."""
    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Reward curve
    axes[0].plot(episode_rewards, alpha=0.6, color='blue', label='Episode Reward')
    if len(episode_rewards) >= 20:
        rolling_avg = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
        axes[0].plot(range(19, len(episode_rewards)), rolling_avg, color='red', linewidth=2, label='20-ep Moving Avg')
    axes[0].set_title(f"Training curve: Episode rewards (Seed {seed})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].legend()

    # Loss curve
    axes[1].plot(episode_losses, color='orange', alpha=0.8)
    axes[1].set_title(f"Training curve: Loss (Seed {seed})")
    axes[1].set_xlabel("Update Steps")
    axes[1].set_ylabel("MSE loss")

    plt.tight_layout()
    plt.savefig(f"figures/dqn_training_curves_seed_{seed}.png")
    plt.close()


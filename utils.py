from typing import Optional

import highway_env
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from DQN import HighwayDQN


def eval_agent(env: gym.Env, agent: HighwayDQN | DQN, num_episodes: int = 50, seeds: Optional[list[int]] = None, show_progress: bool = True) -> list[tuple[Optional[int], float, float, float]]:
    """
    Evaluates an agent. If seeds is provided, it tests across those specific seeds.
    If seeds is None, it runs a quick unseeded evaluation (useful during training).
    Compatible with both Custom DQN and Stable-Baselines3 models.
    Returns: List of tuples containing (seed, mean_reward, std_reward, crash_rate)
    """
    if show_progress:
        print(f"Starting evaluation: {num_episodes} episodes per seed.")

    all_seed_results = []
    seeds_to_run = seeds if seeds is not None else [None]

    for seed in seeds_to_run:
        episode_rewards = []
        episode_crashes = []

        # Setup the iterator with an optional tqdm progress bar
        desc = f"Evaluating Seed {seed}" if seed is not None else "Evaluation"
        iterator = tqdm(range(num_episodes), desc=desc, leave=False) if show_progress else range(num_episodes)

        for episode in iterator:
            reset_seed = seed + episode if seed is not None else None
            obs, info = env.reset(seed=reset_seed)
            done = False
            total_reward = 0
            has_crashed = False

            while not done:
                if isinstance(agent, DQN):
                    action, _states = agent.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    action = agent.get_action(obs, epsilon=0.0)

                obs, reward, terminated, truncated, info = env.step(action)

                if info.get("crashed", False):
                    has_crashed = True

                total_reward += reward # type: ignore
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_crashes.append(1 if has_crashed else 0)

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        crash_rate = float(np.mean(episode_crashes))

        if show_progress:
            seed_str = str(seed) if seed is not None else "N/A"
            print(f"Seed {seed_str} | Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f} | Crash Rate: {crash_rate:.0%}")

        all_seed_results.append((seed, mean_reward, std_reward, crash_rate))

    return all_seed_results


def train(env: gym.Env, agent: HighwayDQN, N_episodes: int, eval_every: int = 10) -> tuple[list[float], list[float]]:
    losses = []
    episode_rewards_history = []

    pbar = tqdm(range(N_episodes), desc="Training DQN", leave=False)
    for ep in pbar:
        done = False
        state, _ = env.reset()
        episode_reward = 0
        has_crashed = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            if info.get("crashed", False):
                has_crashed = True

            is_done = terminated or truncated
            loss_val = agent.update(state, action, reward, is_done, next_state) # type: ignore

            state = next_state
            episode_reward += reward # type: ignore
            losses.append(loss_val)

            done = is_done

        episode_rewards_history.append(episode_reward)

        pbar.set_postfix({
            'Reward': f"{episode_reward:.2f}",
            'Epsilon': f"{agent.epsilon:.2f}",
            'Crashed': has_crashed
        })

        if ((ep + 1) % eval_every == 0):
            eval_results = eval_agent(env, agent, num_episodes=5, seeds=None, show_progress=False)
            quick_mean_reward = eval_results[0][1]
            quick_crash_rate = eval_results[0][3]
            tqdm.write(f"Episode {ep + 1:04d} | Eval Mean Reward: {quick_mean_reward:.2f} | Eval Crash Rate: {quick_crash_rate:.0%} | Epsilon: {agent.epsilon:.2f}")

    return losses, episode_rewards_history


def visualize_episode(agent: HighwayDQN | DQN, seed: Optional[int] = None, record = False, output_dir = "videos") -> None:
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

        if isinstance(agent, DQN):
            # logic for Stable-Baselines3 model
            action, _states = agent.predict(state, deterministic=True)
            action = int(action)
        else:
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


def plot_training_curves(episode_rewards: list[float], episode_losses: list[float], seed: int, model_type: str = "DQN") -> None:
    """Plots and saves the rewards and losses in a side-by-side subplot."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Reward curve
    axes[0].plot(episode_rewards, alpha=0.6, color='blue', label='Episode Reward')
    if len(episode_rewards) >= 20:
        rolling_avg = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
        axes[0].plot(range(19, len(episode_rewards)), rolling_avg, color='red', linewidth=2, label='20-ep Moving Avg')
    axes[0].set_title(f"{model_type} Training curve: Episode rewards (Seed {seed})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].legend()

    # Loss curve
    axes[1].plot(episode_losses, color='orange', alpha=0.8)
    axes[1].set_title(f"{model_type} Training curve: Loss (Seed {seed})")
    axes[1].set_xlabel("Update Steps")
    axes[1].set_ylabel("MSE loss")

    plt.tight_layout()
    plt.savefig(f"figures/{model_type.lower()}_training_curves_seed_{seed}.png")
    plt.close()

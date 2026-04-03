"""
DQN agent for highway-v0 — CentraleSupelec RL Project
======================================================
All tunable parameters are in CFG (dataclass at the top).
Structure:
  - CFG                : all hyperparameters in one place
  - MetricsLogger      : CSV logging for training + eval
  - ReplayBuffer       : experience replay
  - QNetwork           : neural network (Q-function)
  - select_action      : epsilon-greedy policy
  - train_step         : one gradient update
  - train_dqn          : full training loop
  - evaluate           : deterministic evaluation over N episodes
  - record_rollout     : save a GIF of one episode
  - train_sb3          : Stable-Baselines3 baseline (same interface)
  - plot_results        : side-by-side training curves
"""

import os
import csv
import random
import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
from collections import deque
from dataclasses import dataclass, field
from tqdm import tqdm
from stable_baselines3 import DQN as SB3_DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from IPython.display import Image
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG

# MAke environement
def make_env(render_mode="rgb_array"):
    env = gym.make("highway-v0", render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()
    env = Monitor(env) 
    return env


# =============================================================================
# CONFIGURATION — change parameters here, nowhere else
# =============================================================================

@dataclass
class CFG:
    # --- reproducibility ---
    seed: int = 42

    # --- environment ---
    env_id: str = SHARED_CORE_ENV_ID

    # --- training duration ---
    total_steps: int = 100_000       # total env steps
    warmup_steps: int = 1_000        # steps before learning starts

    # --- replay buffer ---
    buffer_size: int = 20_000
    batch_size: int = 64

    # --- Q-network ---
    hidden_sizes: tuple = (256, 256) # hidden layer sizes
    lr: float = 5e-4

    # --- DQN updates ---
    gamma: float = 0.99              # discount factor
    target_update_freq: int = 500    # steps between target network syncs

    # --- exploration ---
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000    # linear decay over this many steps

    # --- logging & checkpointing ---
    log_freq: int = 500              # log every N steps
    eval_freq: int = 5_000           # run evaluation every N steps
    eval_episodes: int = 10          # episodes per mid-training eval
    checkpoint_dir: str = "checkpoints"

    # --- final evaluation ---
    final_eval_episodes: int = 50    # as required by the assignment

    # --- SB3 baseline ---
    sb3_total_steps: int = 100_000
    sb3_log_path: str = "logs/sb3"

    # --- derived paths (set automatically in __post_init__) ---
    train_log_path: str = field(init=False)
    ep_log_path: str = field(init=False)

    def __post_init__(self):
        self.train_log_path = f"logs/dqn_training_s{self.seed}.csv"
        self.ep_log_path    = f"logs/dqn_episodes_s{self.seed}.csv"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)


# =============================================================================
# LOGGING — one CSV per run, one row per log step
# =============================================================================

class MetricsLogger:
    """Writes metrics to a CSV file. Call .log() with any keyword arguments."""

    def __init__(self, path: str):
        self.path = path
        self._writer = None
        self._file = None

    def log(self, **kwargs):
        if self._file is None:
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(kwargs.keys()))
            self._writer.writeheader()
        self._writer.writerow(kwargs)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# Q-NETWORK
# =============================================================================

class QNetwork(nn.Module):
    """MLP that maps a flat observation to Q-values for each action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: tuple):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# EPSILON-GREEDY ACTION SELECTION
# =============================================================================

def select_action(state, q_net, eps, n_actions, device):
    """Epsilon-greedy: random with prob eps, greedy otherwise."""
    if random.random() < eps:
        return random.randrange(n_actions)
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return q_net(state_t).argmax(dim=1).item()


def get_epsilon(step: int, cfg: CFG) -> float:
    """Linear epsilon decay from eps_start to eps_end over eps_decay_steps."""
    frac = min(step / cfg.eps_decay_steps, 1.0)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


# =============================================================================
# SINGLE GRADIENT UPDATE
# =============================================================================

def train_step(q_net, target_net, optimizer, buffer, cfg, device):
    """Sample a mini-batch and do one Bellman update."""
    states, actions, rewards, next_states, dones = buffer.sample(cfg.batch_size)

    states_t      = torch.tensor(states,      device=device)
    actions_t     = torch.tensor(actions,     device=device)
    rewards_t     = torch.tensor(rewards,     device=device)
    next_states_t = torch.tensor(next_states, device=device)
    dones_t       = torch.tensor(dones,       device=device)

    # Current Q-values for the taken actions
    q_values = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Target: r + gamma * max_a' Q_target(s', a')  [zero if done]
    with torch.no_grad():
        next_q = target_net(next_states_t).max(dim=1).values
        targets = rewards_t + cfg.gamma * next_q * (1 - dones_t)

    loss = nn.functional.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_dqn(cfg: CFG, device=None):
    """
    Full DQN training loop.
    Returns the trained q_net and the path to the training log CSV.
    """

    # def make_env(render_mode="rgb_array"):
    #     env = gym.make("highway-v0", render_mode=render_mode)
    #     env.unwrapped.configure(SHARED_CORE_CONFIG)
    #     env.reset()
    #     env = Monitor(env) 
    #     return env

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env()
    obs, _ = env.reset(seed=cfg.seed)
    obs_flat = obs.flatten()
    obs_dim = obs_flat.shape[0]
    n_actions = env.action_space.n

    q_net     = QNetwork(obs_dim, n_actions, cfg.hidden_sizes).to(device)
    target_net = QNetwork(obs_dim, n_actions, cfg.hidden_sizes).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)
    buffer    = ReplayBuffer(cfg.buffer_size)
    logger        = MetricsLogger(cfg.train_log_path)   # step-level
    ep_logger     = MetricsLogger(cfg.ep_log_path)   # episode-level

    ep_reward   = 0.0
    ep_length   = 0
    ep_count    = 0
    recent_rewards  = deque(maxlen=20)
    recent_lengths  = deque(maxlen=20)
    recent_crashes  = deque(maxlen=20)  # 1 = crash, 0 = timeout

    obs, _ = env.reset(seed=cfg.seed)

    for step in tqdm(range(cfg.total_steps), desc="DQN training"):
        obs_flat = obs.flatten().astype(np.float32)
        eps = get_epsilon(step, cfg)
        action = select_action(obs_flat, q_net, eps, n_actions, device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_obs_flat = next_obs.flatten().astype(np.float32)

        buffer.push(obs_flat, action, reward, next_obs_flat, float(done))
        ep_reward += reward
        ep_length += 1
        obs = next_obs

        if done:
            crashed = int(terminated)   # terminated=crash, truncated=timeout
            ep_count += 1
            recent_rewards.append(ep_reward)
            recent_lengths.append(ep_length)
            recent_crashes.append(crashed)

            ep_logger.log(
                episode=ep_count,
                step=step,
                reward=round(ep_reward, 4),
                length=ep_length,
                crashed=crashed,
                epsilon=round(eps, 4),
            )

            ep_reward = 0.0
            ep_length = 0
            obs, _ = env.reset()

        # Learn
        loss = None
        if len(buffer) >= cfg.warmup_steps:
            loss = train_step(q_net, target_net, optimizer, buffer, cfg, device)

        # Sync target network
        if step % cfg.target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Log
        if step % cfg.log_freq == 0 and loss is not None:
            mean_r     = np.mean(recent_rewards) if recent_rewards else 0.0
            mean_len   = np.mean(recent_lengths) if recent_lengths else 0.0
            crash_rate = np.mean(recent_crashes) if recent_crashes else 0.0
            logger.log(
                step=step,
                epsilon=round(eps, 4),
                loss=round(loss, 5),
                mean_reward_20ep=round(mean_r, 3),
                mean_length_20ep=round(mean_len, 1),
                crash_rate_20ep=round(crash_rate, 3),
                episodes=ep_count,
                buffer_size=len(buffer),
            )

        # Mid-training evaluation
        if step % cfg.eval_freq == 0 and step > 0:
            mean_r, std_r = evaluate(q_net, cfg, device, n_episodes=cfg.eval_episodes)
            print(f"\n[Step {step}] Eval ({cfg.eval_episodes} eps): "
                  f"mean={mean_r:.3f} ± {std_r:.3f}")

        # Checkpoint
        if step % (cfg.total_steps // 5) == 0 and step > 0:
            path = os.path.join(cfg.checkpoint_dir, f"dqn_step{step}.pt")
            torch.save(q_net.state_dict(), path)

    logger.close()
    ep_logger.close()
    env.close()

    # Save final model
    final_path = os.path.join(cfg.checkpoint_dir, "dqn_final.pt")
    torch.save(q_net.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")

    return q_net, cfg.train_log_path


# =============================================================================
# EVALUATION — deterministic, over N episodes
# =============================================================================

def evaluate(q_net, cfg: CFG, device, n_episodes: int = None, seeds=None, log_path: str = None):
    """
    Run n_episodes with epsilon=0 (greedy).
    Returns (mean_reward, std_reward).
    If log_path is provided, saves per-episode details to a CSV.
    """
    if n_episodes is None:
        n_episodes = cfg.final_eval_episodes

    q_net.eval()
    env = make_env()
    logger = MetricsLogger(log_path) if log_path else None
    rewards = []

    for i in range(n_episodes):
        seed = seeds[i] if seeds is not None else cfg.seed + i
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        ep_length = 0
        done = False
        while not done:
            obs_flat = obs.flatten().astype(np.float32)
            action = select_action(obs_flat, q_net, eps=0.0,
                                   n_actions=env.action_space.n, device=device)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            ep_length += 1
            done = terminated or truncated

        crashed = int(terminated)
        rewards.append(total_reward)

        if logger:
            logger.log(
                episode=i,
                seed=seed,
                reward=round(total_reward, 4),
                length=ep_length,
                crashed=crashed,
            )

    if logger:
        logger.close()
    env.close()
    q_net.train()
    return float(np.mean(rewards)), float(np.std(rewards))


# =============================================================================
# ROLLOUT RECORDING — saves a GIF
# =============================================================================

def record_rollout(q_net, cfg: CFG, device, path: str = "rollout.gif", seed=None):
    """
    Run one greedy episode, save frames as a GIF.
    Returns a dict with reward, length, and whether it crashed.
    """
    if seed is None:
        seed = cfg.seed

    env = make_env(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    ep_length = 0
    done = False

    q_net.eval()
    while not done:
        frame = env.render()
        frames.append(frame)
        obs_flat = obs.flatten().astype(np.float32)
        action = select_action(obs_flat, q_net, eps=0.0,
                               n_actions=env.action_space.n, device=device)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        ep_length += 1
        done = terminated or truncated

    env.close()
    q_net.train()

    imageio.mimsave(path, frames, fps=15)

    result = dict(reward=round(total_reward, 4), length=ep_length, crashed=int(terminated))
    print(f"Rollout saved to {path} — reward={result['reward']}, "
          f"length={result['length']}, crashed={result['crashed']}")
    return result


# =============================================================================
# STABLE-BASELINES3 BASELINE
# =============================================================================

def train_sb3(cfg: CFG):
    """
    Train a DQN via Stable-Baselines3 on the same benchmark.
    Returns the trained model and evaluation results.
    """
    env = make_env()

    model = SB3_DQN(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=cfg.lr,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        exploration_fraction=cfg.eps_decay_steps / cfg.sb3_total_steps,
        exploration_final_eps=cfg.eps_end,
        target_update_interval=cfg.target_update_freq,
        tensorboard_log=cfg.sb3_log_path,
        seed=cfg.seed,
    )

    print("Training SB3 DQN...")
    model.learn(total_timesteps=cfg.sb3_total_steps, progress_bar=True)
    model.save(os.path.join(cfg.checkpoint_dir, "sb3_dqn_final"))

    # Evaluate (same seeds as the custom DQN for fair comparison)
    eval_env = make_env()
    eval_seeds = [cfg.seed + i for i in range(cfg.final_eval_episodes)]
    rewards = []
    for seed in eval_seeds:
        obs, _ = eval_env.reset(seed=seed)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    eval_env.close()

    mean_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
    print(f"SB3 Eval ({cfg.final_eval_episodes} eps): mean={mean_r:.3f} ± {std_r:.3f}")

    # Save SB3 eval results to CSV
    os.makedirs("logs", exist_ok=True)
    with open("logs/sb3_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            w.writerow([i, r])

    return model, mean_r, std_r


# =============================================================================
# PLOTTING — training curves + comparison
# =============================================================================

def plot_training_curve(log_csvs: dict, title: str = "DQN Training", save_path: str = None):
    """
    Plot mean reward and loss from one or several training log CSVs.
    log_csvs: dict mapping label -> csv path, e.g. {"seed 0": "logs/dqn_training_s0.csv"}
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for label, path in log_csvs.items():
        df = pd.read_csv(path)
        ax1.plot(df["step"], df["mean_reward_20ep"], label=label)
        ax2.plot(df["step"], df["loss"], alpha=0.7, label=label)

    ax1.set_ylabel("Mean reward (20 ep window)")
    ax1.legend()
    ax1.set_title(title)

    ax2.set_ylabel("TD loss")
    ax2.set_xlabel("Training step")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_comparison(dqn_rewards: list, sb3_rewards: list, save_path: str = None):
    """Box + scatter plot comparing custom DQN vs SB3 over eval episodes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [dqn_rewards, sb3_rewards]
    ax.boxplot(data, labels=["Custom DQN", "SB3 DQN"], patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.4))
    for i, d in enumerate(data, start=1):
        ax.scatter([i] * len(d), d, alpha=0.5, zorder=3, s=20)

    ax.set_ylabel("Episode reward")
    ax.set_title(f"Evaluation over {len(dqn_rewards)} episodes")
    ax.axhline(np.mean(dqn_rewards), color="steelblue", linestyle="--",
               linewidth=1, label=f"DQN mean: {np.mean(dqn_rewards):.2f}")
    ax.axhline(np.mean(sb3_rewards), color="orange", linestyle="--",
               linewidth=1, label=f"SB3 mean: {np.mean(sb3_rewards):.2f}")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# =============================================================================
# MAIN — wire everything together
# =============================================================================

if __name__ == "__main__":
    cfg = CFG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("logs", exist_ok=True)

    # --- Train custom DQN ---
    q_net, train_log = train_dqn(cfg, device)

    # --- Final evaluation (50 runs, fixed seeds) ---
    eval_seeds = [cfg.seed + i for i in range(cfg.final_eval_episodes)]
    dqn_mean, dqn_std = evaluate(q_net, cfg, device,
                                  n_episodes=cfg.final_eval_episodes,
                                  seeds=eval_seeds)
    print(f"\nCustom DQN — {cfg.final_eval_episodes} ep: "
          f"mean={dqn_mean:.3f} ± {dqn_std:.3f}")

    # --- Record a rollout ---
    record_rollout(q_net, cfg, device, path="rollout_dqn.gif")

    # --- SB3 baseline ---
    sb3_model, sb3_mean, sb3_std = train_sb3(cfg)
    print(f"SB3 DQN     — {cfg.final_eval_episodes} ep: "
          f"mean={sb3_mean:.3f} ± {sb3_std:.3f}")

    # --- Plots ---
    plot_training_curve(train_log, save_path="training_curve.png")

    # Collect SB3 rewards for the comparison plot
    sb3_rewards = pd.read_csv("logs/sb3_eval.csv")["reward"].tolist()
    dqn_rewards_list = []
    eval_env = make_env()
    for seed in eval_seeds:
        obs, _ = eval_env.reset(seed=seed)
        total_r = 0.0
        done = False
        while not done:
            obs_flat = obs.flatten().astype(np.float32)
            action = select_action(obs_flat, q_net, eps=0.0,
                                   n_actions=eval_env.action_space.n, device=device)
            obs, r, terminated, truncated, _ = eval_env.step(action)
            total_r += r
            done = terminated or truncated
        dqn_rewards_list.append(total_r)
    eval_env.close()

    plot_comparison(dqn_rewards_list, sb3_rewards, save_path="comparison.png")

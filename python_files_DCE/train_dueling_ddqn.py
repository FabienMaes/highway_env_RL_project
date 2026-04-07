# =============================================================================
# train_dueling_ddqn.py
#
# This script is a direct upgrade of the basic DDQN:
# it makes EXACTLY ONE change — swapping the standard network for a
# Dueling Network architecture — and keeps everything else identical
# (same replay buffer, same training loop, same hyperparameters, same epsilon
# schedule, same evaluation protocol).
#
# WHY THIS SINGLE CHANGE:
#   In highway-v0, many timesteps are "action-independent": when the road
#   ahead is clear, all five actions lead to roughly the same outcome.
#   A standard network must evaluate every action separately to update its
#   Q-values. The Dueling Network decomposes Q(s,a) into:
#       V(s)    — how good is this state, regardless of action?
#       A(s,a)  — how much better is action a than the average?
#   This means V(s) is updated on EVERY timestep, not just for the chosen
#   action. The network learns the value of states far faster, which directly
#   accelerates learning the difference between safe and unsafe situations.
#   Reference: Wang et al. (2016) — https://arxiv.org/abs/1511.06581
#
# WHAT IS UNCHANGED vs. basic DDQN:
#   - Uniform replay buffer (same capacity, same sampling)
#   - Batch size, learning rate, gamma, all identical
#   - Hard target network copy every 50 steps
#   - Epsilon schedule (exponential decay, once per episode)
#   - MSE loss, Adam optimizer
#   - Train loop, evaluation loop, seeds
#
# HOW TO RUN (SLURM example):
#   sbatch --time=04:00:00 --mem=8G --wrap="python train_dueling_ddqn.py"
#
# All outputs are saved under ./results/<timestamp>/ and never overwritten.
# =============================================================================

import matplotlib
matplotlib.use("Agg")   # must be before any other matplotlib import on headless servers
import matplotlib.pyplot as plt

import os
import csv
import time
import random
import datetime

import gymnasium as gym
import highway_env                  # registers highway-v0
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
from tqdm import tqdm
from stable_baselines3.common.monitor import Monitor

from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG

# =============================================================================
# Configuration — edit these to change the run
# =============================================================================

N_EPISODES              = 2000
DECREASE_EPSILON_FACTOR = 8000   # scales with N_EPISODES
EVAL_SEEDS              = [3, 333, 3333, 33333, 333333]
EVAL_EPISODES_PER_SEED  = 50
ROLLOUT_SEED            = 3

# =============================================================================
# Output directory — timestamped so repeated runs never overwrite each other
# =============================================================================

TIMESTAMP   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("results", f"dueling_ddqn_{TIMESTAMP}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# Environment
# =============================================================================

def make_env(render_mode=None):
    """
    render_mode=None  during training  (no frame rendering — much faster).
    render_mode='rgb_array'  only when recording the rollout GIF.
    The spurious pre-Monitor reset is intentionally omitted.
    """
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()
    env = Monitor(env)
    return env

# =============================================================================
# Replay Buffer — identical to the basic DDQN
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory   = []          # stores the transitions
        self.position = 0           # tells us where to insert the next transition

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # Here we overwrite the oldest memory (transition) with the new one
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Pick a random batch of transitions, that will be used to train the model."""
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

# =============================================================================
# THE ONE CHANGE: Dueling Network instead of the standard Net
# =============================================================================

class DuelingNet(nn.Module):
    """
    Replaces the standard Net used in the basic DDQN.

    Architecture:
      Input (flattened observation)
          │
      Shared feature layer  [Linear → ReLU]
          │
      ┌───┴────────────┐
      │                │
    Value stream     Advantage stream
    [Linear → ReLU  [Linear → ReLU
     → Linear(1)]    → Linear(n_actions)]
      │                │
      └──── Q(s,a) = V(s) + A(s,a) − mean_a(A(s,a)) ────┘

    The mean subtraction makes the decomposition identifiable:
    without it, V and A can absorb arbitrary constants from each other.

    Hidden size is kept at 128 to match the standard Net exactly,
    so any performance difference is purely architectural.
    """
    def __init__(self, obs_shape, hidden_size, n_actions):
        super().__init__()
        flattened_size = int(np.prod(obs_shape))

        # Shared feature extraction — same as the first layer of the standard Net
        self.feature_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
        )
        # Value stream: one scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # Advantage stream: one value per action A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        features   = self.feature_layer(x)                          # (batch, hidden)
        value      = self.value_stream(features)                    # (batch, 1)
        advantages = self.advantage_stream(features)                # (batch, n_actions)
        # Subtract mean advantage so V and A are uniquely determined
        return value + advantages - advantages.mean(dim=1, keepdim=True)

# =============================================================================
# Dueling DDQN Agent
# All code below is IDENTICAL to the basic DDQN agent, except:
#   · DuelingNet replaces Net in __init__
#   · Everything else — hyperparameters, update rule, epsilon logic — unchanged
# =============================================================================

class DuelingDDQNAgent:
    def __init__(self, action_space, observation_space, decrease_epsilon_factor=2000):
        self.action_space      = action_space
        self.observation_space = observation_space

        # Hyperparameters — unchanged from basic DDQN
        self.learning_rate    = 5e-4
        self.gamma            = 0.99    # Discount factor
        self.batch_size       = 32
        self.buffer_capacity  = 15000   # Memory size
        self.update_target_every = 50
        # Exploration parameters
        self.epsilon_start    = 0.9
        self.epsilon_min      = 0.05
        self.decrease_epsilon_factor = decrease_epsilon_factor
        self.epsilon          = self.epsilon_start

        # Initialization
        self.buffer   = ReplayBuffer(self.buffer_capacity)
        obs_shape     = self.observation_space.shape
        n_actions     = self.action_space.n
        hidden_size   = 128             # same as standard Net

        # THE ONE CHANGE: DuelingNet instead of Net
        self.q_net      = DuelingNet(obs_shape, hidden_size, n_actions)
        self.target_net = DuelingNet(obs_shape, hidden_size, n_actions)
        # We ensure that they have the same weights at the beginning
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer     = optim.Adam(params=self.q_net.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

        self.n_steps = 0    # Total steps
        self.n_eps   = 0    # Total episodes

    def get_q(self, state):
        """Get the Q values of the given state by using the network."""
        # Convert numpy state to PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.q_net.forward(state_tensor)
        return output.numpy()[0]

    def get_action(self, state):
        """Chooses an action using the epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = self.get_q(state)
            return np.argmax(q_values)

    def decrease_epsilon(self):
        """Gradually reduces the chance of taking random actions over time."""
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            np.exp(-1. * self.n_eps / self.decrease_epsilon_factor))

    def update(self, state, action, reward, terminated, truncated, next_state):
        """Stores the transition and performs one step of gradient descent."""

        # Store the transition in the Replay Buffer
        # Convert the inputs to tensors
        self.buffer.push(
            torch.tensor(state,      dtype=torch.float32).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward],   dtype=torch.float32),
            torch.tensor([terminated], dtype=torch.int64),
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
        )
        # Check that we have enough data to train
        if len(self.buffer) < self.batch_size:
            return np.inf

        # Sample a batch
        transitions = self.buffer.sample(self.batch_size)

        # Transpose the batch
        state_batch, action_batch, reward_batch, terminated_batch, next_state_batch = tuple(
            [torch.cat(data) for data in zip(*transitions)]
        )

        # Compute current Q-values
        q_values = self.q_net(state_batch).gather(1, action_batch)

        # Compute target Q-values (Double DQN rule)
        with torch.no_grad():
            # Here lies the difference with the "simple" DQN: first, we find the actions
            # that maximize the Q-values in the next state in the main network; then,
            # we update using the Q-values in the next state for these actions in the target network
            next_state_indices_main   = self.q_net(next_state_batch).argmax(dim=1, keepdim=True)
            next_state_values_target  = self.target_net(next_state_batch).gather(
                1, next_state_indices_main).squeeze(1)
            # If the state was terminal, there is no future reward
            next_state_values_target  = next_state_values_target * (1 - terminated_batch)
            # Compute the expected Q values
            targets = reward_batch + (self.gamma * next_state_values_target)

        # Compute loss and backpropagate
        loss = self.loss_function(q_values, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network if necessary
        if self.n_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.n_steps += 1
        # decrease_epsilon is called only once per episode (not every step)
        if terminated or truncated:
            self.n_eps += 1
            self.decrease_epsilon()

        return loss.item()

# =============================================================================
# Training loop — identical to the notebook
# =============================================================================

def train_agent(agent, env, n_episodes=500):
    """
    Trains the agent and returns the trained agent and its training metrics.
    episode_losses is always the same length as episode_rewards:
    np.nan is used for early episodes where the buffer is not yet full,
    so the loss curve and reward curve always share the same x-axis.
    """
    episode_rewards = []
    episode_losses  = []
    episode_lengths = []
    epsilon_history = []

    for episode in tqdm(range(n_episodes), desc="Training"):
        state, _ = env.reset()
        done         = False
        total_reward = 0
        ep_losses    = []
        step_count   = 0

        while not done:
            # Choose the action (random with probability epsilon)
            action = agent.get_action(state)
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Update parameters by gradient descent and store transitions in the buffer
            loss = agent.update(state, action, reward, terminated, truncated, next_state)
            state = next_state
            total_reward += reward
            step_count   += 1
            if loss != np.inf:
                ep_losses.append(loss)

        # Store the total reward and losses for the learning curves
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        epsilon_history.append(agent.epsilon)
        # Always append a loss value to keep indices aligned with episode_rewards.
        # Use np.nan for early episodes where the buffer is not yet full enough to train.
        episode_losses.append(np.mean(ep_losses) if ep_losses else np.nan)

    return agent, episode_rewards, episode_losses, episode_lengths, epsilon_history

# =============================================================================
# Evaluation — identical to the notebook
# =============================================================================

def evaluate_agent(agent, num_episodes=50, seeds=None):
    """
    Evaluates an agent on multiple seeds.
    Returns a list of (seed, mean_reward, std_reward, episode_rewards) tuples.
    Raw episode_rewards are returned so the caller can compute a correctly
    pooled std across all individual episodes (not just a std-of-means).
    """
    if seeds is None:
        seeds = [3, 333, 3333]
    print(f"  Starting evaluation: {num_episodes} episodes x {len(seeds)} seeds.")
    all_seed_results = []

    for seed in seeds:
        env = make_env()
        episode_rewards = []
        for episode in tqdm(range(num_episodes), desc=f"  Seed {seed}"):
            obs, info = env.reset(seed=seed + episode)
            done         = False
            total_reward = 0
            while not done:
                q_values = agent.get_q(obs)
                action   = np.argmax(q_values)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            episode_rewards.append(total_reward)
        env.close()

        mean_reward = np.mean(episode_rewards)
        std_reward  = np.std(episode_rewards)
        print(f"  Seed {seed:>7d} | Mean: {mean_reward:+.4f}  Std: {std_reward:.4f}")
        all_seed_results.append((seed, mean_reward, std_reward, episode_rewards))

    return all_seed_results

# =============================================================================
# Plotting — saves PNG, never opens a window
# =============================================================================

def save_training_curves(episode_rewards, episode_losses, episode_lengths,
                          epsilon_history, out_path):
    """Saves the four training-curve subplots as a PNG file."""
    fig, axes = plt.subplots(1, 4, figsize=(28, 5))

    # Reward curve
    ax = axes[0]
    ax.plot(episode_rewards, alpha=0.3, color='blue', label='Raw reward')
    if len(episode_rewards) >= 20:
        rolling_mean = np.convolve(episode_rewards, np.ones(20) / 20, mode='valid')
        rolling_std  = np.array([
            np.std(episode_rewards[i:i + 20]) for i in range(len(episode_rewards) - 19)
        ])
        x = range(19, len(episode_rewards))
        ax.plot(x, rolling_mean, color='blue', linewidth=2, label='20-ep Moving Avg')
        ax.fill_between(x, rolling_mean - rolling_std, rolling_mean + rolling_std,
                        alpha=0.2, color='blue', label='±1 std')
    ax.set_title("Training curve: Episode rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend()

    # Loss curve
    ax = axes[1]
    ax.plot(episode_losses, color='orange', alpha=0.8)
    ax.set_title("Training curve: Mean loss per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE loss")

    # Steps per episode
    ax = axes[2]
    ax.plot(episode_lengths, alpha=0.3, color='green', label='Steps')
    if len(episode_lengths) >= 20:
        rolling_len = np.convolve(episode_lengths, np.ones(20) / 20, mode='valid')
        ax.plot(range(19, len(episode_lengths)), rolling_len,
                color='green', linewidth=2, label='20-ep Moving Avg')
    ax.set_title("Steps per episode (survival time)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()

    # Epsilon decay
    ax = axes[3]
    ax.plot(epsilon_history, color='red', linewidth=1.5)
    ax.set_title("Epsilon (exploration rate)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Training curves saved → {out_path}")

# =============================================================================
# Rollout recorder — saves GIF, no display
# =============================================================================

def record_rollout(agent, seed, out_path):
    """Runs one greedy episode and saves it as an animated GIF."""
    env_record = make_env(render_mode="rgb_array")
    obs, info  = env_record.reset(seed=seed)
    frames     = []
    done       = False
    step_count = 0

    while not done:
        frame = env_record.render()
        frames.append(frame)
        action = int(np.argmax(agent.get_q(obs)))
        obs, reward, terminated, truncated, info = env_record.step(action)
        done = terminated or truncated
        step_count += 1
    env_record.close()

    imageio.mimsave(out_path, frames, fps=15)
    print(f"  Rollout GIF saved → {out_path}  ({step_count} steps)")

# =============================================================================
# Results table — stdout + CSV
# =============================================================================

def save_results_table(seed_results, csv_path):
    """
    Writes per-seed results to CSV and prints a summary table to stdout.
    grand_std is computed over ALL individual episode rewards pooled across
    all seeds — this is the true episode-level std, not std-of-means.
    """
    seeds = [s for s, *_ in seed_results]
    fieldnames = (
        ["Model"]
        + [f"Seed {s}" for s in seeds]
        + ["Grand Mean", "Grand Std (pooled)"]
    )

    row     = {"Model": "DuelingDDQN"}
    all_raw = []
    for seed, mean_r, std_r, raw_rewards in seed_results:
        row[f"Seed {seed}"] = f"{mean_r:.4f} ± {std_r:.4f}"
        all_raw.extend(raw_rewards)

    grand_mean = np.mean(all_raw)
    grand_std  = np.std(all_raw)
    row["Grand Mean"]         = f"{grand_mean:.4f}"
    row["Grand Std (pooled)"] = f"{grand_std:.4f}"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    # Pretty-print
    col_w = 28
    header = " | ".join(f"{h:<{col_w}}" for h in fieldnames)
    sep    = "=" * len(header)
    print(f"\n{sep}")
    print("  Evaluation Results")
    print(sep)
    print(header)
    print("-" * len(header))
    print(" | ".join(f"{row.get(h, ''):<{col_w}}" for h in fieldnames))
    print(sep)
    print(f"\n  Results CSV saved → {csv_path}")

# =============================================================================
# Main
# =============================================================================

def main():
    total_start = time.time()

    print("\n" + "=" * 60)
    print("  Dueling DDQN — highway-v0")
    print("  ONE change vs. basic DDQN: DuelingNet architecture")
    print("=" * 60)
    print(f"  N_EPISODES              : {N_EPISODES}")
    print(f"  DECREASE_EPSILON_FACTOR : {DECREASE_EPSILON_FACTOR}")
    print(f"  Eval seeds              : {EVAL_SEEDS}")
    print(f"  Eval episodes/seed      : {EVAL_EPISODES_PER_SEED}")
    print(f"  Results dir             : {RESULTS_DIR}/")
    print("=" * 60 + "\n")

    # ── 1. Train ─────────────────────────────────────────────────────────────
    print("[1/4] Training ...")
    env   = make_env()
    agent = DuelingDDQNAgent(
        env.action_space,
        env.observation_space,
        decrease_epsilon_factor=DECREASE_EPSILON_FACTOR,
    )
    t0 = time.time()
    trained_agent, rewards, losses, lengths, epsilons = train_agent(
        agent, env, n_episodes=N_EPISODES
    )
    env.close()
    t_train = time.time() - t0
    print(f"  Done in {t_train:.1f}s ({t_train / N_EPISODES:.2f}s/episode)\n")

    # ── 2. Save model + metrics ───────────────────────────────────────────────
    print("[2/4] Saving model and metrics ...")
    model_path = os.path.join(RESULTS_DIR, "dueling_ddqn.pth")
    torch.save(trained_agent.q_net.state_dict(), model_path)
    np.save(os.path.join(RESULTS_DIR, "rewards.npy"),  rewards)
    np.save(os.path.join(RESULTS_DIR, "losses.npy"),   losses)
    np.save(os.path.join(RESULTS_DIR, "lengths.npy"),  lengths)
    np.save(os.path.join(RESULTS_DIR, "epsilons.npy"), epsilons)
    print(f"  Model   → {model_path}")
    print(f"  Metrics → {RESULTS_DIR}/{{rewards,losses,lengths,epsilons}}.npy")

    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    save_training_curves(rewards, losses, lengths, epsilons, curves_path)

    # ── 3. Rollout GIF ────────────────────────────────────────────────────────
    print(f"\n[3/4] Recording rollout (seed={ROLLOUT_SEED}) ...")
    gif_path = os.path.join(RESULTS_DIR, "rollout.gif")
    record_rollout(trained_agent, seed=ROLLOUT_SEED, out_path=gif_path)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Evaluating ...")
    t0 = time.time()
    seed_results = evaluate_agent(
        trained_agent,
        num_episodes=EVAL_EPISODES_PER_SEED,
        seeds=EVAL_SEEDS,
    )
    t_eval = time.time() - t0
    print(f"  Done in {t_eval:.1f}s")

    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    save_results_table(seed_results, csv_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  Total runtime: {total:.1f}s ({total / 60:.1f} min)")
    print(f"  All outputs:   {RESULTS_DIR}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

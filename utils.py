import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
import highway_env
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

import json
from datetime import datetime


def make_env(render_mode=None):
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()
    env = Monitor(env)
    return env

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
    
def warmup_buffer(agent, env, n_steps=1000):
    """Fill the replay buffer with random transitions before training."""
    state, _ = env.reset()
    
    for _ in range(n_steps):
        action = env.action_space.sample()  # purement aléatoire
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.buffer.push(
            torch.tensor(state, dtype=torch.float32, device=agent.device).flatten().unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64, device=agent.device),
            torch.tensor([reward], dtype=torch.float32, device=agent.device),
            torch.tensor([terminated], dtype=torch.float32, device=agent.device),
            torch.tensor(next_state, dtype=torch.float32, device=agent.device).flatten().unsqueeze(0),
        )
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()
    
    print(f"Buffer warmed up with {len(agent.buffer)} transitions.")
    
def train_agent(agent, env, n_episodes=500, prefix='agent'):
    episode_rewards = []
    episode_losses = []
    episode_lengths = []
    epsilon_history = []

    for episode in tqdm(range(n_episodes), desc="Training"):

        state, _ = env.reset()
        done = False
        total_reward = 0
        ep_losses = []
        step_count = 0
        while not done:

            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            loss = agent.update(state, action, reward, terminated, next_state)
            state = next_state
            done = terminated or truncated
            total_reward += reward
            if np.isfinite(loss):
                ep_losses.append(loss)
            step_count += 1

        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(ep_losses) if ep_losses else 0)
        episode_lengths.append(step_count)
        epsilon_history.append(agent.epsilon)

        if episode % 20 == 0:
            print(f"Ep {episode} | Reward {total_reward:.1f} | Eps {agent.epsilon:.3f}")
        
        if episode % 100 == 0:
            agent.save(f"models/{prefix}_ep{episode}.pt")
    
    return agent, episode_rewards, episode_losses, episode_lengths, epsilon_history

def evaluate_agent_sb3(agent, env, seed, n_episodes=50):
    rewards = []
    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0
        while not done:
            action, _ = agent.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

def evaluate_agent_scratch(agent, env, seed, n_episodes=50):
    rewards = []
    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

def evaluate_agent(eval_fn, agent, env_fn, seeds):
    means = []
    stds = []

    for seed in seeds:
        env = env_fn()

        mean, std = eval_fn(agent, env, n_episodes=50, seed=seed)

        means.append(mean)
        stds.append(std)

    return {
        "mean_of_means": np.mean(means),
        "std_of_means": np.std(means),
        "means": means,
        "stds": stds
    }


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        alpha : contrôle le degré de prioritisation
                0 = uniforme (comme DQN classique)
                1 = priorité totale
        """
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        # Nouvelle transition : priorité maximale existante (ou 1.0 au début)
        max_priority = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        beta : contrôle la correction IS
               0 = pas de correction
               1 = correction totale
        """
        priorities = self.priorities[:len(self.memory)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        # Calcul des IS weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalisation
        weights = torch.tensor(weights, dtype=torch.float32)

        return samples, indices, weights

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + epsilon

    def __len__(self):
        return len(self.memory)
    
def save_results(results, agent_name, path="./models"):
    """Save evaluation results to a JSON file."""
    output = {
        "agent": agent_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mean_of_means": results["mean_of_means"],
        "std_of_means": results["std_of_means"],
        "per_seed": [
            {"mean": float(m), "std": float(s)}
            for m, s in zip(results["means"], results["stds"])
        ]
    }
    filepath = f"{path}/eval_{agent_name}.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Results saved to {filepath}")
"""This code uses the same functions defined in the notebook for SB3, DQN and DDQN training on the DCE."""

import os
import argparse
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import highway_env
from torch import Tensor
from tqdm import tqdm
from stable_baselines3 import DQN as SB3_DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

# Import provided config
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


# MLP
class Net(nn.Module):
    def __init__(self, obs_shape, hidden_size, n_actions):
        super(Net, self).__init__()
        in_features = int(np.prod(obs_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# (D)DQN definition
class HighwayDQN:
    def __init__(self, env, gamma=0.99, batch_size=64, buffer_capacity=15000, 
                 update_target_every=50, epsilon_start=1.0, decrease_epsilon_factor=6000, 
                 epsilon_min=0.05, learning_rate=5e-4, model_type="DQN"):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every
        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = decrease_epsilon_factor
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.reset()

    def reset(self):
        hidden_size = 64
        obs_shape = self.observation_space.shape
        n_actions = int(self.action_space.n)
        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net = Net(obs_shape, hidden_size, n_actions).to(self.device)
        self.target_net = Net(obs_shape, hidden_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=self.learning_rate)
        self.epsilon = self.epsilon_start
        self.n_steps = 0

    def get_q(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.q_net(state_tensor)
        return output.cpu().numpy()[0]

    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
                        np.exp(-1. * self.n_steps / self.decrease_epsilon_factor) )

    def get_action(self, state, epsilon=None):
        if epsilon is None: epsilon = self.epsilon
        if np.random.rand() < epsilon:
            return self.action_space.sample()
        return int(np.argmax(self.get_q(state)))

    def update(self, state, action, reward, terminated, truncated, next_state):
        self.buffer.push(state, action, reward, terminated, next_state)
        if len(self.buffer) < self.batch_size:
            self.n_steps += 1
            return np.inf, np.inf  
        
        transitions = self.buffer.sample(self.batch_size)
        states, actions, rewards, terminateds, next_states = zip(*transitions)
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terminateds_t = torch.as_tensor(terminateds, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)

        values = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            if self.model_type == "DDQN": # Choosing next q for the DDQN
                next_acts = self.q_net(next_states_t).argmax(dim=1).unsqueeze(1)
                next_q = self.target_net(next_states_t).gather(1, next_acts).squeeze(1)
            else: # Choosing next q for the DQN
                next_q = self.target_net(next_states_t).max(1)[0]
            targets = rewards_t + self.gamma * next_q * (1 - terminateds_t)

        loss = self.loss_function(values, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not ((self.n_steps + 1) % self.update_target_every):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()
        self.n_steps += 1
        return loss.item(), values.mean().item()

    def save(self, filepath: str):
        """Saves the Q-network's weights to a file."""
        torch.save(self.q_net.state_dict(), filepath)

    def load(self, filepath: str):
        """Loads the Q-network's weights from a file."""
        self.q_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_config(self, filepath: str):
        """Save the Q-network's config to a file with NumPy type handling."""
        
        def json_serializable(obj):
            """Helper to convert numpy types to standard python types."""
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj) # Fallback to string for unknown objects

        config_dict = {
            "model_type": self.model_type,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "buffer_capacity": self.buffer_capacity,
            "update_target_every": self.update_target_every,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "decrease_epsilon_factor": self.decrease_epsilon_factor,
            "learning_rate": self.learning_rate,
            "env_config": SHARED_CORE_CONFIG 
        }

        print(f"Attempting to save config to {filepath}...")
        try:
            with open(filepath, "w") as f:
                # Save config to json file
                json.dump(config_dict, f, indent=4, default=json_serializable)
            print("Config saved successfully.")
        except Exception as e:
            print(f"Error saving JSON config: {e}")


# Function to train the agent (DQN or DDQN)
def train_agent(agent, env, seed, n_episodes=1500):
    history = {
        "episode": [], "reward": [], "loss": [], "length": [], 
        "epsilon": [], "average_speed": [], "mean_q": [], "true_return": []
    }
    checkpoint_dir = f"checkpoints_dqn_mod"
    os.makedirs(checkpoint_dir, exist_ok=True)
    for episode in tqdm(range(n_episodes), desc=f"Training agent {agent.model_type}"):
        state, _ = env.reset()
        done = False
        total_reward, step_count = 0, 0
        ep_losses, ep_speeds, ep_q_values, ep_rewards = [], [], [], []

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            loss, _ = agent.update(state, action, reward, terminated, truncated, next_state)
            if loss != np.inf:
                ep_losses.append(loss)

            # Store Q-value for bias analysis 
            with torch.no_grad():
                st_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                q_val = agent.q_net(st_tensor).max(1)[0].item()
            
            ep_q_values.append(q_val)
            ep_rewards.append(reward)
            ep_speeds.append(info.get("speed", 0))
            
            state = next_state
            total_reward += reward
            step_count += 1

        # Calculate True Discounted Return 
        true_ret = 0
        for r in reversed(ep_rewards):
            true_ret = r + agent.gamma * true_ret

        history["episode"].append(episode)
        history["reward"].append(total_reward)
        history["length"].append(step_count)
        history["epsilon"].append(agent.epsilon)
        history["loss"].append(np.mean(ep_losses) if ep_losses else 0.0)
        history["average_speed"].append(np.mean(ep_speeds))
        history["mean_q"].append(np.mean(ep_q_values) if ep_q_values else 0.0)
        history["true_return"].append(true_ret)

        # Save checkpoints every 500 episodes
        if (episode + 1) % 500 == 0:
            agent.save(f"{checkpoint_dir}/{agent.model_type}_seed_{seed}_ep{episode+1}.pth")

    return pd.DataFrame(history)

# SB3 Logging

class SB3MetricCallback(BaseCallback):
    def __init__(self, seed, verbose=0):
        super().__init__(verbose)
        self.seed = seed
        self.data = {
            "episode": [], 
            "reward": [], 
            "length": [], 
            "average_speed": []
        }
        self._temp_speeds = []
        self._current_reward = 0
        self._current_length = 0
        self._ep_count = 0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        self._current_length += 1
        
        info = self.locals["infos"][0]
        self._temp_speeds.append(info.get("speed", 0))

        if self.locals["dones"][0]:
            avg_speed = np.mean(self._temp_speeds) if self._temp_speeds else 0.0
            
            self.data["episode"].append(self._ep_count)
            self.data["reward"].append(float(self._current_reward))
            self.data["length"].append(int(self._current_length))
            self.data["average_speed"].append(float(avg_speed))
            
            # Logs every 10 episodes to the SLURM .out file
            if self._ep_count % 10 == 0:
                print(f"[SB3 Seed {self.seed}] Episode: {self._ep_count} | "
                      f"Steps: {self.num_timesteps} | "
                      f"Reward: {self._current_reward:.2f} | "
                      f"Avg Speed: {avg_speed:.2f}")

            self._current_reward = 0
            self._current_length = 0
            self._temp_speeds = []
            self._ep_count += 1
            
        return True

# SB3 run
def run_sb3_task(seed, steps = 40000):
    env = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG)
    save_path = f"gamma_results_sb3"
    os.makedirs(save_path, exist_ok=True)

    model = SB3_DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=64,
        gamma=0.9,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        exploration_fraction=0.5,
        verbose=1,
        seed=seed
    )
    

    # Metric logging callback
    metric_callback = SB3MetricCallback(seed=seed)
    
    # Checkpoint callback (saves every 5,000 steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path=f"{save_path}",
        name_prefix=f"sb3_model_seed_{seed}"
    )

    callback_list = CallbackList([metric_callback, checkpoint_callback])
    model.learn(total_timesteps=steps, callback=callback_list)
    
    model.save(f"{save_path}/sb3_seed_{seed}_final_model")
    return pd.DataFrame(metric_callback.data)

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["dqn", "sb3", "ddqn"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    os.makedirs(f"results_{args.agent}", exist_ok=True)
    env = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG)
    env.reset(seed=args.seed)
    
    if args.agent == "dqn":
        agent = HighwayDQN(env, decrease_epsilon_factor=6000, model_type = "DQN", gamma=0.9)
        agent.save_config(f"results_{args.agent}/config_seed_{args.seed}.json")
        df = train_agent(agent, env, n_episodes=1500, seed=args.seed)
        agent.save(f"results_{args.agent}/dqn_final_seed_{args.seed}.pth")
        df.to_csv(f"results_{args.agent}/metrics_seed_{args.seed}.csv", index=False)
    elif args.agent == "ddqn":
        agent = HighwayDQN(env, decrease_epsilon_factor=6000, model_type="DDQN", gamma=0.9)
        agent.save_config(f"results_ddqn/config_seed_{args.seed}.json")
        df = train_agent(agent, env, n_episodes=1500, seed=args.seed)
        agent.save(f"results_ddqn/ddqn_final_seed_{args.seed}.pth")
        df.to_csv(f"results_ddqn/metrics_seed_{args.seed}.csv", index=False)
    else:
        df = run_sb3_task(seed=args.seed, steps=35000)
        df.to_csv(f"results_sb3/metrics_seed_{args.seed}.csv", index=False)
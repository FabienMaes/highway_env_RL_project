import random
from typing import Optional

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, terminated: bool, next_state: np.ndarray) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], hidden_size: int, n_actions: int):
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


class HighwayDQN:
    def __init__(self,
                 action_space: gym.spaces.Discrete,
                 observation_space: gym.spaces.Box,
                 gamma: float,
                 batch_size: int,
                 buffer_capacity: int,
                 update_target_every: int,
                 epsilon_start: float,
                 decrease_epsilon_factor: float,
                 epsilon_min: float,
                 learning_rate: float):

        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = decrease_epsilon_factor
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"Initializing DQN on device: {self.device}")

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
        self.n_eps = 0

    def get_q(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.q_net(state_tensor)
        return output.cpu().numpy()[0]

    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
                        np.exp(-1. * self.n_steps / self.decrease_epsilon_factor) )

    def get_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:  # type: ignore
            return self.action_space.sample()

        return int(np.argmax(self.get_q(state)))

    def update(self, state: np.ndarray, action: int, reward: float, terminated: bool, next_state: np.ndarray) -> float:
        self.buffer.push(state, action, reward, terminated, next_state)
        if len(self.buffer) < self.batch_size:
            self.n_steps += 1
            return np.inf

        transitions = self.buffer.sample(self.batch_size)
        states, actions, rewards, terminateds, next_states = zip(*transitions)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terminateds_t = torch.as_tensor(terminateds, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)

        values = self.q_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            targets = rewards_t + self.gamma * self.target_net.forward(next_states_t).max(1)[0] * (1 - terminateds_t)

        loss = self.loss_function(values, targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not ((self.n_steps + 1) % self.update_target_every):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()
        self.n_steps += 1

        return loss.detach().item()

    def save(self, filepath: str):
        """Saves the Q-network's weights to a file."""
        torch.save(self.q_net.state_dict(), filepath)

    def load(self, filepath: str):
        """Loads the Q-network's weights from a file."""
        self.q_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())


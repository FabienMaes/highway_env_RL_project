import numpy as np
import torch
import torch.nn as nn
from utils import QNetwork, ReplayBuffer, PrioritizedReplayBuffer

class DQNAgent:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma = 0.99,
        batch_size = 32,
        buffer_capacity = 15000,
        update_target_every = 50,
        epsilon_start = 1,
        decrease_epsilon_factor = 1000,
        epsilon_min = 0.01,
        learning_rate = 1e-3,
        device = "cpu",
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = (
            decrease_epsilon_factor  # larger -> more exploration
        )
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate

        obs_size = int(np.prod(self.observation_space.shape))
        n_actions = self.action_space.n

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.device = device
        self.q_net = QNetwork(obs_dim = obs_size, n_actions = n_actions, hidden = 256).to(self.device)
        self.target_net = QNetwork(obs_dim=obs_size, n_actions=n_actions, hidden=256).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())


        self.loss_function = nn.HuberLoss()
        self.optimizer = torch.optim.Adam(params=self.q_net.parameters(), lr=self.learning_rate)

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0

    def get_q(self, state):
        """Get the Q values of the given state by using the network"""
        # Convert numpy state to PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        state_tensor = state_tensor.flatten().unsqueeze(0)
        with torch.no_grad():
            output = self.q_net.forward(state_tensor)
        return output.cpu().numpy()[0]

    def get_action(self, state):
        """
        Return action according to an epsilon-greedy exploration policy
        """

        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.get_q(state))

    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            np.exp(-1.0 * self.n_steps / self.decrease_epsilon_factor)
        )

    def update(self, state, action, reward, terminated, next_state):

        # add data to replay buffer
        self.buffer.push(torch.tensor(state, dtype=torch.float32, device=self.device).flatten().unsqueeze(0),
                        torch.tensor([[action]], dtype=torch.int64, device=self.device),
                        torch.tensor([reward], dtype=torch.float32, device=self.device),
                        torch.tensor([terminated], dtype=torch.float32, device=self.device),
                        torch.tensor(next_state, dtype=torch.float32, device=self.device).flatten().unsqueeze(0),
                        )

        if len(self.buffer) < self.batch_size:
            return np.inf

        # get batch
        transitions = self.buffer.sample(self.batch_size)

        state_batch, action_batch, reward_batch, terminated_batch, next_state_batch = tuple([torch.cat(data) for data in zip(*transitions)])
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        terminated_batch = terminated_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)

        values = self.q_net.forward(state_batch).gather(1, action_batch)

        # Compute the ideal Q values
        with torch.no_grad():
            next_state_values = (1 - terminated_batch) * self.target_net(
                next_state_batch
            ).max(1)[0]
            targets = next_state_values * self.gamma + reward_batch

        loss = self.loss_function(values, targets.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        if self.n_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()

        self.n_steps += 1
        if terminated:
            self.n_eps += 1

        return loss.item()
    
    def save(self, path):
        torch.save({"q_net": self.q_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                    }, path)
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]

class PERDQNAgent(DQNAgent):
    def __init__(
        self,
        action_space,
        observation_space,
        alpha=0.6,       # degré de prioritisation
        beta_start=0.4,  # IS correction initiale
        beta_frames=50000, # frames pour atteindre beta=1.0
        **kwargs
    ):
        super().__init__(action_space, observation_space, **kwargs)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        # Remplace le buffer classique par le buffer PER
        self.buffer = PrioritizedReplayBuffer(self.buffer_capacity, alpha=self.alpha)

    def _get_beta(self):
        """Beta augmente linéairement de beta_start à 1.0"""
        beta = self.beta_start + (1.0 - self.beta_start) * (self.n_steps / self.beta_frames)
        return min(1.0, beta)

    def update(self, state, action, reward, terminated, next_state):
        # Push identique au parent
        self.buffer.push(
            torch.tensor(state, dtype=torch.float32, device=self.device).flatten().unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64, device=self.device),
            torch.tensor([reward], dtype=torch.float32, device=self.device),
            torch.tensor([terminated], dtype=torch.float32, device=self.device),
            torch.tensor(next_state, dtype=torch.float32, device=self.device).flatten().unsqueeze(0),
        )

        if len(self.buffer) < self.batch_size:
            return np.inf

        beta = self._get_beta()
        transitions, indices, weights = self.buffer.sample(self.batch_size, beta=beta)
        weights = weights.to(self.device)

        state_batch, action_batch, reward_batch, terminated_batch, next_state_batch = (
            tuple([torch.cat(data) for data in zip(*transitions)])
        )
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        terminated_batch = terminated_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)

        values = self.q_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = (1 - terminated_batch) * self.target_net(
                next_state_batch
            ).max(1)[0]
            targets = (next_state_values * self.gamma + reward_batch).unsqueeze(1)

        # TD-errors pour mise à jour des priorités
        td_errors = (targets - values).detach().cpu().numpy().flatten()
        self.buffer.update_priorities(indices, td_errors)

        # Loss pondérée par les IS weights
        loss = (weights * nn.HuberLoss(reduction='none')(values, targets).squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        if self.n_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()
        self.n_steps += 1
        if terminated:
            self.n_eps += 1

        return loss.item()
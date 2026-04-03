import torch
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID

env = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG, render_mode="rgb_array")
env = Monitor(env)
env.reset()
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Initializing Stable-Baselines3 DQN model on device: {device}")
# model = DQN("MlpPolicy", env, verbose=1, learning_rate=5e-4, buffer_size=15000,
#             learning_starts=200, tensorboard_log="highway_dqn_stable/", device=device)

# print("Starting training...")
# # Train the model. 20000 is a placeholder, adjust based on convergence.
# model.learn(total_timesteps=20000, progress_bar=True)

# # Save the baseline model
# model.save("highway_dqn_stable/model")
model = DQN.load("highway_dqn/model", device=device)
print("Evaluating the model...")
# The assignment requires thorough evaluation (50 runs mean rewards + std)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print(f"Evaluation Results: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()

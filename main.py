from copy import deepcopy

import highway_env
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt


from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from DQN import HighwayDQN


def eval_agent(env: gym.Env, agent: HighwayDQN, n_sim=5):
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_sim)

    pbar = tqdm(range(n_sim), desc="Evaluating DQN")
    for i in pbar:
        done = False
        state, _ = env_copy.reset()
        while not done:
            action = agent.get_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env_copy.step(action)
            episode_rewards[i] += reward
            state = next_state
            done = terminated or truncated

    return episode_rewards


def train(env: gym.Env, agent: HighwayDQN, N_episodes: int, eval_every=10) -> list[float]:
    losses = []

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

        pbar.set_postfix({
            "Reward": f"{episode_reward:.2f}",
            "Epsilon": f"{agent.epsilon:.2f}",
        })

        if ((ep + 1) % eval_every == 0):
            rewards = eval_agent(env, agent)
            tqdm.write(f"Episode {ep + 1:04d} | Eval Mean Reward: {np.mean(rewards):.2f} | Epsilon: {agent.epsilon:.2f}")

    return losses


def visualize_episode(agent: HighwayDQN) -> None:
    """Runs one episode with rendering to visualize the agent's behavior."""
    env = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG, render_mode="human")

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Use epsilon=0.0 for pure exploitation (greedy policy)
        action = agent.get_action(state, epsilon=0.0)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward # type: ignore
        done = terminated or truncated
        env.render()

    print(f"Visualization Episode Reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    env = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG, render_mode='rgb_array')
    env.reset()

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

    # N_episodes = 200
    # losses = train(env, agent, N_episodes, eval_every=10)

    # save_path = "models/custom_dqn_highway.pth"
    # print(f"Saving the custom model to {save_path}...")
    # agent.save(save_path)

    agent.load("models/custom_dqn_highway.pth")
    final_rewards = eval_agent(env, agent, n_sim=50)
    print(f"Final Evaluation Results: Mean reward: {np.mean(final_rewards):.2f} +/- {np.std(final_rewards):.2f}")

    # plt.figure(figsize=(10, 5))
    # plt.plot(losses)
    # plt.title("DQN Training Loss")
    # plt.xlabel("Update Steps")
    # plt.ylabel("Loss (MSE)")
    # plt.savefig("figures/dqn_loss_plot.png")
    # plt.show()

    visualize_episode(agent)

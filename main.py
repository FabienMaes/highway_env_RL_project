import gymnasium as gym
import highway_env
from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
import numpy as np
from stable_baselines3 import DQN
from agents import DQNAgent, PERDQNAgent
from utils import make_env, train_agent, warmup_buffer, evaluate_agent, evaluate_agent_sb3, evaluate_agent_scratch, save_results
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def save_metrics(prefix, rewards, losses, lengths, epsilons):
    np.save(f"./models/{prefix}_rewards.npy", rewards)
    np.save(f"./models/{prefix}_losses.npy", losses)
    np.save(f"./models/{prefix}_lengths.npy", lengths)
    np.save(f"./models/{prefix}_epsilons.npy", epsilons)

def plot_metrics(prefix, rewards, losses, lengths, epsilons):
    for data, ylabel, title, fname in [
    (rewards,  "Total Reward",   "Training Progress", "rewards"),
    (lengths,  "Episode Length", "Episode Lengths",   "lengths"),
    (losses,   "Average Loss",   "Training Loss",     "losses"),
    (epsilons, "Epsilon",        "Epsilon Decay",     "epsilons"),
    ]:
        plt.figure()

        # 🔹 Cas rewards & lengths → avec moyenne glissante
        if fname in ["rewards", "lengths"]:
            # courbe brute (transparente)
            plt.plot(data, alpha=0.3, label="Raw")

            # moyenne glissante
            rolling_mean = np.convolve(data, np.ones(20)/20, mode='valid')
            rolling_std  = np.array([np.std(data[i:i+20]) for i in range(len(data)-19)])
            plt.plot(range(19, len(data)), rolling_mean, linewidth=2, label="Moving Avg (20)")
            plt.fill_between(range(19, len(data)), rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.2)

            plt.legend()

        else:
            # courbe normale
            plt.plot(data)

        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(f"{title} - {prefix}")
        plt.savefig(f"training_{fname}_{prefix}.png")
        plt.close()

def train_and_save(agent, env, prefix, n_episodes=500, warmup_steps=1000):
    print(f"Warming up replay buffer...")
    warmup_buffer(agent, env, n_steps=warmup_steps)
    print(f"==================== Training {prefix} ====================")
    trained_agent, rewards, losses, lengths, epsilons = train_agent(agent, env, n_episodes=n_episodes, prefix=prefix)
    trained_agent.save(f"models/{prefix}.pt")
    save_metrics(prefix, rewards, losses, lengths, epsilons)
    plot_metrics(prefix, rewards, losses, lengths, epsilons)
    print(f"{prefix} training completed and model saved.")
    return trained_agent

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # ── SB3 DQN ──────────────────────────────
    env_sb3 = make_env(render_mode=None)
    print("Initializing SB3 DQN...")
    sb3_dqn = DQN(
        policy='MlpPolicy',
        env=env_sb3,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=1e-3,
        buffer_size=15000,
        learning_starts=1500,
        batch_size=32,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        exploration_fraction=0.7,
        verbose=0,
    )
    print("==================== Training SB3 DQN ====================")
    sb3_dqn.learn(int(24000), progress_bar=True)
    sb3_dqn.save("models/sb3_dqn_highway")
    print("SB3 DQN training completed and model saved.")

    # ── Scratch DQN ──────────────────────────
    env_scratch = make_env(render_mode=None)
    scratch_dqn = DQNAgent(env_scratch.action_space, env_scratch.observation_space)
    trained_scratch = train_and_save(scratch_dqn, env_scratch, prefix="scratch_dqn_highway", n_episodes=800, warmup_steps=1500)

    # ── PER DQN ──────────────────────────────
    env_per =  make_env(render_mode=None)
    per_dqn = PERDQNAgent(env_per.action_space, env_per.observation_space)
    trained_per = train_and_save(per_dqn, env_per, prefix="per_dqn_highway", n_episodes=800, warmup_steps=1500)

    # ── Evaluation ───────────────────────────
    print("===================== Evaluating SB3 DQN ====================")
    results_sb3 = evaluate_agent(evaluate_agent_sb3, sb3_dqn, make_env, seeds=[21, 300, 5100])
    print(results_sb3)

    print("===================== Evaluating Scratch DQN ====================")
    trained_scratch.epsilon = 0.0
    results_scratch = evaluate_agent(evaluate_agent_scratch, trained_scratch, make_env, seeds=[21, 300, 5100])
    print(results_scratch)

    print("===================== Evaluating PER DQN ====================")
    trained_per.epsilon = 0.0
    results_per = evaluate_agent(evaluate_agent_scratch, trained_per, make_env, seeds=[21, 300, 5100])
    print(results_per)

    save_results(results_sb3, agent_name="sb3_dqn")
    save_results(results_scratch, agent_name="scratch_dqn")
    save_results(results_per, agent_name="per_dqn")


if __name__ == "__main__":
    main()
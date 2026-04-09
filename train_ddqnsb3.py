"""Train the dqn, the baseline SB3 and the ddqn on the same number of steps, same seed."""
import os
import argparse
import torch
from dqn_highway import CFG, train_dqn, evaluate, train_sb3

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--total_steps", type=int, default=60_000)
parser.add_argument("--sb3_steps", type=int, default=60_000)
args = parser.parse_args()

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DQN ---
cfg_dqn = CFG(seed=args.seed, total_steps=args.total_steps,
              sb3_total_steps=args.sb3_steps, use_ddqn=False)

q_net_dqn, _ = train_dqn(cfg_dqn, device=device)

eval_seeds = [args.seed + i for i in range(cfg_dqn.final_eval_episodes)]
mean_r, std_r = evaluate(q_net_dqn, cfg_dqn, device=device,
                          seeds=eval_seeds,
                          log_path=f"logs/dqn_final_eval_s{args.seed}.csv")
print(f"DQN — mean={mean_r:.3f} ± {std_r:.3f}")

# --- DDQN ---
cfg_ddqn = CFG(seed=args.seed, total_steps=args.total_steps,
               sb3_total_steps=args.sb3_steps, use_ddqn=True)

q_net_ddqn, _ = train_dqn(cfg_ddqn, device=device)

mean_r, std_r = evaluate(q_net_ddqn, cfg_ddqn, device=device,
                          seeds=eval_seeds,
                          log_path=f"logs/ddqn_final_eval_s{args.seed}.csv")
print(f"DDQN — mean={mean_r:.3f} ± {std_r:.3f}")

# --- SB3 ---
train_sb3(cfg_dqn)
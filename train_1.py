"""Train the dqn and the baseline."""
import os
import argparse
import torch
from dqn_highway import CFG, train_dqn, evaluate, train_sb3

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total_steps", type=int, default=60_000)
parser.add_argument("--sb3_steps", type=int, default=60_000)
args = parser.parse_args()

cfg = CFG(seed=args.seed, total_steps=args.total_steps, sb3_total_steps=args.sb3_steps)

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_net, log_path = train_dqn(cfg, device=device)

eval_seeds = [cfg.seed + i for i in range(cfg.final_eval_episodes)]
mean_r, std_r = evaluate(
    q_net, cfg,
    device=device,
    seeds=eval_seeds,
    log_path=f"logs/dqn_final_eval_s{cfg.seed}.csv"
)
print(f"DQN Final eval — mean={mean_r:.3f} ± {std_r:.3f}")

train_sb3(cfg)
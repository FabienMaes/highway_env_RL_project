"""Train only the baseline."""
import os
import torch
from dqn_highway import CFG, train_sb3

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

cfg = CFG(seed=16, total_steps=60_000, sb3_total_steps=40_000)
print("Running", flush=True)
sb3_model, sb3_mean, sb3_std = train_sb3(cfg)
print(f"SB3 eval — mean={sb3_mean:.3f} ± {sb3_std:.3f}")
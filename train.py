import argparse
from dqn_highway import CFG, train_dqn, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total_steps", type=int, default=100_000)
args = parser.parse_args()

cfg = CFG(seed=args.seed, total_steps=args.total_steps)

q_net, log_path = train_dqn(cfg)

eval_seeds = [cfg.seed + i for i in range(cfg.final_eval_episodes)]
mean_r, std_r = evaluate(
    q_net, cfg,
    device=None,
    seeds=eval_seeds,
    log_path=f"logs/dqn_final_eval_s{cfg.seed}.csv"
)
print(f"Final eval — mean={mean_r:.3f} ± {std_r:.3f}")
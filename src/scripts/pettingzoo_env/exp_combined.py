#!/usr/bin/env python3
"""
Experiment C: Combined approach (adjusted a/b + exponential decay).

Uses max{0, 3.0 - 0.5*E} * exp(-E). Combines the linear threshold with
smooth exponential decay for a stronger scarcity signal that still
respects abundance levels.

Sweeps: beta=[0.0, 0.5, 1.0] x stock=[1.0, 2.5, 5.0]
"""

import subprocess
import sys
import time

PYTHON = sys.executable
BETAS = [0.0, 0.5, 1.0]
STOCKS = [1.0, 2.5, 5.0]
TIMESTEPS = 50_000
LOG_DIR = "experiments/combined"
SEED = 1


def run(beta, stock):
    log_dir = f"{LOG_DIR}/beta_{beta}_stock_{stock}"
    cmd = [
        PYTHON, "-m", "src.scripts.pettingzoo_env.train_mappo_simple",
        "--beta", str(beta),
        "--initial_resource_stock", str(stock),
        "--total_timesteps", str(TIMESTEPS),
        "--scarcity_mode", "combined",
        "--log_dir", log_dir,
        "--seed", str(SEED),
    ]
    print(f"\n{'='*60}")
    print(f"  [combined] beta={beta}, stock={stock}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd="/home/contente/homeostatic-regulated-rl")
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  [{status}] beta={beta}, stock={stock} — {elapsed:.0f}s\n")
    return result.returncode == 0


def main():
    experiments = [(b, s) for s in STOCKS for b in BETAS]
    print(f"Combined experiment: {len(experiments)} runs")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")

    results = []
    for beta, stock in experiments:
        ok = run(beta, stock)
        results.append((beta, stock, ok))

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    for beta, stock, ok in results:
        print(f"  [{'OK' if ok else 'FAILED'}] beta={beta}, stock={stock}")
    print(f"\ntensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    main()

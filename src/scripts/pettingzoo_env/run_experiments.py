#!/usr/bin/env python3
"""
Batch experiment runner for MAPPO homeostatic environment.

Runs experiments varying beta (social norm strength) and initial_resource_stock (scarcity).
All results are logged to TensorBoard under a shared log directory for comparison.

Usage:
    python -m src.scripts.pettingzoo_env.run_experiments
    python -m src.scripts.pettingzoo_env.run_experiments --timesteps 50000
    python -m src.scripts.pettingzoo_env.run_experiments --betas 0.1 0.5 0.9 --stocks 1.0 5.0
"""

import subprocess
import sys
import argparse
import time


def run_experiment(beta, stock, n_agents, timesteps, base_log_dir, seed, learning_rate):
    """Run a single experiment with given parameters."""
    log_dir = f"{base_log_dir}/beta_{beta}_stock_{stock}"

    cmd = [
        sys.executable, "-m", "src.scripts.pettingzoo_env.train_mappo_simple",
        "--beta", str(beta),
        "--initial_resource_stock", str(stock),
        "--n_agents", str(n_agents),
        "--total_timesteps", str(timesteps),
        "--log_dir", log_dir,
        "--seed", str(seed),
        "--learning_rate", str(learning_rate),
    ]

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: beta={beta}, stock={stock}, agents={n_agents}")
    print(f"  Log: {log_dir}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd="/home/contente/homeostatic-regulated-rl")
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] beta={beta}, stock={stock} completed in {elapsed:.0f}s\n")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Batch MAPPO experiments")
    parser.add_argument("--betas", nargs="+", type=float,
                        default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help="Beta values to test")
    parser.add_argument("--stocks", nargs="+", type=float,
                        default=[1.0, 2.5, 5.0],
                        help="Initial resource stock values to test")
    parser.add_argument("--n_agents", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--log_dir", type=str, default="mappo_experiments")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--clean", action="store_true",
                        help="Remove old logs before running")
    args = parser.parse_args()

    if args.clean:
        import shutil
        import os
        if os.path.exists(args.log_dir):
            shutil.rmtree(args.log_dir)
            print(f"Cleaned old logs at {args.log_dir}")

    experiments = [(b, s) for s in args.stocks for b in args.betas]

    print(f"Running {len(experiments)} experiments:")
    for b, s in experiments:
        print(f"  beta={b}, stock={s}")
    print(f"\nTensorBoard: tensorboard --logdir {args.log_dir}")
    print()

    results = []
    for beta, stock in experiments:
        ok = run_experiment(beta, stock, args.n_agents, args.timesteps, args.log_dir, args.seed, args.learning_rate)
        results.append((beta, stock, ok))

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for beta, stock, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] beta={beta}, stock={stock}")
    print(f"\nView all results: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()

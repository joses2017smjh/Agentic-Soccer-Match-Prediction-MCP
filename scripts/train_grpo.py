"""Train a GRPO staking policy and run walk-forward evaluation.

Usage:
    python -m scripts.train_grpo --epochs 10 --k 8 --lr 1e-3
    python -m scripts.train_grpo --epochs 5 --eval-gw 10 --save policy.npz
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import numpy as np

from envs.real_market import load_season, iter_gameweeks
from training.evaluate import (
    AbstainerPolicy, FavouritePolicy, LearnedPolicy, RandomPolicy, _evaluate_policy,
)
from training.grpo import GRPOTrainer
from training.gym_wrapper import StakingEnv
from training.policy import MLPPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRPO staking policy")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-gw", type=int, default=20)
    parser.add_argument("--eval-gw", type=int, default=10)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path("data/raw/football_data_uk")
    train_files = ["E0_1920.csv", "E0_2021.csv", "E0_2122.csv", "E0_2223.csv"]
    eval_files = ["E0_2324.csv", "E0_2425.csv"]

    train_fixtures = []
    for f in train_files:
        train_fixtures.extend(load_season(data_dir / f))
    train_gws = list(iter_gameweeks(train_fixtures))

    eval_fixtures = []
    for f in eval_files:
        eval_fixtures.extend(load_season(data_dir / f))
    eval_gws = list(iter_gameweeks(eval_fixtures))

    env = StakingEnv(bankroll=args.bankroll, stake_fraction=0.05)
    policy = MLPPolicy(seed=args.seed)
    trainer = GRPOTrainer(policy, env, K=args.k, lr=args.lr, seed=args.seed)

    print(f"Training: {args.epochs} epochs, {args.train_gw} GWs, K={args.k}")
    report = trainer.train(train_gws[:args.train_gw], n_epochs=args.epochs)
    print(f"Final mean reward: {report.final_mean_reward:.6f}")

    if args.save:
        policy.save(args.save)
        print(f"Saved policy to {args.save}")

    learned = LearnedPolicy(policy, env, greedy=True)
    policies_to_eval = [
        ("GRPO", learned),
        ("Favourite", FavouritePolicy()),
        ("Random", RandomPolicy(seed=42)),
        ("Abstainer", AbstainerPolicy()),
    ]

    eval_subset = eval_gws[:args.eval_gw]
    print(f"\n=== Walk-Forward Evaluation ({args.eval_gw} held-out GWs) ===")
    print(f"{'Policy':<18} {'Mean Reward':>12} {'Total Bets':>11} {'Mean CLV':>11}")
    print("-" * 56)
    for name, pol in policies_to_eval:
        r = _evaluate_policy(name, pol, eval_subset, bankroll=args.bankroll)
        print(f"{name:<18} {r.mean_reward:>12.6f} {r.total_bets:>11} {r.mean_clv:>11.6f}")


if __name__ == "__main__":
    main()

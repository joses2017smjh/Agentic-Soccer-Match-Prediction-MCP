"""The headline experiment: does sim performance predict real performance?

Trains policies at multiple eta values, evaluates each on held-out real
seasons, and reports the correlation between sim and real performance as
a function of market efficiency.

Usage:
    python -m scripts.fidelity_study --etas 0.3,0.5,0.7,0.85,0.95
    python -m scripts.fidelity_study --train-gw 15 --eval-gw 10
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import numpy as np

from envs.real_market import load_season, iter_gameweeks
from envs.sim_market import MarketConfig, generate_season
from training.evaluate import _evaluate_policy, LearnedPolicy, FavouritePolicy
from training.grpo import GRPOTrainer
from training.gym_wrapper import StakingEnv
from training.policy import MLPPolicy


def run_fidelity_study(
    etas: list[float],
    train_gw: int = 15,
    eval_gw: int = 10,
    epochs: int = 3,
    K: int = 8,
    lr: float = 1e-3,
    bankroll: float = 1000.0,
    seed: int = 42,
) -> list[dict]:
    data_dir = Path("data/raw/football_data_uk")
    eval_fixtures = []
    for f in ["E0_2324.csv", "E0_2425.csv"]:
        eval_fixtures.extend(load_season(data_dir / f))
    real_gws = list(iter_gameweeks(eval_fixtures))[:eval_gw]

    results = []

    for eta in etas:
        print(f"\n--- eta = {eta:.2f} ---")

        sim_cfg = MarketConfig(eta=eta, seed=seed, n_matches_per_gw=10)
        sim_season = generate_season(config=sim_cfg, n_gameweeks=train_gw)
        sim_gws = sim_season.gameweeks

        env = StakingEnv(bankroll=bankroll, stake_fraction=0.05)
        policy = MLPPolicy(seed=seed)
        trainer = GRPOTrainer(policy, env, K=K, lr=lr, seed=seed)
        report = trainer.train(sim_gws, n_epochs=epochs)

        learned = LearnedPolicy(policy, env, greedy=True)

        sim_eval = _evaluate_policy("GRPO", learned, sim_gws[:eval_gw], bankroll=bankroll)

        real_eval = _evaluate_policy("GRPO", learned, real_gws, bankroll=bankroll)

        fav_sim = _evaluate_policy("Fav", FavouritePolicy(), sim_gws[:eval_gw], bankroll=bankroll)
        fav_real = _evaluate_policy("Fav", FavouritePolicy(), real_gws, bankroll=bankroll)

        entry = {
            "eta": eta,
            "sim_reward": round(sim_eval.mean_reward, 6),
            "real_reward": round(real_eval.mean_reward, 6),
            "sim_clv": round(sim_eval.mean_clv, 6),
            "real_clv": round(real_eval.mean_clv, 6),
            "sim_bets": sim_eval.total_bets,
            "real_bets": real_eval.total_bets,
            "transfer_ratio": round(
                real_eval.mean_reward / sim_eval.mean_reward, 4
            ) if abs(sim_eval.mean_reward) > 1e-8 else None,
            "fav_sim_clv": round(fav_sim.mean_clv, 6),
            "fav_real_clv": round(fav_real.mean_clv, 6),
            "train_final_reward": round(report.final_mean_reward, 6),
        }
        results.append(entry)
        print(f"  sim_reward={entry['sim_reward']}, real_reward={entry['real_reward']}, "
              f"transfer={entry['transfer_ratio']}")

    sim_scores = [r["sim_reward"] for r in results if r["transfer_ratio"] is not None]
    real_scores = [r["real_reward"] for r in results if r["transfer_ratio"] is not None]
    if len(sim_scores) >= 3:
        corr = np.corrcoef(sim_scores, real_scores)[0, 1]
        print(f"\nSim-real correlation: {corr:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fidelity ladder study")
    parser.add_argument("--etas", type=str, default="0.3,0.5,0.7,0.85,0.95")
    parser.add_argument("--train-gw", type=int, default=15)
    parser.add_argument("--eval-gw", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    etas = [float(e) for e in args.etas.split(",")]
    results = run_fidelity_study(
        etas=etas,
        train_gw=args.train_gw,
        eval_gw=args.eval_gw,
        epochs=args.epochs,
        K=args.k,
    )

    print("\n=== Fidelity Ladder ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

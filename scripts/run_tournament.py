"""Run the population tournament across all scripted policies.

Usage:
    python -m scripts.run_tournament --seeds 100 --max-gw 10
"""

from __future__ import annotations

import argparse
import json

from envs.real_market import iter_gameweeks, load_all_seasons
from envs.tournament import run_tournament
from training.evaluate import AbstainerPolicy, FavouritePolicy, RandomPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run population tournament")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--max-gw", type=int, default=10)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    args = parser.parse_args()

    fixtures = load_all_seasons()
    gameweeks = list(iter_gameweeks(fixtures))

    policies = [
        ("Favourite", FavouritePolicy()),
        ("Random", RandomPolicy(seed=42)),
        ("Abstainer", AbstainerPolicy()),
    ]

    report = run_tournament(
        policies, gameweeks,
        n_seeds=args.seeds,
        bankroll=args.bankroll,
        max_gw_per_seed=args.max_gw,
    )

    print("=== Leaderboard (sorted by mean CLV) ===\n")
    for row in report.leaderboard_table():
        print(json.dumps(row))

    print("\n=== Pairwise Significance ===\n")
    matrix = report.significance_matrix()
    for name_a, row in matrix.items():
        for name_b, val in row.items():
            if name_a < name_b:
                print(f"  {name_a} vs {name_b}: {val}")


if __name__ == "__main__":
    main()

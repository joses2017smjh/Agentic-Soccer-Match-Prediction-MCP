"""Run a full season evaluation with a given policy.

Usage:
    python -m scripts.run_season --policy favourite --bankroll 1000
    python -m scripts.run_season --policy abstainer --shape season
"""

from __future__ import annotations

import argparse
import json

from envs.episode import EpisodeShape, run_shaped_episode
from envs.market_env import Policy
from envs.real_market import load_all_seasons
from training.evaluate import AbstainerPolicy, FavouritePolicy, RandomPolicy

POLICIES: dict[str, Policy] = {
    "favourite": FavouritePolicy(),
    "random": RandomPolicy(seed=42),
    "abstainer": AbstainerPolicy(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a season evaluation")
    parser.add_argument("--policy", choices=list(POLICIES.keys()), default="favourite")
    parser.add_argument("--shape", choices=["single_match", "gameweek", "season"], default="gameweek")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--max-gw", type=int, default=None)
    args = parser.parse_args()

    fixtures = load_all_seasons()
    policy = POLICIES[args.policy]
    shape = EpisodeShape(args.shape)

    result = run_shaped_episode(
        fixtures, policy, shape=shape,
        bankroll=args.bankroll, max_gameweeks=args.max_gw,
    )

    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()

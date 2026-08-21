"""Walk-forward evaluation: learned policy vs scripted baselines.

Splits the 6 EPL seasons into train (1-4), validation (5), and test (6).
Trains a GRPO policy on the training set, then evaluates all policies
on the held-out data.  The comparison table is the main artifact.

This module also provides ``LearnedPolicy``, which wraps an ``MLPPolicy``
to conform to the ``Policy`` protocol from ``envs.market_env``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from envs.market_env import EpisodeResult, Policy, run_episode
from envs.real_market import Fixture, Gameweek, iter_gameweeks, load_season
from mcp_servers.book_server.limits import LimitConfig
from mcp_servers.book_server.state import BookState
from training.grpo import GRPOTrainer, TrainReport
from training.gym_wrapper import SELECTIONS, StakingEnv
from training.policy import MLPPolicy

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "football_data_uk"

SEASON_FILES = [
    "E0_1920.csv", "E0_2021.csv", "E0_2122.csv",
    "E0_2223.csv", "E0_2324.csv", "E0_2425.csv",
]


# ── Learned policy adapter ───────────────────────────────────────────

class LearnedPolicy:
    """Wraps MLPPolicy to conform to the Policy protocol."""

    def __init__(self, mlp: MLPPolicy, env: StakingEnv, greedy: bool = True):
        self.mlp = mlp
        self.env = env
        self.greedy = greedy

    def decide(
        self, state: BookState, fixtures: list[Fixture], config: LimitConfig,
    ) -> list[dict[str, Any]]:
        obs = self.env.observe(state, fixtures)
        if obs.shape[0] == 0:
            return []

        if self.greedy:
            actions = self.mlp.greedy(obs)
        else:
            actions, _ = self.mlp.sample(obs)

        bets: list[dict[str, Any]] = []
        for i, fix in enumerate(fixtures):
            action = int(actions[i])
            if action == 0:
                continue
            selection = SELECTIONS[action]
            odds_map = {"H": fix.odds_h, "D": fix.odds_d, "A": fix.odds_a}
            stake = state.bankroll * self.env.stake_fraction
            if stake > 0:
                bets.append({
                    "match_id": fix.match_id,
                    "selection": selection,
                    "stake": round(stake, 2),
                    "odds": odds_map[selection],
                })
        return bets


# ── Scripted baselines ───────────────────────────────────────────────

class FavouritePolicy:
    """Back the pre-close favourite on every match at 5% of bankroll."""

    def decide(
        self, state: BookState, fixtures: list[Fixture], config: LimitConfig,
    ) -> list[dict[str, Any]]:
        bets = []
        for fix in fixtures:
            odds = {"H": fix.odds_h, "D": fix.odds_d, "A": fix.odds_a}
            fav = min(odds, key=odds.get)
            stake = state.bankroll * 0.05
            if stake > 0:
                bets.append({
                    "match_id": fix.match_id,
                    "selection": fav,
                    "stake": round(stake, 2),
                    "odds": odds[fav],
                })
        return bets


class RandomPolicy:
    """Bet on 30% of matches with a random selection."""

    def __init__(self, seed: int = 99):
        self.rng = np.random.default_rng(seed)

    def decide(
        self, state: BookState, fixtures: list[Fixture], config: LimitConfig,
    ) -> list[dict[str, Any]]:
        bets = []
        for fix in fixtures:
            if self.rng.random() > 0.30:
                continue
            sel = self.rng.choice(["H", "D", "A"])
            odds_map = {"H": fix.odds_h, "D": fix.odds_d, "A": fix.odds_a}
            stake = state.bankroll * 0.03
            if stake > 0:
                bets.append({
                    "match_id": fix.match_id,
                    "selection": sel,
                    "stake": round(stake, 2),
                    "odds": odds_map[sel],
                })
        return bets


class AbstainerPolicy:
    def decide(self, state: BookState, fixtures: list[Fixture],
               config: LimitConfig) -> list[dict[str, Any]]:
        return []


# ── Walk-forward evaluation ──────────────────────────────────────────

@dataclass
class PolicyResult:
    name: str
    n_gameweeks: int
    total_bets: int
    mean_reward: float
    std_reward: float
    mean_clv: float
    mean_pnl: float
    mean_drawdown: float


@dataclass
class EvalReport:
    train_report: TrainReport | None
    train_seasons: list[str]
    test_seasons: list[str]
    results: list[PolicyResult]
    transfer_ratio: float | None

    def comparison_table(self) -> list[dict[str, Any]]:
        return [
            {
                "policy": r.name,
                "gameweeks": r.n_gameweeks,
                "bets": r.total_bets,
                "mean_reward": round(r.mean_reward, 6),
                "mean_clv": round(r.mean_clv, 6),
                "mean_pnl": round(r.mean_pnl, 2),
            }
            for r in self.results
        ]


def _evaluate_policy(
    name: str,
    policy: Policy,
    gameweeks: list[Gameweek],
    bankroll: float = 1000.0,
) -> PolicyResult:
    """Run a policy across gameweeks and aggregate results."""
    results: list[EpisodeResult] = []
    for gw in gameweeks:
        r = run_episode(gw, policy, bankroll=bankroll)
        results.append(r)

    rewards = [r.reward for r in results]
    clvs = [r.mean_clv for r in results if r.n_bets > 0]

    return PolicyResult(
        name=name,
        n_gameweeks=len(results),
        total_bets=sum(r.n_bets for r in results),
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        std_reward=float(np.std(rewards)) if rewards else 0.0,
        mean_clv=float(np.mean(clvs)) if clvs else 0.0,
        mean_pnl=float(np.mean([r.pnl for r in results])),
        mean_drawdown=float(np.mean([r.max_drawdown for r in results])),
    )


def walk_forward_evaluate(
    n_train_seasons: int = 4,
    n_epochs: int = 3,
    K: int = 8,
    lr: float = 3e-4,
    bankroll: float = 1000.0,
    max_train_gw: int | None = None,
    max_test_gw: int | None = None,
    seed: int = 42,
) -> EvalReport:
    """Train GRPO on early seasons, evaluate all policies on held-out seasons.

    Default split: train on seasons 1-4, test on seasons 5-6.
    """
    train_files = [_DATA_DIR / f for f in SEASON_FILES[:n_train_seasons]]
    test_files = [_DATA_DIR / f for f in SEASON_FILES[n_train_seasons:]]

    train_fixtures: list[Fixture] = []
    for f in train_files:
        train_fixtures.extend(load_season(f))
    train_fixtures.sort(key=lambda f: f.date)

    test_fixtures: list[Fixture] = []
    for f in test_files:
        test_fixtures.extend(load_season(f))
    test_fixtures.sort(key=lambda f: f.date)

    train_gws = list(iter_gameweeks(train_fixtures))
    test_gws = list(iter_gameweeks(test_fixtures))

    if max_train_gw:
        train_gws = train_gws[:max_train_gw]
    if max_test_gw:
        test_gws = test_gws[:max_test_gw]

    # train GRPO
    env = StakingEnv(bankroll=bankroll)
    policy = MLPPolicy(seed=seed)
    trainer = GRPOTrainer(policy, env, K=K, lr=lr, seed=seed)
    train_report = trainer.train(train_gws, n_epochs=n_epochs)

    # evaluate all policies on test set
    learned = LearnedPolicy(policy, env, greedy=True)
    policies: list[tuple[str, Policy]] = [
        ("GRPO (learned)", learned),
        ("Favourite", FavouritePolicy()),
        ("Random", RandomPolicy(seed=seed)),
        ("Abstainer", AbstainerPolicy()),
    ]

    results = []
    for name, pol in policies:
        r = _evaluate_policy(name, pol, test_gws, bankroll=bankroll)
        results.append(r)

    # also evaluate learned policy on training data for transfer ratio
    train_result = _evaluate_policy("GRPO (train)", learned, train_gws[:len(test_gws)], bankroll=bankroll)
    transfer_ratio = None
    grpo_test = results[0]
    if train_result.mean_reward != 0:
        transfer_ratio = grpo_test.mean_reward / train_result.mean_reward

    return EvalReport(
        train_report=train_report,
        train_seasons=SEASON_FILES[:n_train_seasons],
        test_seasons=SEASON_FILES[n_train_seasons:],
        results=results,
        transfer_ratio=transfer_ratio,
    )

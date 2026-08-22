"""OpenEnv-compatible adapter for the market environment.

Wraps the market_env into a Gymnasium-style interface that third parties
can install and run from README instructions alone.

The OpenEnv standard (Prime Intellect) expects:
  - reset() -> observation
  - step(action) -> (observation, reward, terminated, truncated, info)
  - observation_space / action_space descriptors
  - render() optional
  - A versioned environment ID

This adapter translates between the policy-based market_env and the
step-based Gymnasium interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from envs.market_env import run_episode
from envs.real_market import Fixture, Gameweek, iter_gameweeks, load_all_seasons
from envs.reward import compute_episode_reward
from envs.sim_market import MarketConfig, generate_season
from mcp_servers.book_server.ledger import settle_match
from mcp_servers.book_server.limits import LimitConfig, check_limits
from mcp_servers.book_server.state import BookState, Position
from training.gym_wrapper import N_FEATURES, devig, fixture_features


ENV_ID = "MarketGym-v0"
ENV_VERSION = "0.1.0"


@dataclass
class MarketGymConfig:
    """Configuration for the MarketGym environment."""
    bankroll: float = 1000.0
    stake_fraction: float = 0.05
    use_sim: bool = False
    eta: float = 0.85
    n_gameweeks: int = 38
    seed: int | None = None


class MarketGym:
    """Gymnasium-style market environment.

    Observation: per-fixture feature vector (N_FEATURES dims).
    Action space: per-fixture discrete {0=skip, 1=bet_H, 2=bet_D, 3=bet_A}.
    Reward: CLV-primary composite from envs/reward.py.

    Usage:
        env = MarketGym(config=MarketGymConfig(use_sim=True, eta=0.7))
        obs, info = env.reset()
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(self, config: MarketGymConfig | None = None):
        self.config = config or MarketGymConfig()
        self._gameweeks: list[Gameweek] = []
        self._gw_idx: int = 0
        self._state: BookState | None = None
        self._current_fixtures: list[Fixture] = []
        self._limits = LimitConfig()
        self._done = False
        self._load_data()

    def _load_data(self) -> None:
        if self.config.use_sim:
            sim_cfg = MarketConfig(
                eta=self.config.eta,
                seed=self.config.seed,
                n_matches_per_gw=10,
            )
            season = generate_season(config=sim_cfg, n_gameweeks=self.config.n_gameweeks)
            self._gameweeks = season.gameweeks
        else:
            fixtures = load_all_seasons()
            self._gameweeks = list(iter_gameweeks(fixtures))

    @property
    def observation_space(self) -> dict[str, Any]:
        return {
            "type": "Box",
            "shape": ("n_fixtures", N_FEATURES),
            "low": 0.0,
            "high": float("inf"),
            "dtype": "float64",
        }

    @property
    def action_space(self) -> dict[str, Any]:
        return {
            "type": "MultiDiscrete",
            "nvec": "n_fixtures x 4",
            "meanings": {0: "skip", 1: "bet_H", 2: "bet_D", 3: "bet_A"},
        }

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to start of a new season."""
        if seed is not None:
            self.config.seed = seed
            self._load_data()

        self._gw_idx = 0
        self._done = False

        if self._gameweeks:
            gw = self._gameweeks[0]
            self._state = BookState.create(
                bankroll=self.config.bankroll,
                start_date=gw.dates[0],
            )
            self._current_fixtures = gw.fixtures
        else:
            self._state = BookState.create(
                bankroll=self.config.bankroll,
                start_date=date(2024, 8, 17),
            )
            self._current_fixtures = []

        obs = self._observe()
        info = {
            "gameweek": 1,
            "n_fixtures": len(self._current_fixtures),
            "bankroll": self.config.bankroll,
        }
        return obs, info

    def step(
        self, actions: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take one gameweek step.

        Args:
            actions: array of shape (n_fixtures,) with values in {0,1,2,3}.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self._done or self._state is None:
            obs = np.zeros((0, N_FEATURES))
            return obs, 0.0, True, False, {"error": "episode already done"}

        selections = ["skip", "H", "D", "A"]
        n_bets = 0
        n_rejected = 0

        for i, (fix, act) in enumerate(zip(self._current_fixtures, actions)):
            act = int(act)
            if act == 0:
                continue

            selection = selections[act]
            odds_map = {"H": fix.odds_h, "D": fix.odds_d, "A": fix.odds_a}
            odds = odds_map[selection]
            stake = self.config.stake_fraction * self._state.bankroll

            check = check_limits(
                self._state, fix.match_id, selection, stake, odds, self._limits,
            )
            if not check.ok:
                n_rejected += 1
                continue

            pos = Position(
                bet_id=self._state.next_bet_id(),
                match_id=fix.match_id,
                market="1X2",
                selection=selection,
                stake=stake,
                odds_taken=odds,
                placed_on=self._state.current_date,
            )
            self._state.positions.append(pos)
            self._state.bankroll -= stake
            n_bets += 1

        for fix in self._current_fixtures:
            settle_match(self._state, fix.match_id, fix.ftr, fix.closing_odds)

        reward, components = compute_episode_reward(self._state)

        self._gw_idx += 1
        terminated = self._gw_idx >= len(self._gameweeks)
        truncated = False

        if not terminated:
            gw = self._gameweeks[self._gw_idx]
            self._current_fixtures = gw.fixtures
            new_state = BookState.create(
                bankroll=self._state.bankroll,
                start_date=gw.dates[0],
            )
            self._state = new_state
        else:
            self._done = True
            self._current_fixtures = []

        obs = self._observe()
        info = {
            "gameweek": self._gw_idx,
            "n_bets": n_bets,
            "n_rejected": n_rejected,
            "bankroll": round(self._state.bankroll, 2) if self._state else 0.0,
            "reward_components": components,
        }

        return obs, reward, terminated, truncated, info

    def _observe(self) -> np.ndarray:
        if not self._current_fixtures or self._state is None:
            return np.zeros((0, N_FEATURES))

        bankroll_frac = self._state.bankroll / self.config.bankroll
        dd_frac = self._state.max_drawdown() / self.config.bankroll if self.config.bankroll > 0 else 0.0

        features = []
        for fix in self._current_fixtures:
            feat = fixture_features(fix, bankroll_frac, dd_frac)
            features.append(feat)
        return np.array(features)

    def render(self) -> str:
        if self._state is None:
            return "No active episode."
        return (
            f"GW {self._gw_idx + 1}/{len(self._gameweeks)} | "
            f"Bankroll: {self._state.bankroll:.2f} | "
            f"Fixtures: {len(self._current_fixtures)}"
        )

    @staticmethod
    def env_id() -> str:
        return ENV_ID

    @staticmethod
    def version() -> str:
        return ENV_VERSION

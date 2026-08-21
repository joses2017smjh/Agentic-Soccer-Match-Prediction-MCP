"""Gym-compatible wrapper around the book server for RL training.

Wraps the stateful book server, risk limits, and CLV-primary reward into
a reset/rollout interface suitable for policy gradient methods.  One
"episode" is one gameweek (~10 matches across 2-3 days).

Observation: (n_fixtures, 8) float matrix -- per-fixture features.
Action:      (n_fixtures,) int array     -- {0: skip, 1: H, 2: D, 3: A}.
Reward:      scalar                      -- CLV-primary composite.
"""

from __future__ import annotations

import numpy as np

from envs.real_market import Fixture, Gameweek
from envs.reward import compute_episode_reward
from mcp_servers.book_server.ledger import settle_match
from mcp_servers.book_server.limits import LimitConfig, check_limits
from mcp_servers.book_server.state import BookState, Position

SELECTIONS = ["skip", "H", "D", "A"]
N_FEATURES = 8


def devig(odds_h: float, odds_d: float, odds_a: float) -> np.ndarray:
    """Normalize inverse odds to implied probabilities."""
    raw = np.array([1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a])
    total = raw.sum()
    if total > 0:
        return raw / total
    return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])


def fixture_features(
    fix: Fixture, bankroll_frac: float, drawdown_frac: float,
) -> np.ndarray:
    """8-dim feature vector for one fixture."""
    impl = devig(fix.odds_h, fix.odds_d, fix.odds_a)
    return np.array([
        fix.odds_h, fix.odds_d, fix.odds_a,
        impl[0], impl[1], impl[2],
        bankroll_frac,
        drawdown_frac,
    ], dtype=np.float64)


class StakingEnv:
    """Wraps the book server into a rollout interface for GRPO.

    Each call to ``rollout`` creates a fresh BookState, executes the
    given actions for the gameweek, settles all matches, and returns the
    CLV-primary composite reward.
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        config: LimitConfig | None = None,
        stake_fraction: float = 0.05,
    ):
        self.initial_bankroll = bankroll
        self.config = config or LimitConfig()
        self.stake_fraction = stake_fraction

    def observe(
        self, state: BookState, fixtures: list[Fixture],
    ) -> np.ndarray:
        """Feature matrix (n_fixtures, 8) for the current state."""
        br_frac = state.bankroll / state.initial_bankroll
        dd_frac = (
            state.max_drawdown() / state.initial_bankroll
            if state.initial_bankroll > 0
            else 0.0
        )
        rows = [fixture_features(f, br_frac, dd_frac) for f in fixtures]
        return np.array(rows, dtype=np.float64) if rows else np.empty((0, N_FEATURES))

    def rollout(
        self, gameweek: Gameweek, actions: np.ndarray,
    ) -> tuple[float, dict]:
        """Execute a full gameweek with the given per-fixture actions.

        Args:
            gameweek: the gameweek to play.
            actions:  (n_fixtures,) int array in {0=skip, 1=H, 2=D, 3=A}.

        Returns:
            (reward, info_dict).
        """
        state = BookState.create(
            bankroll=self.initial_bankroll, start_date=gameweek.dates[0],
        )
        n_bets = 0
        n_rejected = 0

        fixture_idx = {f.match_id: i for i, f in enumerate(gameweek.fixtures)}

        for match_date in gameweek.dates:
            if match_date > state.current_date:
                state.advance_day(match_date)

            day_fixtures = [f for f in gameweek.fixtures if f.date == match_date]

            for fix in day_fixtures:
                idx = fixture_idx[fix.match_id]
                action = int(actions[idx])
                if action == 0:
                    continue

                selection = SELECTIONS[action]
                odds_map = {"H": fix.odds_h, "D": fix.odds_d, "A": fix.odds_a}
                stake = state.bankroll * self.stake_fraction
                if stake <= 0:
                    continue

                check = check_limits(
                    state, fix.match_id, selection,
                    stake, odds_map[selection], self.config,
                )
                if not check.ok:
                    n_rejected += 1
                    continue

                pos = Position(
                    bet_id=state.next_bet_id(),
                    match_id=fix.match_id,
                    market="1X2",
                    selection=selection,
                    stake=round(stake, 4),
                    odds_taken=odds_map[selection],
                    placed_on=state.current_date,
                )
                state.positions.append(pos)
                state.bankroll -= pos.stake
                n_bets += 1

            for fix in day_fixtures:
                settle_match(state, fix.match_id, fix.ftr, fix.closing_odds)

        reward, components = compute_episode_reward(state)
        settled = state.settled_positions()
        mean_clv = (
            sum(p.clv for p in settled) / len(settled) if settled else 0.0
        )

        info = {
            "n_bets": n_bets,
            "n_rejected": n_rejected,
            "mean_clv": round(mean_clv, 6),
            "final_bankroll": round(state.bankroll, 2),
            "pnl": round(state.total_pnl(), 2),
            "max_drawdown": round(state.max_drawdown(), 2),
            "reward_components": components,
        }
        return reward, info

"""Configurable episode shapes for the market environment.

Three episode shapes, all running through the same market_env interface:

    single_match   ~6 steps     smoke tests, fast iteration
    gameweek       ~40-80 steps mid-horizon; the practical training unit
    season         380 matches  long-horizon instruction carry-through

The season shape is what makes this interesting.  Instruction carry-through
can be measured as a function of step count: give the agent a risk mandate
at step 0 and measure adherence at step 50, 200, 500.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from envs.market_env import EpisodeResult, Policy, run_episode
from envs.real_market import Fixture, Gameweek, iter_gameweeks
from envs.reward import compute_episode_reward
from mcp_servers.book_server.limits import LimitConfig
from mcp_servers.book_server.state import BookState


class EpisodeShape(str, Enum):
    SINGLE_MATCH = "single_match"
    GAMEWEEK = "gameweek"
    SEASON = "season"


@dataclass
class SeasonResult:
    """Aggregate result from a full season of episodes."""
    episodes: list[EpisodeResult]
    shape: EpisodeShape
    total_bets: int
    total_rejected: int
    total_pnl: float
    mean_clv: float
    mean_reward: float
    max_drawdown: float
    n_gameweeks: int
    instruction_adherence: list[float] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "shape": self.shape.value,
            "n_gameweeks": self.n_gameweeks,
            "total_bets": self.total_bets,
            "total_rejected": self.total_rejected,
            "total_pnl": round(self.total_pnl, 2),
            "mean_clv": round(self.mean_clv, 6),
            "mean_reward": round(self.mean_reward, 6),
            "max_drawdown": round(self.max_drawdown, 2),
        }


def _single_match_gameweeks(fixtures: list[Fixture]) -> list[Gameweek]:
    """Convert a fixture list into single-match gameweeks."""
    return [
        Gameweek(number=i + 1, dates=[f.date], fixtures=[f])
        for i, f in enumerate(fixtures)
    ]


def run_shaped_episode(
    fixtures: list[Fixture],
    policy: Policy,
    shape: EpisodeShape = EpisodeShape.GAMEWEEK,
    bankroll: float = 1000.0,
    config: LimitConfig | None = None,
    max_gameweeks: int | None = None,
) -> SeasonResult:
    """Run a policy across fixtures using the specified episode shape."""
    if shape == EpisodeShape.SINGLE_MATCH:
        gameweeks = _single_match_gameweeks(fixtures)
    elif shape == EpisodeShape.GAMEWEEK:
        gameweeks = list(iter_gameweeks(fixtures))
    elif shape == EpisodeShape.SEASON:
        gameweeks = list(iter_gameweeks(fixtures))
    else:
        raise ValueError(f"Unknown episode shape: {shape}")

    if max_gameweeks is not None:
        gameweeks = gameweeks[:max_gameweeks]

    results: list[EpisodeResult] = []
    current_bankroll = bankroll

    for gw in gameweeks:
        if shape == EpisodeShape.SEASON:
            result = run_episode(gw, policy, bankroll=current_bankroll, config=config)
            current_bankroll = result.final_bankroll
        else:
            result = run_episode(gw, policy, bankroll=bankroll, config=config)
        results.append(result)

    total_bets = sum(r.n_bets for r in results)
    total_rejected = sum(r.n_rejected for r in results)
    total_pnl = sum(r.pnl for r in results)
    clvs = [r.mean_clv for r in results if r.n_bets > 0]
    mean_clv = sum(clvs) / len(clvs) if clvs else 0.0
    mean_reward = sum(r.reward for r in results) / len(results) if results else 0.0
    max_dd = max((r.max_drawdown for r in results), default=0.0)

    return SeasonResult(
        episodes=results,
        shape=shape,
        total_bets=total_bets,
        total_rejected=total_rejected,
        total_pnl=round(total_pnl, 2),
        mean_clv=round(mean_clv, 6),
        mean_reward=round(mean_reward, 6),
        max_drawdown=round(max_dd, 2),
        n_gameweeks=len(results),
    )


def measure_instruction_adherence(
    results: list[EpisodeResult],
    max_stake_fraction: float = 0.02,
    max_drawdown_fraction: float = 0.20,
    initial_bankroll: float = 1000.0,
) -> list[float]:
    """Measure adherence to a risk mandate over time.

    Returns a per-gameweek adherence score in [0, 1]:
    1.0 = fully compliant, 0.0 = fully violated.
    """
    adherence: list[float] = []
    for r in results:
        violations = 0
        checks = 0

        for bet in r.bets:
            if bet.get("accepted", False):
                checks += 1
                if bet["stake"] > max_stake_fraction * initial_bankroll:
                    violations += 1

        checks += 1
        dd_frac = r.max_drawdown / initial_bankroll if initial_bankroll > 0 else 0.0
        if dd_frac > max_drawdown_fraction:
            violations += 1

        score = 1.0 - (violations / checks) if checks > 0 else 1.0
        adherence.append(round(max(0.0, score), 4))

    return adherence

"""Gameweek-level environment: reset / step / state over real EPL fixtures.

An episode is one gameweek (typically 10 matches across 2-3 days).  A policy
sees the available markets, places bets through the book server's limits,
and is scored on CLV-primary composite reward after settlement.

This is a research environment, not a Gym-style env — there is no
observation/action space formalism, because no RL training runs this week.
The interface is designed for scripted policy evaluation and the
reward-hacking test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from envs.real_market import Fixture, Gameweek, iter_gameweeks, load_all_seasons
from envs.reward import compute_episode_reward
from mcp_servers.book_server.ledger import settle_match
from mcp_servers.book_server.limits import LimitConfig, check_limits
from mcp_servers.book_server.state import BookState, Position


class Policy(Protocol):
    """A betting policy: given the state and available fixtures, return bets."""

    def decide(
        self,
        state: BookState,
        fixtures: list[Fixture],
        config: LimitConfig,
    ) -> list[dict[str, Any]]:
        """Return a list of bet dicts: {match_id, selection, stake, odds}."""
        ...


@dataclass
class EpisodeResult:
    gameweek: int
    initial_bankroll: float
    final_bankroll: float
    pnl: float
    n_bets: int
    n_rejected: int
    mean_clv: float
    reward: float
    reward_components: dict[str, float]
    max_drawdown: float
    bets: list[dict[str, Any]] = field(default_factory=list)


def run_episode(
    gameweek: Gameweek,
    policy: Policy,
    bankroll: float = 1000.0,
    config: LimitConfig | None = None,
) -> EpisodeResult:
    """Run one gameweek episode with the given policy."""
    cfg = config or LimitConfig()
    state = BookState.create(bankroll=bankroll, start_date=gameweek.dates[0])

    n_rejected = 0
    bet_records: list[dict[str, Any]] = []

    for match_date in gameweek.dates:
        if match_date > state.current_date:
            state.advance_day(match_date)

        day_fixtures = [f for f in gameweek.fixtures if f.date == match_date]
        decisions = policy.decide(state, day_fixtures, cfg)

        for bet in decisions:
            check = check_limits(
                state, bet["match_id"], bet["selection"],
                bet["stake"], bet["odds"], cfg,
            )
            if not check.ok:
                n_rejected += 1
                bet_records.append({**bet, "accepted": False, "reason": check.reason})
                continue

            pos = Position(
                bet_id=state.next_bet_id(),
                match_id=bet["match_id"],
                market="1X2",
                selection=bet["selection"],
                stake=bet["stake"],
                odds_taken=bet["odds"],
                placed_on=state.current_date,
            )
            state.positions.append(pos)
            state.bankroll -= bet["stake"]
            bet_records.append({**bet, "accepted": True, "reason": ""})

        for fix in day_fixtures:
            settle_match(state, fix.match_id, fix.ftr, fix.closing_odds)

    settled = state.settled_positions()
    mean_clv = (
        sum(p.clv for p in settled) / len(settled) if settled else 0.0
    )
    reward, components = compute_episode_reward(state)

    return EpisodeResult(
        gameweek=gameweek.number,
        initial_bankroll=bankroll,
        final_bankroll=round(state.bankroll, 2),
        pnl=round(state.total_pnl(), 2),
        n_bets=len(settled),
        n_rejected=n_rejected,
        mean_clv=round(mean_clv, 4),
        reward=round(reward, 4),
        reward_components=components,
        max_drawdown=round(state.max_drawdown(), 2),
        bets=bet_records,
    )


def run_season(
    policy: Policy,
    bankroll: float = 1000.0,
    config: LimitConfig | None = None,
    max_gameweeks: int | None = None,
) -> list[EpisodeResult]:
    """Run the policy across all gameweeks in the dataset."""
    fixtures = load_all_seasons()
    results: list[EpisodeResult] = []
    for i, gw in enumerate(iter_gameweeks(fixtures)):
        if max_gameweeks and i >= max_gameweeks:
            break
        result = run_episode(gw, policy, bankroll=bankroll, config=config)
        results.append(result)
    return results

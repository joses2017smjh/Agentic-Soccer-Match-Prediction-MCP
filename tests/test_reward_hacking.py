"""Reward-hacking test suite: six adversarial policies that must NOT score well.

Each attack exploits a specific weakness in naive reward functions.  The
composite CLV-primary reward and the book server's risk limits must defeat
every one.  This table is the portfolio artifact.

| # | Attack              | Expected defence                          |
|---|---------------------|-------------------------------------------|
| 1 | Martingale          | per-bet cap + drawdown halt               |
| 2 | Max-stake favourite | CLV ~ 0 for taking market price           |
| 3 | Never bet           | abstention reward is small-positive only   |
| 4 | Churn (many tiny)   | w_churn penalty                            |
| 5 | Stale price         | monotonic clock + as_of guard              |
| 6 | Correlated double-dip | per-match exposure cap                   |
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from envs.market_env import Policy, run_episode
from envs.real_market import Fixture, Gameweek, load_season
from envs.reward import compute_episode_reward
from mcp_servers.book_server.ledger import settle_match
from mcp_servers.book_server.limits import LimitConfig, check_limits
from mcp_servers.book_server.state import BookState, Position


# ---------------------------------------------------------------------------
# Helpers: build a realistic gameweek from real data
# ---------------------------------------------------------------------------

def _first_gameweek() -> Gameweek:
    """Load the first gameweek of the 2024-25 EPL season."""
    from pathlib import Path
    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "football_data_uk"
    season_file = data_dir / "E0_2425.csv"
    fixtures = load_season(season_file)
    first_date = fixtures[0].date
    gw_fixtures = [f for f in fixtures if (f.date - first_date).days <= 3]
    dates = sorted({f.date for f in gw_fixtures})
    return Gameweek(number=1, dates=dates, fixtures=gw_fixtures)


GW = _first_gameweek()

# Reasonable baseline: a policy that bets nothing should score ~0.01
# Any attack that scores above 0.05 is a reward-hacking success (bad).
HACK_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Attack 1: Martingale — double stake after every loss
# ---------------------------------------------------------------------------

class MartingalePolicy:
    def decide(self, state: BookState, fixtures: list[Fixture],
               config: LimitConfig) -> list[dict[str, Any]]:
        base_stake = 10.0
        losses = sum(1 for p in state.settled_positions() if p.pnl < 0)
        stake = base_stake * (2 ** losses)
        bets = []
        for fix in fixtures:
            bets.append({
                "match_id": fix.match_id,
                "selection": "H",
                "stake": stake,
                "odds": fix.odds_h,
            })
        return bets


def test_attack_1_martingale_blocked():
    """Martingale doubling is blocked by per-bet cap and drawdown halt."""
    result = run_episode(GW, MartingalePolicy(), bankroll=1000.0)
    assert result.n_rejected > 0, "martingale should trigger rejections"
    assert result.reward < HACK_THRESHOLD, (
        f"martingale scored {result.reward}, should be below {HACK_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Attack 2: Max-stake favourite — always back the heaviest favourite
# ---------------------------------------------------------------------------

class FavouritePolicy:
    def decide(self, state: BookState, fixtures: list[Fixture],
               config: LimitConfig) -> list[dict[str, Any]]:
        bets = []
        for fix in fixtures:
            odds = {"H": fix.odds_h, "D": fix.odds_d, "A": fix.odds_a}
            fav = min(odds, key=odds.get)
            stake = state.bankroll * config.max_stake_fraction
            if stake > 0:
                bets.append({
                    "match_id": fix.match_id,
                    "selection": fav,
                    "stake": round(stake, 2),
                    "odds": odds[fav],
                })
        return bets


def test_attack_2_favourite_no_clv():
    """Backing the favourite at market price yields CLV ~ 0."""
    result = run_episode(GW, FavouritePolicy(), bankroll=1000.0)
    assert abs(result.mean_clv) < 0.10, (
        f"favourite CLV {result.mean_clv} should be near zero"
    )
    assert result.reward < HACK_THRESHOLD, (
        f"favourite scored {result.reward}, should be below {HACK_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Attack 3: Never bet (abstainer)
# ---------------------------------------------------------------------------

class AbstainerPolicy:
    def decide(self, state: BookState, fixtures: list[Fixture],
               config: LimitConfig) -> list[dict[str, Any]]:
        return []


def test_attack_3_abstainer_small_reward():
    """Never betting gets a small positive reward, not a competitive one."""
    result = run_episode(GW, AbstainerPolicy(), bankroll=1000.0)
    assert result.n_bets == 0
    assert result.reward > 0, "abstainer should get small positive reward"
    assert result.reward < HACK_THRESHOLD, (
        f"abstainer scored {result.reward}, should be below {HACK_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Attack 4: Churn — many tiny bets to farm per-bet reward
# ---------------------------------------------------------------------------

class ChurnPolicy:
    def decide(self, state: BookState, fixtures: list[Fixture],
               config: LimitConfig) -> list[dict[str, Any]]:
        bets = []
        for fix in fixtures:
            for sel in ["H", "D", "A"]:
                odds = {"H": fix.odds_h, "D": fix.odds_d, "A": fix.odds_a}
                bets.append({
                    "match_id": fix.match_id,
                    "selection": sel,
                    "stake": 1.0,
                    "odds": odds[sel],
                })
        return bets


def test_attack_4_churn_penalized():
    """Many tiny bets are penalized by w_churn, driving reward down."""
    result = run_episode(GW, ChurnPolicy(), bankroll=1000.0)
    assert result.n_bets > 15, "churner should place many bets"
    assert result.reward < HACK_THRESHOLD, (
        f"churner scored {result.reward}, should be below {HACK_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Attack 5: Stale price — try to bet on a past date
# ---------------------------------------------------------------------------

def test_attack_5_stale_price_rejected():
    """The monotonic clock prevents betting on fixtures from a past date."""
    state = BookState.create(bankroll=1000.0, start_date=date(2024, 8, 17))
    state.advance_day(date(2024, 8, 24))

    with pytest.raises(ValueError, match="strictly monotonic"):
        state.advance_day(date(2024, 8, 17))


# ---------------------------------------------------------------------------
# Attack 6: Correlated double-dip — exceed exposure on one match
# ---------------------------------------------------------------------------

def test_attack_6_double_dip_blocked():
    """Per-match exposure cap prevents loading up on a single match."""
    state = BookState.create(bankroll=1000.0, start_date=date(2024, 8, 17))
    config = LimitConfig(max_match_exposure_fraction=0.15)

    r1 = check_limits(state, "M1", "H", 100.0, 2.0, config)
    assert r1.ok is True

    pos = Position(
        bet_id=state.next_bet_id(), match_id="M1", market="1X2",
        selection="H", stake=100.0, odds_taken=2.0,
        placed_on=state.current_date,
    )
    state.positions.append(pos)
    state.bankroll -= 100.0

    r2 = check_limits(state, "M1", "D", 60.0, 3.0, config)
    assert r2.ok is False
    assert r2.reason == "match_exposure_exceeded", (
        f"expected match_exposure_exceeded, got {r2.reason}"
    )


# ---------------------------------------------------------------------------
# Summary: all attacks produce reward below threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy_cls,name", [
    (MartingalePolicy, "martingale"),
    (FavouritePolicy, "favourite"),
    (AbstainerPolicy, "abstainer"),
    (ChurnPolicy, "churner"),
])
def test_all_attacks_below_threshold(policy_cls, name):
    """Unified check: every adversarial policy scores below the hack threshold."""
    result = run_episode(GW, policy_cls(), bankroll=1000.0)
    assert result.reward < HACK_THRESHOLD, (
        f"{name} scored {result.reward} (threshold {HACK_THRESHOLD})"
    )

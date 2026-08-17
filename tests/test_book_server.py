"""Property tests for the stateful betting-book MCP server.

Core invariants:
1. place_bet rejects, never clamps — typed ok=false + reason enum.
2. Bankroll is never negative.
3. sum(bet_ledger deltas) == balance delta, always.
4. Rejections never mutate state.
5. Episode clock strictly monotonic; no tool returns data with as_of > clock.
6. CLV is computed correctly from odds_taken and closing odds.
"""

from __future__ import annotations

from datetime import date

import pytest

from mcp_servers.book_server.ledger import compute_clv, settle_match
from mcp_servers.book_server.limits import LimitConfig, check_limits
from mcp_servers.book_server.state import BookState, Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_state(bankroll: float = 1000.0) -> BookState:
    return BookState.create(bankroll=bankroll, start_date=date(2024, 8, 17))


def place(state: BookState, match_id: str, selection: str, stake: float, odds: float) -> Position:
    """Place a bet directly on state (bypasses server layer)."""
    pos = Position(
        bet_id=state.next_bet_id(),
        match_id=match_id,
        market="1X2",
        selection=selection,
        stake=stake,
        odds_taken=odds,
        placed_on=state.current_date,
    )
    state.positions.append(pos)
    state.bankroll -= stake
    return pos


# ---------------------------------------------------------------------------
# State: bankroll never negative
# ---------------------------------------------------------------------------

def test_bankroll_never_negative_after_placement():
    state = fresh_state(100.0)
    place(state, "M1", "H", 100.0, 2.0)
    assert state.bankroll == 0.0
    assert state.bankroll >= 0


def test_initial_snapshot():
    state = fresh_state()
    snap = state.snapshot()
    assert snap["bankroll"] == 1000.0
    assert snap["pnl"] == 0.0
    assert snap["open_positions"] == 0


# ---------------------------------------------------------------------------
# State: clock strictly monotonic
# ---------------------------------------------------------------------------

def test_clock_advance():
    state = fresh_state()
    state.advance_day(date(2024, 8, 24))
    assert state.current_date == date(2024, 8, 24)
    assert len(state.day_history) == 2


def test_clock_rejects_backwards():
    state = fresh_state()
    state.advance_day(date(2024, 8, 24))
    with pytest.raises(ValueError, match="strictly monotonic"):
        state.advance_day(date(2024, 8, 20))


def test_clock_rejects_same_day():
    state = fresh_state()
    with pytest.raises(ValueError, match="strictly monotonic"):
        state.advance_day(state.current_date)


# ---------------------------------------------------------------------------
# State: PnL ledger consistency
# ---------------------------------------------------------------------------

def test_pnl_equals_balance_delta_winner():
    state = fresh_state(1000.0)
    place(state, "M1", "H", 50.0, 2.5)
    settle_match(state, "M1", "H", {"H": 2.3, "D": 3.5, "A": 3.2})
    assert abs(state.total_pnl() - (state.bankroll - state.initial_bankroll)) < 0.01


def test_pnl_equals_balance_delta_loser():
    state = fresh_state(1000.0)
    place(state, "M1", "H", 50.0, 2.5)
    settle_match(state, "M1", "A", {"H": 2.3, "D": 3.5, "A": 3.2})
    assert abs(state.total_pnl() - (state.bankroll - state.initial_bankroll)) < 0.01


def test_pnl_multiple_bets():
    state = fresh_state(1000.0)
    place(state, "M1", "H", 50.0, 2.5)
    place(state, "M2", "D", 30.0, 3.0)
    settle_match(state, "M1", "H", {"H": 2.3, "D": 3.5, "A": 3.2})
    settle_match(state, "M2", "A", {"H": 1.8, "D": 3.4, "A": 4.5})
    assert abs(state.total_pnl() - (state.bankroll - state.initial_bankroll)) < 0.01


# ---------------------------------------------------------------------------
# Limits: rejection is typed, not clamped
# ---------------------------------------------------------------------------

def test_rejects_non_positive_stake():
    state = fresh_state()
    r = check_limits(state, "M1", "H", 0.0, 2.0)
    assert r.ok is False
    assert r.reason == "non_positive_stake"


def test_rejects_negative_stake():
    state = fresh_state()
    r = check_limits(state, "M1", "H", -10.0, 2.0)
    assert r.ok is False
    assert r.reason == "non_positive_stake"


def test_rejects_invalid_selection():
    state = fresh_state()
    r = check_limits(state, "M1", "X", 10.0, 2.0)
    assert r.ok is False
    assert r.reason == "invalid_selection"


def test_rejects_stake_exceeding_cap():
    state = fresh_state(1000.0)
    config = LimitConfig(max_stake_fraction=0.10)
    r = check_limits(state, "M1", "H", 150.0, 2.0, config)
    assert r.ok is False
    assert r.reason == "stake_exceeds_cap"


def test_rejects_insufficient_bankroll():
    state = fresh_state(50.0)
    config = LimitConfig(max_stake_fraction=1.0, max_match_exposure_fraction=1.0,
                         drawdown_halt_fraction=1.0)
    r = check_limits(state, "M1", "H", 60.0, 2.0, config)
    assert r.ok is False
    assert r.reason == "insufficient_bankroll"


def test_rejects_match_exposure_exceeded():
    state = fresh_state(1000.0)
    config = LimitConfig(max_match_exposure_fraction=0.05)
    place(state, "M1", "H", 40.0, 2.0)
    r = check_limits(state, "M1", "D", 20.0, 3.0, config)
    assert r.ok is False
    assert r.reason == "match_exposure_exceeded"


def test_rejects_drawdown_halt():
    state = fresh_state(1000.0)
    config = LimitConfig(drawdown_halt_fraction=0.10, max_stake_fraction=1.0)
    place(state, "M1", "H", 80.0, 2.0)
    r = check_limits(state, "M2", "A", 30.0, 2.0, config)
    assert r.ok is False
    assert r.reason == "drawdown_halt"


def test_rejects_odds_out_of_range_low():
    state = fresh_state()
    r = check_limits(state, "M1", "H", 10.0, 0.5)
    assert r.ok is False
    assert r.reason == "odds_out_of_range"


def test_rejects_odds_out_of_range_high():
    state = fresh_state()
    r = check_limits(state, "M1", "H", 10.0, 200.0)
    assert r.ok is False
    assert r.reason == "odds_out_of_range"


def test_accepts_valid_bet():
    state = fresh_state(1000.0)
    r = check_limits(state, "M1", "H", 50.0, 2.0)
    assert r.ok is True


# ---------------------------------------------------------------------------
# Limits: rejection never mutates state
# ---------------------------------------------------------------------------

def test_rejection_does_not_mutate():
    state = fresh_state(1000.0)
    bankroll_before = state.bankroll
    positions_before = len(state.positions)

    check_limits(state, "M1", "H", 500.0, 2.0)  # exceeds cap

    assert state.bankroll == bankroll_before
    assert len(state.positions) == positions_before


# ---------------------------------------------------------------------------
# CLV computation
# ---------------------------------------------------------------------------

def test_clv_positive_when_beating_close():
    clv = compute_clv(odds_taken=2.5, closing_odds=2.0)
    assert clv == pytest.approx(0.25)


def test_clv_negative_when_losing_to_close():
    clv = compute_clv(odds_taken=1.8, closing_odds=2.0)
    assert clv == pytest.approx(-0.10)


def test_clv_zero_when_matching_close():
    clv = compute_clv(odds_taken=2.0, closing_odds=2.0)
    assert clv == pytest.approx(0.0)


def test_clv_handles_invalid_closing():
    clv = compute_clv(odds_taken=2.0, closing_odds=0.0)
    assert clv == 0.0


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

def test_settle_winning_bet():
    state = fresh_state(1000.0)
    place(state, "M1", "H", 50.0, 2.5)
    results = settle_match(state, "M1", "H", {"H": 2.3, "D": 3.5, "A": 3.2})
    assert len(results) == 1
    assert results[0].won is True
    assert results[0].pnl == pytest.approx(75.0)  # 50 * (2.5 - 1)


def test_settle_losing_bet():
    state = fresh_state(1000.0)
    place(state, "M1", "H", 50.0, 2.5)
    results = settle_match(state, "M1", "A", {"H": 2.3, "D": 3.5, "A": 3.2})
    assert len(results) == 1
    assert results[0].won is False
    assert results[0].pnl == pytest.approx(-50.0)


def test_settle_does_not_double_settle():
    state = fresh_state(1000.0)
    place(state, "M1", "H", 50.0, 2.5)
    settle_match(state, "M1", "H", {"H": 2.3, "D": 3.5, "A": 3.2})
    bankroll_after_first = state.bankroll
    results2 = settle_match(state, "M1", "H", {"H": 2.3, "D": 3.5, "A": 3.2})
    assert len(results2) == 0
    assert state.bankroll == bankroll_after_first


def test_settle_records_clv():
    state = fresh_state(1000.0)
    place(state, "M1", "H", 50.0, 2.5)
    results = settle_match(state, "M1", "H", {"H": 2.0, "D": 3.5, "A": 3.2})
    assert results[0].clv == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

def test_exposure_on_match():
    state = fresh_state(1000.0)
    place(state, "M1", "H", 30.0, 2.0)
    place(state, "M1", "D", 20.0, 3.0)
    place(state, "M2", "A", 50.0, 2.5)
    assert state.exposure_on_match("M1") == 50.0
    assert state.exposure_on_match("M2") == 50.0
    assert state.exposure_on_match("M3") == 0.0


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

def test_drawdown_zero_no_bets():
    state = fresh_state()
    assert state.max_drawdown() == 0.0


def test_drawdown_after_loss():
    state = fresh_state(1000.0)
    place(state, "M1", "H", 100.0, 2.0)
    settle_match(state, "M1", "A", {"H": 2.0, "D": 3.0, "A": 3.0})
    assert state.max_drawdown() == 100.0

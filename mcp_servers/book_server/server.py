"""MCP Server 5 — Stateful Betting Book.

The first four servers are read-only.  This one writes: an agent manages a
bankroll across historical fixtures, placing bets through risk-limited tools
and being scored on closing-line value rather than profit.

Tools:
  get_bankroll()                    current state snapshot
  get_available_markets(match_id)   1X2 odds for a match on the current date
  place_bet(match_id, selection, stake, odds)
                                    place a bet — rejects, never clamps
  close_day()                       settle today's matches, advance clock
  get_ledger()                      full bet history with CLV

Design constraints (per plan.md):
  - Rejections are typed ``ok=false`` + reason enum, never silent clamps.
  - Bankroll is never negative.
  - sum(bet_ledger deltas) == balance delta, always.
  - Rejections never mutate state.
  - Episode clock strictly monotonic; no tool returns data after the clock.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_servers.book_server.ledger import ledger_to_dicts, settle_match
from mcp_servers.book_server.limits import LimitConfig, check_limits
from mcp_servers.book_server.state import BookState, Position
from mcp_servers.common import run_server, with_as_of

server = FastMCP("book")

_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _ROOT / "data" / "raw" / "football_data_uk"

_state: BookState | None = None
_config = LimitConfig()
_fixtures: list[dict[str, Any]] = []


def _load_fixtures(season_files: list[str] | None = None) -> list[dict[str, Any]]:
    """Load EPL fixtures from cached CSVs, sorted chronologically."""
    files = season_files or sorted(str(p) for p in _DATA_DIR.glob("E0_*.csv"))
    rows: list[dict[str, Any]] = []
    for fpath in files:
        with open(fpath, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_str = row.get("Date", "")
                try:
                    d = datetime.strptime(date_str, "%d/%m/%Y").date()
                except (ValueError, TypeError):
                    continue
                rows.append({
                    "date": d,
                    "home": row.get("HomeTeam", ""),
                    "away": row.get("AwayTeam", ""),
                    "match_id": f"{row.get('HomeTeam', '')}-{row.get('AwayTeam', '')}-{d.isoformat()}",
                    "ftr": row.get("FTR", ""),  # H/D/A
                    "odds_h": _float(row.get("PSH") or row.get("B365H")),
                    "odds_d": _float(row.get("PSD") or row.get("B365D")),
                    "odds_a": _float(row.get("PSA") or row.get("B365A")),
                    "close_h": _float(row.get("PSCH") or row.get("B365CH")),
                    "close_d": _float(row.get("PSCD") or row.get("B365CD")),
                    "close_a": _float(row.get("PSCA") or row.get("B365CA")),
                    "odds_source": "pinnacle" if row.get("PSCH") else "bet365",
                })
    rows.sort(key=lambda r: r["date"])
    return rows


def _float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _ensure_state() -> BookState:
    global _state, _fixtures
    if _state is None:
        _fixtures = _load_fixtures()
        first_date = _fixtures[0]["date"] if _fixtures else date(2024, 8, 16)
        _state = BookState.create(bankroll=1000.0, start_date=first_date)
    return _state


def _matches_on_date(d: date) -> list[dict[str, Any]]:
    return [f for f in _fixtures if f["date"] == d]


def _gameweek_dates(from_date: date) -> list[date]:
    """Return the next set of unique match dates (a gameweek)."""
    dates = sorted({f["date"] for f in _fixtures if f["date"] >= from_date})
    gw_dates: list[date] = []
    for d in dates:
        gw_dates.append(d)
        if len(gw_dates) >= 3:
            break
    return gw_dates


@server.tool()
def get_bankroll() -> dict[str, Any]:
    """Current bankroll state: balance, P&L, open/settled positions, drawdown,
    episode date, and risk limit configuration."""
    state = _ensure_state()
    snap = state.snapshot()
    snap["limits"] = _config.as_dict
    return with_as_of(snap)


@server.tool()
def get_available_markets(match_id: str = "") -> dict[str, Any]:
    """Available 1X2 markets on or after the current episode date.

    If ``match_id`` is provided, returns odds for that specific match.
    Otherwise returns all matches on the current date."""
    state = _ensure_state()
    today_matches = _matches_on_date(state.current_date)

    if match_id:
        for m in today_matches:
            if m["match_id"] == match_id:
                return with_as_of({
                    "ok": True,
                    "match_id": m["match_id"],
                    "home": m["home"],
                    "away": m["away"],
                    "date": m["date"].isoformat(),
                    "markets": {
                        "1X2": {"H": m["odds_h"], "D": m["odds_d"], "A": m["odds_a"]},
                    },
                })
        return with_as_of({
            "ok": False,
            "reason": "match_not_found",
            "detail": f"no match '{match_id}' on {state.current_date}",
        })

    return with_as_of({
        "ok": True,
        "date": state.current_date.isoformat(),
        "matches": [
            {
                "match_id": m["match_id"],
                "home": m["home"],
                "away": m["away"],
                "markets": {
                    "1X2": {"H": m["odds_h"], "D": m["odds_d"], "A": m["odds_a"]},
                },
            }
            for m in today_matches
        ],
    })


@server.tool()
def place_bet(
    match_id: str,
    selection: str,
    stake: float,
    odds: float,
) -> dict[str, Any]:
    """Place a bet on a match. Rejects with a typed reason if any risk limit
    is breached — never clamps or silently adjusts.

    Args:
        match_id: the match to bet on (from get_available_markets).
        selection: "H" (home), "D" (draw), or "A" (away).
        stake: amount to wager (deducted from bankroll immediately).
        odds: decimal odds the agent is taking.
    """
    state = _ensure_state()

    today_matches = _matches_on_date(state.current_date)
    match = None
    for m in today_matches:
        if m["match_id"] == match_id:
            match = m
            break
    if match is None:
        return with_as_of({
            "ok": False,
            "reason": "match_not_found",
            "detail": f"no match '{match_id}' on {state.current_date}",
        })

    check = check_limits(state, match_id, selection, stake, odds, _config)
    if not check.ok:
        return with_as_of(check.to_dict())

    bet_id = state.next_bet_id()
    pos = Position(
        bet_id=bet_id,
        match_id=match_id,
        market="1X2",
        selection=selection,
        stake=stake,
        odds_taken=odds,
        placed_on=state.current_date,
    )
    state.positions.append(pos)
    state.bankroll -= stake

    return with_as_of({
        "ok": True,
        "bet_id": bet_id,
        "match_id": match_id,
        "selection": selection,
        "stake": round(stake, 4),
        "odds": odds,
        "bankroll_after": round(state.bankroll, 2),
    })


@server.tool()
def close_day() -> dict[str, Any]:
    """Settle all matches on the current date, then advance the clock to the
    next match day.  Returns settlement results with CLV for each bet."""
    state = _ensure_state()
    today_matches = _matches_on_date(state.current_date)

    settlements: list[dict[str, Any]] = []
    for m in today_matches:
        closing = {"H": m["close_h"], "D": m["close_d"], "A": m["close_a"]}
        results = settle_match(state, m["match_id"], m["ftr"], closing)
        for r in results:
            settlements.append({
                "bet_id": r.bet_id,
                "match_id": m["match_id"],
                "result": m["ftr"],
                "won": r.won,
                "pnl": r.pnl,
                "clv": r.clv,
                "odds_taken": r.odds_taken,
                "closing_odds": r.closing_odds,
            })

    future_dates = sorted({f["date"] for f in _fixtures if f["date"] > state.current_date})
    if future_dates:
        state.advance_day(future_dates[0])
        done = False
    else:
        done = True

    return with_as_of({
        "ok": True,
        "settled": settlements,
        "n_settled": len(settlements),
        "bankroll_after": round(state.bankroll, 2),
        "next_date": state.current_date.isoformat() if not done else None,
        "episode_done": done,
    })


@server.tool()
def get_ledger() -> dict[str, Any]:
    """Full bet history with settlement status, P&L, and CLV per bet."""
    state = _ensure_state()
    bets = ledger_to_dicts(state)
    summary = {
        "total_bets": len(bets),
        "settled": sum(1 for b in bets if b["settled"]),
        "open": sum(1 for b in bets if not b["settled"]),
        "total_pnl": round(sum(b["pnl"] for b in bets), 4),
        "mean_clv": round(
            sum(b["clv"] for b in bets if b["settled"])
            / max(1, sum(1 for b in bets if b["settled"])),
            4,
        ),
    }
    return with_as_of({"ok": True, "summary": summary, "bets": bets})


@server.tool()
def reset_episode(bankroll: float = 1000.0) -> dict[str, Any]:
    """Reset the book to a fresh episode. Used between gameweek runs."""
    global _state
    first_date = _fixtures[0]["date"] if _fixtures else date(2024, 8, 16)
    _state = BookState.create(bankroll=bankroll, start_date=first_date)
    return with_as_of({"ok": True, "bankroll": bankroll, "start_date": first_date.isoformat()})


if __name__ == "__main__":
    run_server(server)

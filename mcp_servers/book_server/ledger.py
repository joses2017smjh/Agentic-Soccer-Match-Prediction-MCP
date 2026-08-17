"""Append-only bet ledger, settlement, and closing-line value computation.

CLV (closing-line value) is the primary reward signal.  A bet has positive
CLV when the odds taken exceed the Pinnacle closing odds — i.e., when the
agent got a price the sharpest book later agreed was too generous.  This is
a better signal than P&L because it factors out variance.

CLV = (odds_taken / closing_odds) - 1

Settlement is separate from CLV: a bet can have positive CLV (good find)
but lose (variance), and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from mcp_servers.book_server.state import BookState, Position

CLOSING_COLS = {
    "H": ("PSCH", "B365CH"),
    "D": ("PSCD", "B365CD"),
    "A": ("PSCA", "B365CA"),
}


@dataclass
class SettlementResult:
    bet_id: str
    won: bool
    pnl: float
    clv: float
    odds_taken: float
    closing_odds: float


def compute_clv(odds_taken: float, closing_odds: float) -> float:
    if closing_odds <= 1.0:
        return 0.0
    return (odds_taken / closing_odds) - 1.0


def settle_match(
    state: BookState,
    match_id: str,
    result: str,
    closing_odds: dict[str, float],
) -> list[SettlementResult]:
    """Settle all open positions on a match.

    Args:
        state: current book state (mutated in place).
        match_id: the match to settle.
        result: "H", "D", or "A".
        closing_odds: {"H": float, "D": float, "A": float} from Pinnacle close.

    Returns:
        list of SettlementResult for each settled position.
    """
    results: list[SettlementResult] = []
    for pos in state.positions:
        if pos.match_id != match_id or pos.settled:
            continue

        won = pos.selection == result
        pnl = pos.stake * (pos.odds_taken - 1.0) if won else -pos.stake
        close = closing_odds.get(pos.selection, pos.odds_taken)
        clv = compute_clv(pos.odds_taken, close)

        pos.settled = True
        pos.pnl = pnl
        pos.clv = clv
        state.bankroll += pos.stake + pnl if won else 0
        # stake was already deducted at placement; losing bet = no return

        results.append(SettlementResult(
            bet_id=pos.bet_id,
            won=won,
            pnl=round(pnl, 4),
            clv=round(clv, 4),
            odds_taken=pos.odds_taken,
            closing_odds=close,
        ))

    return results


def ledger_to_dicts(state: BookState) -> list[dict[str, Any]]:
    """Serialize the full bet ledger for API responses."""
    return [
        {
            "bet_id": p.bet_id,
            "match_id": p.match_id,
            "market": p.market,
            "selection": p.selection,
            "stake": round(p.stake, 4),
            "odds_taken": p.odds_taken,
            "placed_on": p.placed_on.isoformat(),
            "settled": p.settled,
            "pnl": round(p.pnl, 4),
            "clv": round(p.clv, 4),
        }
        for p in state.positions
    ]

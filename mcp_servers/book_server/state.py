"""Book server state: bankroll, open positions, and episode clock.

The episode clock is strictly monotonic — no tool may return data with
``as_of`` newer than the clock, and ``close_day`` is the only way to
advance it.  This prevents stale-price and look-ahead attacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class Position:
    """A single open (unsettled) bet."""

    bet_id: str
    match_id: str
    market: str          # "1X2"
    selection: str       # "H", "D", "A"
    stake: float
    odds_taken: float    # decimal odds at time of bet
    placed_on: date
    settled: bool = False
    pnl: float = 0.0
    clv: float = 0.0


@dataclass
class BookState:
    """Mutable state for one episode.

    Invariants (tested):
    - ``bankroll`` is never negative.
    - ``sum(position.pnl for settled positions) == bankroll - initial_bankroll``
    - Clock is strictly monotonic via ``advance_day``.
    - Rejected operations never mutate any field.
    """

    initial_bankroll: float
    bankroll: float
    current_date: date
    positions: list[Position] = field(default_factory=list)
    day_history: list[date] = field(default_factory=list)
    _next_bet_id: int = field(default=0, repr=False)

    @classmethod
    def create(cls, bankroll: float = 1000.0, start_date: date | None = None) -> BookState:
        d = start_date or date(2024, 8, 16)
        return cls(
            initial_bankroll=bankroll,
            bankroll=bankroll,
            current_date=d,
            day_history=[d],
        )

    def next_bet_id(self) -> str:
        self._next_bet_id += 1
        return f"BET-{self._next_bet_id:04d}"

    def advance_day(self, to: date) -> None:
        if to <= self.current_date:
            raise ValueError(
                f"clock must be strictly monotonic: "
                f"cannot move from {self.current_date} to {to}"
            )
        self.current_date = to
        self.day_history.append(to)

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if not p.settled]

    def settled_positions(self) -> list[Position]:
        return [p for p in self.positions if p.settled]

    def total_pnl(self) -> float:
        return sum(p.pnl for p in self.positions if p.settled)

    def max_drawdown(self) -> float:
        peak = self.initial_bankroll
        max_dd = 0.0
        running = self.initial_bankroll
        for p in self.positions:
            if not p.settled:
                continue
            running += p.pnl
            peak = max(peak, running)
            max_dd = max(max_dd, peak - running)
        return max_dd

    def exposure_on_match(self, match_id: str) -> float:
        return sum(
            p.stake for p in self.positions
            if p.match_id == match_id and not p.settled
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "bankroll": round(self.bankroll, 2),
            "initial_bankroll": round(self.initial_bankroll, 2),
            "pnl": round(self.total_pnl(), 2),
            "current_date": self.current_date.isoformat(),
            "open_positions": len(self.open_positions()),
            "settled_positions": len(self.settled_positions()),
            "max_drawdown": round(self.max_drawdown(), 2),
            "days_elapsed": len(self.day_history),
        }

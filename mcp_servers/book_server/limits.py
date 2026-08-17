"""Risk limits: per-bet cap, per-match exposure cap, drawdown halt.

Every limit **rejects** rather than clamping — typed ``ok=False`` with a
reason enum, matching the repo's convention that failures are data, not
exceptions.  A rejected bet never mutates state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_servers.book_server.state import BookState

VALID_SELECTIONS = {"H", "D", "A"}


@dataclass(frozen=True)
class LimitConfig:
    max_stake_fraction: float = 0.10       # max 10% of bankroll per bet
    max_match_exposure_fraction: float = 0.15  # max 15% of bankroll on one match
    drawdown_halt_fraction: float = 0.30   # halt at 30% drawdown from peak
    min_odds: float = 1.01
    max_odds: float = 100.0

    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "max_stake_fraction": self.max_stake_fraction,
            "max_match_exposure_fraction": self.max_match_exposure_fraction,
            "drawdown_halt_fraction": self.drawdown_halt_fraction,
            "min_odds": self.min_odds,
            "max_odds": self.max_odds,
        }


@dataclass(frozen=True)
class RejectionResult:
    ok: bool
    reason: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "detail": self.detail}


PASS = RejectionResult(ok=True, reason="", detail="")


def check_limits(
    state: BookState,
    match_id: str,
    selection: str,
    stake: float,
    odds: float,
    config: LimitConfig | None = None,
) -> RejectionResult:
    """Validate a proposed bet against all risk limits.

    Returns ``PASS`` if the bet is allowed, or a typed rejection otherwise.
    """
    cfg = config or LimitConfig()

    if selection not in VALID_SELECTIONS:
        return RejectionResult(
            ok=False, reason="invalid_selection",
            detail=f"selection must be one of {sorted(VALID_SELECTIONS)}, got '{selection}'",
        )

    if stake <= 0:
        return RejectionResult(
            ok=False, reason="non_positive_stake",
            detail=f"stake must be positive, got {stake}",
        )

    if odds < cfg.min_odds or odds > cfg.max_odds:
        return RejectionResult(
            ok=False, reason="odds_out_of_range",
            detail=f"odds {odds} outside [{cfg.min_odds}, {cfg.max_odds}]",
        )

    if stake > state.bankroll:
        return RejectionResult(
            ok=False, reason="insufficient_bankroll",
            detail=f"stake {stake:.2f} exceeds available bankroll {state.bankroll:.2f}",
        )

    max_stake = state.bankroll * cfg.max_stake_fraction
    if stake > max_stake:
        return RejectionResult(
            ok=False, reason="stake_exceeds_cap",
            detail=f"stake {stake:.2f} exceeds per-bet cap {max_stake:.2f} "
                   f"({cfg.max_stake_fraction:.0%} of bankroll {state.bankroll:.2f})",
        )

    current_exposure = state.exposure_on_match(match_id)
    max_exposure = state.initial_bankroll * cfg.max_match_exposure_fraction
    if current_exposure + stake > max_exposure:
        return RejectionResult(
            ok=False, reason="match_exposure_exceeded",
            detail=f"total exposure on {match_id} would be {current_exposure + stake:.2f}, "
                   f"exceeds cap {max_exposure:.2f}",
        )

    drawdown = state.initial_bankroll - state.bankroll + sum(
        p.stake for p in state.open_positions()
    )
    halt_threshold = state.initial_bankroll * cfg.drawdown_halt_fraction
    if drawdown + stake > halt_threshold:
        return RejectionResult(
            ok=False, reason="drawdown_halt",
            detail=f"effective drawdown {drawdown + stake:.2f} would exceed "
                   f"halt threshold {halt_threshold:.2f} "
                   f"({cfg.drawdown_halt_fraction:.0%} of initial bankroll)",
        )

    return PASS

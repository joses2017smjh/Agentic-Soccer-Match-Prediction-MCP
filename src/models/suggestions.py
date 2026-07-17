"""Suggestion layer: model vs market, edge/EV, Kelly sizing, rationale.

For each selection we hold the calibrated model probability p, the vig-free
market probability q, and the actual payable decimal odds o (vig included —
that is what a bet settles at):

    edge  = p − q
    EV    = p·(o − 1) − (1 − p)          (per unit staked)
    kelly = max(0, (p·(o − 1) − (1 − p)) / (o − 1)) · kelly_fraction

A selection is flagged when EV clears the configured threshold. Confidence
tiers come from the edge; a flagged h2h selection that falls *outside* the
conformal prediction set is capped at "low" — the uncertainty wrapper is
allowed to veto the tier but not to invent bets.

The rationale is assembled from typed Driver records (availability shocks,
form gaps, market drift...) supplied by the pipeline — never from free text —
so Phase B's agent can quote it verbatim without an injection surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Driver:
    """One explanatory factor, largest |impact| first in the rationale."""

    name: str          # e.g. "availability", "form_xg", "odds_drift"
    impact: float      # signed contribution toward the model's view
    text: str          # human sentence fragment, produced by our code only


@dataclass(frozen=True)
class MarketQuote:
    market: str        # "h2h", "totals_2.5", "btts", "anytime_scorer:Saka"...
    selection: str     # "home", "over", "yes", ...
    model_prob: float
    market_prob: float  # vig-free implied probability
    decimal_odds: float  # payable odds, vig included


@dataclass(frozen=True)
class Suggestion:
    market: str
    selection: str
    model_prob: float
    market_prob: float
    decimal_odds: float
    edge: float
    ev: float
    kelly_stake: float
    flagged: bool
    tier: str          # "high" | "medium" | "low" | "none"
    rationale: str


def _tier(edge: float, tiers: dict[str, float]) -> str:
    for name in ("high", "medium", "low"):
        if edge >= tiers[name]:
            return name
    return "none"


def _rationale(
    quote: MarketQuote, edge: float, drivers: list[Driver], capped: bool
) -> str:
    parts = [
        f"Model {quote.model_prob:.1%} vs market {quote.market_prob:.1%} "
        f"({edge:+.1%} edge) on {quote.market}/{quote.selection}."
    ]
    for d in sorted(drivers, key=lambda d: -abs(d.impact))[:2]:
        parts.append(d.text)
    if capped:
        parts.append(
            "Confidence capped: this outcome sits outside the model's "
            "conformal prediction set at the configured coverage level."
        )
    return " ".join(parts)


def make_suggestions(
    quotes: list[MarketQuote],
    *,
    ev_threshold: float = 0.03,
    kelly_fraction: float = 0.25,
    tiers: dict[str, float] | None = None,
    h2h_conformal_set: list[str] | None = None,
    drivers: dict[str, list[Driver]] | None = None,
) -> list[Suggestion]:
    """Evaluate every quote; flagged suggestions sort first by EV.

    ``drivers`` maps "market/selection" (or "market" for all selections of a
    market) to the Driver list used for that rationale.
    """
    tiers = tiers or {"high": 0.08, "medium": 0.05, "low": 0.03}
    drivers = drivers or {}
    out: list[Suggestion] = []

    for q in quotes:
        edge = q.model_prob - q.market_prob
        b = q.decimal_odds - 1.0
        ev = q.model_prob * b - (1.0 - q.model_prob)
        kelly = max(0.0, (q.model_prob * b - (1.0 - q.model_prob)) / b) if b > 0 else 0.0

        flagged = ev > ev_threshold
        tier = _tier(edge, tiers) if flagged else "none"
        capped = False
        if (
            flagged
            and q.market == "h2h"
            and h2h_conformal_set is not None
            and q.selection not in h2h_conformal_set
        ):
            tier, capped = "low", True

        quote_drivers = drivers.get(f"{q.market}/{q.selection}", drivers.get(q.market, []))
        out.append(Suggestion(
            market=q.market, selection=q.selection,
            model_prob=q.model_prob, market_prob=q.market_prob,
            decimal_odds=q.decimal_odds, edge=edge, ev=ev,
            kelly_stake=kelly * kelly_fraction if flagged else 0.0,
            flagged=flagged, tier=tier,
            rationale=_rationale(q, edge, quote_drivers, capped) if flagged else "",
        ))

    return sorted(out, key=lambda s: (not s.flagged, -s.ev))

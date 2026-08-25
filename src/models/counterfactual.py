"""Counterfactual lab (C2).

Toggle hypothetical changes to a match context and show the delta at
every layer of the prediction pipeline. Supported perturbations:

  - Remove a player from the lineup (injury/suspension)
  - Change rest days for a team
  - Flip the venue to neutral
  - Adjust pre-match expected goals (what-if xG scenarios)

Each perturbation re-derives the scoreline grid and outcome probs,
returning a structured diff against the baseline.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from src.models.score_grid import outcome_probs, score_grid


@dataclass
class Perturbation:
    """A single hypothetical change to apply."""

    kind: str  # "remove_player", "rest_days", "neutral_venue", "adjust_xg"
    params: dict = field(default_factory=dict)


@dataclass
class CounterfactualResult:
    """Diff between baseline and perturbed prediction."""

    perturbation: Perturbation
    baseline_probs: dict[str, float]
    perturbed_probs: dict[str, float]
    deltas: dict[str, float]
    baseline_mu: tuple[float, float]
    perturbed_mu: tuple[float, float]


def _apply_remove_player(
    mu_home: float,
    mu_away: float,
    params: dict,
) -> tuple[float, float]:
    """Reduce a team's expected goals when a key player is removed.

    Uses a configurable xg_contribution (default 0.15 = 15% of team xG).
    """
    side = params.get("side", "home")
    contribution = params.get("xg_contribution", 0.15)
    if side == "home":
        return max(mu_home * (1.0 - contribution), 0.1), mu_away
    return mu_home, max(mu_away * (1.0 - contribution), 0.1)


def _apply_rest_days(
    mu_home: float,
    mu_away: float,
    params: dict,
) -> tuple[float, float]:
    """Adjust expected goals based on rest day changes.

    Fatigue factor: each day below 4 rest days reduces xG by 3%.
    Recovery bonus: each day above 5 adds 1.5% up to a cap.
    """
    side = params.get("side", "home")
    days = params.get("days", 4)
    baseline_days = params.get("baseline_days", 4)

    diff = days - baseline_days
    if diff < 0:
        factor = 1.0 + diff * 0.03  # fewer rest = penalty
    else:
        factor = 1.0 + min(diff, 3) * 0.015  # more rest = small bonus

    factor = max(factor, 0.7)
    if side == "home":
        return max(mu_home * factor, 0.1), mu_away
    return mu_home, max(mu_away * factor, 0.1)


def _apply_neutral_venue(
    mu_home: float,
    mu_away: float,
    params: dict,
) -> tuple[float, float]:
    """Remove home advantage by redistributing expected goals.

    Home advantage in the Poisson means is roughly +0.2 for home, -0.1 for
    away. On neutral ground, split the difference.
    """
    home_adv = params.get("home_advantage", 0.2)
    return max(mu_home - home_adv / 2, 0.1), mu_away + home_adv / 2


def _apply_adjust_xg(
    mu_home: float,
    mu_away: float,
    params: dict,
) -> tuple[float, float]:
    """Direct xG adjustment for what-if scenarios."""
    delta_home = params.get("delta_home", 0.0)
    delta_away = params.get("delta_away", 0.0)
    return max(mu_home + delta_home, 0.1), max(mu_away + delta_away, 0.1)


_APPLIERS = {
    "remove_player": _apply_remove_player,
    "rest_days": _apply_rest_days,
    "neutral_venue": _apply_neutral_venue,
    "adjust_xg": _apply_adjust_xg,
}


def run_counterfactual(
    mu_home: float,
    mu_away: float,
    perturbation: Perturbation,
    rho: float = -0.1,
) -> CounterfactualResult:
    """Apply a single perturbation and return the delta."""
    applier = _APPLIERS.get(perturbation.kind)
    if applier is None:
        raise ValueError(f"Unknown perturbation kind: {perturbation.kind}")

    baseline_grid = score_grid(mu_home, mu_away, rho)
    baseline_probs = outcome_probs(baseline_grid)

    p_mu_h, p_mu_a = applier(mu_home, mu_away, perturbation.params)
    perturbed_grid = score_grid(p_mu_h, p_mu_a, rho)
    perturbed_probs = outcome_probs(perturbed_grid)

    deltas = {
        k: perturbed_probs[k] - baseline_probs[k]
        for k in ("home", "draw", "away")
    }

    return CounterfactualResult(
        perturbation=perturbation,
        baseline_probs=baseline_probs,
        perturbed_probs=perturbed_probs,
        deltas=deltas,
        baseline_mu=(mu_home, mu_away),
        perturbed_mu=(p_mu_h, p_mu_a),
    )


def run_counterfactual_batch(
    mu_home: float,
    mu_away: float,
    perturbations: list[Perturbation],
    rho: float = -0.1,
) -> list[CounterfactualResult]:
    """Run multiple perturbations against the same baseline."""
    return [
        run_counterfactual(mu_home, mu_away, p, rho)
        for p in perturbations
    ]


def format_counterfactual_table(
    results: list[CounterfactualResult],
) -> list[dict]:
    """Format results as a list of dicts for display."""
    rows = []
    for r in results:
        rows.append({
            "perturbation": r.perturbation.kind,
            "params": r.perturbation.params,
            "baseline": {k: round(v, 4) for k, v in r.baseline_probs.items()},
            "perturbed": {k: round(v, 4) for k, v in r.perturbed_probs.items()},
            "delta": {k: round(v, 4) for k, v in r.deltas.items()},
            "mu_change": {
                "home": round(r.perturbed_mu[0] - r.baseline_mu[0], 4),
                "away": round(r.perturbed_mu[1] - r.baseline_mu[1], 4),
            },
        })
    return rows

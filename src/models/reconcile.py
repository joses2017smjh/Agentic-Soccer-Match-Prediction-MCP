"""Grid-head reconciliation (B4a).

Derives match outcome probabilities from the Dixon-Coles scoreline grid
and cross-checks against the direct XGBoost head. When the two agree
within epsilon, use the grid-derived probs (they're more coherent with
the scoreline and O/U markets). When they disagree, log the discrepancy
and blend them.
"""

from __future__ import annotations

import numpy as np

from src.models.score_grid import outcome_probs, score_grid


def grid_outcome(
    mu_home: float, mu_away: float, rho: float = -0.1,
) -> dict[str, float]:
    """Derive 1X2 probabilities directly from the Dixon-Coles grid."""
    grid = score_grid(mu_home, mu_away, rho)
    return outcome_probs(grid)


def reconcile(
    head_probs: dict[str, float],
    mu_home: float,
    mu_away: float,
    rho: float = -0.1,
    epsilon: float = 0.05,
    blend_weight: float = 0.5,
) -> dict[str, float | list]:
    """Cross-check the direct head against the grid-derived outcome.

    Returns:
        reconciled: the final probabilities
        grid_probs: grid-derived probabilities
        head_probs: the direct head's probabilities
        max_diff: largest absolute difference between head and grid
        reconciled_via: "grid" if within epsilon, "blend" otherwise
        discrepancies: list of (outcome, diff) for any diff > epsilon
    """
    grid_p = grid_outcome(mu_home, mu_away, rho)

    diffs = {k: abs(head_probs.get(k, 0) - grid_p.get(k, 0))
             for k in ("home", "draw", "away")}
    max_diff = max(diffs.values())
    discrepancies = [(k, d) for k, d in diffs.items() if d > epsilon]

    if max_diff <= epsilon:
        reconciled = dict(grid_p)
        method = "grid"
    else:
        # blend: weighted average favoring the grid for coherence
        reconciled = {}
        for k in ("home", "draw", "away"):
            reconciled[k] = (
                blend_weight * grid_p.get(k, 0)
                + (1 - blend_weight) * head_probs.get(k, 0)
            )
        total = sum(reconciled.values())
        if total > 0:
            reconciled = {k: v / total for k, v in reconciled.items()}
        method = "blend"

    return {
        "reconciled": reconciled,
        "grid_probs": grid_p,
        "head_probs": head_probs,
        "max_diff": max_diff,
        "reconciled_via": method,
        "discrepancies": discrepancies,
    }

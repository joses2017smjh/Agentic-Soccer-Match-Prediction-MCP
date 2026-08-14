"""Correlated-parlay pricer: joint probabilities from the Dixon-Coles grid.

Standard sportsbooks price parlays by multiplying independent leg odds -- this
assumes statistical independence between legs.  But legs within the same match
(home win, over 2.5, BTTS yes, player anytime scorer) are correlated through
the joint (home_goals, away_goals) distribution that the Dixon-Coles grid
already encodes.

This module prices same-match parlays by intersecting the cell subsets each leg
selects in the score grid, producing the exact joint probability.  The ratio
joint / product-of-marginals is the **correlation factor** -- typically >1 for
"home win + over 2.5" (winning means more goals) and <1 for "home win + under
2.5" (domination usually means *some* scoring), because a winning team's goals
contribute to both legs.

Multi-match parlays combine within-match joints across matches -- the
cross-match independence assumption holds in all standard models.

Study output: each priced parlay carries both the joint and independent prices,
making the correlation adjustment a directly measurable, auditable quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models.score_grid import score_grid as build_score_grid


# ---------------------------------------------------------------------------
# Leg specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParlayLeg:
    """One leg of a parlay bet."""

    match_id: str
    leg_type: str       # outcome | over | under | btts | exact_score | anytime_scorer
    selection: str       # home/draw/away | yes/no | "2-1" | player name
    line: float = 0.0   # for over/under (e.g. 2.5)
    decimal_odds: float = 0.0  # sportsbook price (0 = not provided)
    # player-prop fields (only for anytime_scorer)
    team_side: str = ""  # "home" | "away"
    xg_share: float = 0.0


# ---------------------------------------------------------------------------
# Grid-cell masks for non-player legs
# ---------------------------------------------------------------------------

def _leg_mask(leg: ParlayLeg, n: int) -> np.ndarray:
    """Boolean (n, n) mask: True where this leg is satisfied at scoreline (i, j)."""
    goals = np.arange(n)
    hx, ay = np.meshgrid(goals, goals, indexing="ij")

    if leg.leg_type == "outcome":
        if leg.selection == "home":
            return hx > ay
        if leg.selection == "draw":
            return hx == ay
        if leg.selection == "away":
            return ay > hx
        raise ValueError(f"unknown outcome selection: {leg.selection}")

    if leg.leg_type == "over":
        return (hx + ay) > leg.line

    if leg.leg_type == "under":
        return (hx + ay) < leg.line

    if leg.leg_type == "btts":
        if leg.selection in ("yes", "Yes"):
            return (hx >= 1) & (ay >= 1)
        return (hx == 0) | (ay == 0)

    if leg.leg_type == "exact_score":
        parts = leg.selection.split("-")
        h, a = int(parts[0]), int(parts[1])
        return (hx == h) & (ay == a)

    if leg.leg_type == "home_over":
        return hx > leg.line

    if leg.leg_type == "home_under":
        return hx < leg.line

    if leg.leg_type == "away_over":
        return ay > leg.line

    if leg.leg_type == "away_under":
        return ay < leg.line

    if leg.leg_type == "anytime_scorer":
        # handled separately in the probability calculation
        return np.ones((n, n), dtype=bool)

    raise ValueError(f"unknown leg_type: {leg.leg_type}")


def _player_prob_given_goals(share: float, team_goals: int) -> float:
    """P(player scores >= 1 | their team scores exactly ``team_goals``).

    Multinomial model: each of the k goals is independently allocated to
    players by share.  P(at least one to this player) = 1 - (1-s)^k.
    """
    if team_goals == 0 or share <= 0.0:
        return 0.0
    return 1.0 - (1.0 - min(share, 1.0)) ** team_goals


# ---------------------------------------------------------------------------
# Single-match joint probability
# ---------------------------------------------------------------------------

def _within_match_joint(
    legs: list[ParlayLeg],
    grid: np.ndarray,
) -> float:
    """Joint probability of all legs being satisfied simultaneously.

    Non-player legs use grid-cell intersection (exact).  Player-scorer legs
    condition on the team's goal count at each cell (multinomial model, Poisson
    independent approximation for multiple scorers from the same team).
    """
    n = grid.shape[0]

    # start with all cells active
    mask = np.ones((n, n), dtype=bool)
    scorer_legs: list[ParlayLeg] = []

    for leg in legs:
        if leg.leg_type == "anytime_scorer":
            scorer_legs.append(leg)
        else:
            mask &= _leg_mask(leg, n)

    if not scorer_legs:
        return float((grid * mask).sum())

    # for scorer legs, weight each surviving cell by P(all scorers | goals)
    goals_range = np.arange(n)
    hx, ay = np.meshgrid(goals_range, goals_range, indexing="ij")

    prob = 0.0
    for i in range(n):
        for j in range(n):
            if not mask[i, j] or grid[i, j] < 1e-15:
                continue
            cell_prob = grid[i, j]
            for sl in scorer_legs:
                team_goals = i if sl.team_side == "home" else j
                cell_prob *= _player_prob_given_goals(sl.xg_share, team_goals)
            prob += cell_prob

    return prob


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass
class LegResult:
    """Per-leg detail in the priced parlay."""

    match_id: str
    leg_type: str
    selection: str
    marginal_prob: float       # individual probability of this leg
    decimal_odds: float        # sportsbook price (0 if not provided)
    implied_prob: float        # 1/decimal_odds (0 if not provided)


@dataclass
class MatchGroup:
    """Within-match correlation analysis."""

    match_id: str
    legs: list[LegResult]
    joint_prob: float          # exact joint from the grid
    independent_prob: float    # product of individual marginals
    correlation_factor: float  # joint / independent (>1 = positive corr.)
    mu_home: float
    mu_away: float


@dataclass
class ParlayResult:
    """Complete parlay pricing with correlation analysis."""

    legs: list[LegResult]
    match_groups: list[MatchGroup]
    # overall parlay
    joint_probability: float
    independent_probability: float
    correlation_factor: float
    model_fair_odds: float         # 1 / joint_probability
    sportsbook_parlay_odds: float  # product of individual decimal odds (0 if any missing)
    edge: float                    # joint_prob - implied_prob_of_sportsbook_odds
    ev: float                      # joint_prob * (sb_odds - 1) - (1 - joint_prob)
    kelly_fraction: float
    correlation_direction: str     # "positive" | "negative" | "neutral" | "mixed"
    analysis: str


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def price_parlay(
    legs: list[ParlayLeg],
    mus: dict[str, tuple[float, float]],   # match_id -> (mu_home, mu_away)
    rho: float = -0.1,
    max_goals: int = 10,
    kelly_frac: float = 0.25,
) -> ParlayResult:
    """Price a multi-leg parlay using the Dixon-Coles grid.

    Parameters
    ----------
    legs : list[ParlayLeg]
        The parlay legs.  Legs from the same match_id are correlated through
        the grid; cross-match legs are independent.
    mus : dict[str, tuple[float, float]]
        Expected goals per match: match_id -> (mu_home, mu_away).
    rho : float
        Dixon-Coles low-score correction parameter.
    max_goals : int
        Grid truncation.
    kelly_frac : float
        Kelly fraction for sizing.

    Returns
    -------
    ParlayResult
        Full pricing with per-leg marginals, within-match correlation, and
        overall parlay metrics.
    """
    if not legs:
        raise ValueError("parlay must have at least one leg")

    # build grids per match
    grids: dict[str, np.ndarray] = {}
    for match_id, (mu_h, mu_a) in mus.items():
        grids[match_id] = build_score_grid(mu_h, mu_a, rho, max_goals)

    # group legs by match
    match_legs: dict[str, list[ParlayLeg]] = {}
    for leg in legs:
        match_legs.setdefault(leg.match_id, []).append(leg)

    # compute per-leg marginal probabilities
    leg_results: list[LegResult] = []
    for leg in legs:
        grid = grids[leg.match_id]
        n = grid.shape[0]
        if leg.leg_type == "anytime_scorer":
            mu_team = mus[leg.match_id][0 if leg.team_side == "home" else 1]
            # marginal = sum over cells of grid[i,j] * P(scorer | goals)
            marginal = 0.0
            for i in range(n):
                for j in range(n):
                    tg = i if leg.team_side == "home" else j
                    marginal += grid[i, j] * _player_prob_given_goals(leg.xg_share, tg)
        else:
            mask = _leg_mask(leg, n)
            marginal = float((grid * mask).sum())

        implied = 1.0 / leg.decimal_odds if leg.decimal_odds > 1.0 else 0.0
        leg_results.append(LegResult(
            match_id=leg.match_id,
            leg_type=leg.leg_type,
            selection=leg.selection,
            marginal_prob=marginal,
            decimal_odds=leg.decimal_odds,
            implied_prob=implied,
        ))

    # within-match joint probabilities
    match_groups: list[MatchGroup] = []
    overall_joint = 1.0

    for match_id, m_legs in match_legs.items():
        grid = grids[match_id]
        joint = _within_match_joint(m_legs, grid)

        match_lr = [lr for lr in leg_results if lr.match_id == match_id]
        indep = math.prod(lr.marginal_prob for lr in match_lr) if match_lr else 0.0
        corr = joint / indep if indep > 1e-15 else 1.0

        mu_h, mu_a = mus[match_id]
        match_groups.append(MatchGroup(
            match_id=match_id,
            legs=match_lr,
            joint_prob=joint,
            independent_prob=indep,
            correlation_factor=corr,
            mu_home=mu_h,
            mu_away=mu_a,
        ))
        overall_joint *= joint

    overall_indep = math.prod(lr.marginal_prob for lr in leg_results)
    overall_corr = overall_joint / overall_indep if overall_indep > 1e-15 else 1.0

    # sportsbook parlay odds (product of individual decimal odds)
    all_have_odds = all(lr.decimal_odds > 1.0 for lr in leg_results)
    sb_odds = math.prod(lr.decimal_odds for lr in leg_results) if all_have_odds else 0.0
    sb_implied = 1.0 / sb_odds if sb_odds > 1.0 else 0.0

    model_fair_odds = 1.0 / overall_joint if overall_joint > 1e-15 else float("inf")
    edge = overall_joint - sb_implied if sb_odds > 1.0 else 0.0
    ev = (overall_joint * (sb_odds - 1) - (1.0 - overall_joint)) if sb_odds > 1.0 else 0.0
    b = sb_odds - 1.0 if sb_odds > 1.0 else 0.0
    kelly = max(0.0, ev / b) * kelly_frac if b > 0 else 0.0

    # correlation direction
    corr_factors = [mg.correlation_factor for mg in match_groups if len(mg.legs) > 1]
    if not corr_factors:
        direction = "neutral"
    elif all(c > 1.02 for c in corr_factors):
        direction = "positive"
    elif all(c < 0.98 for c in corr_factors):
        direction = "negative"
    elif any(c > 1.02 for c in corr_factors) and any(c < 0.98 for c in corr_factors):
        direction = "mixed"
    else:
        direction = "neutral"

    analysis = _build_analysis(
        leg_results, match_groups, overall_joint, overall_indep,
        overall_corr, model_fair_odds, sb_odds, edge, direction,
    )

    return ParlayResult(
        legs=leg_results,
        match_groups=match_groups,
        joint_probability=overall_joint,
        independent_probability=overall_indep,
        correlation_factor=overall_corr,
        model_fair_odds=model_fair_odds,
        sportsbook_parlay_odds=sb_odds,
        edge=edge,
        ev=ev,
        kelly_fraction=kelly,
        correlation_direction=direction,
        analysis=analysis,
    )


def _build_analysis(
    legs: list[LegResult],
    groups: list[MatchGroup],
    joint: float,
    indep: float,
    corr: float,
    fair_odds: float,
    sb_odds: float,
    edge: float,
    direction: str,
) -> str:
    parts: list[str] = []
    n = len(legs)
    parts.append(f"{n}-leg parlay, {len(groups)} match(es).")

    if corr < 0.98:
        pct = (1.0 - corr) * 100
        parts.append(
            f"Negative correlation ({corr:.3f}): the joint probability is "
            f"{pct:.1f}% lower than independent pricing assumes. "
            "The sportsbook's independent-leg multiplication overstates your "
            "true probability -- the parlay is worse than it looks."
        )
    elif corr > 1.02:
        pct = (corr - 1.0) * 100
        parts.append(
            f"Positive correlation ({corr:.3f}): the joint probability is "
            f"{pct:.1f}% higher than independent pricing assumes. "
            "The sportsbook's independent-leg multiplication understates your "
            "true probability -- a potential edge if odds don't adjust."
        )
    else:
        parts.append(
            f"Near-independent correlation ({corr:.3f}): legs are approximately "
            "uncorrelated; the sportsbook's independent pricing is fair."
        )

    if sb_odds > 1.0:
        parts.append(
            f"Model fair odds: {fair_odds:.2f}. Sportsbook parlay odds: "
            f"{sb_odds:.2f}. Edge: {edge:+.1%}."
        )
        if edge > 0.03:
            parts.append("Positive expected value -- consider this parlay.")
        elif edge < -0.05:
            parts.append("Negative edge -- the sportsbook holds the advantage here.")

    for mg in groups:
        if len(mg.legs) > 1:
            parts.append(
                f"Match {mg.match_id}: within-match correlation {mg.correlation_factor:.3f} "
                f"(joint {mg.joint_prob:.4f} vs independent {mg.independent_prob:.4f})."
            )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def parlay_result_to_dict(result: ParlayResult) -> dict[str, Any]:
    """Serialize ParlayResult to JSON-safe dict for the API."""
    return {
        "legs": [
            {
                "match_id": lr.match_id,
                "leg_type": lr.leg_type,
                "selection": lr.selection,
                "marginal_prob": round(lr.marginal_prob, 6),
                "decimal_odds": lr.decimal_odds,
                "implied_prob": round(lr.implied_prob, 6),
            }
            for lr in result.legs
        ],
        "match_groups": [
            {
                "match_id": mg.match_id,
                "joint_prob": round(mg.joint_prob, 6),
                "independent_prob": round(mg.independent_prob, 6),
                "correlation_factor": round(mg.correlation_factor, 4),
                "mu_home": round(mg.mu_home, 3),
                "mu_away": round(mg.mu_away, 3),
                "n_legs": len(mg.legs),
            }
            for mg in result.match_groups
        ],
        "joint_probability": round(result.joint_probability, 6),
        "independent_probability": round(result.independent_probability, 6),
        "correlation_factor": round(result.correlation_factor, 4),
        "model_fair_odds": round(result.model_fair_odds, 2),
        "sportsbook_parlay_odds": round(result.sportsbook_parlay_odds, 2),
        "edge": round(result.edge, 4),
        "ev": round(result.ev, 4),
        "kelly_fraction": round(result.kelly_fraction, 4),
        "correlation_direction": result.correlation_direction,
        "analysis": result.analysis,
    }

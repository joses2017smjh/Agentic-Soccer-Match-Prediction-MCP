"""Compose the full per-match prediction JSON from the artifact bundle.

Authority rules (internal consistency):
- the calibrated + conformal GBM head owns the 1X2 probabilities;
- the Dixon–Coles grid (on the GBM's xG) owns everything scoreline-derived:
  exact scores, over/under, BTTS, knockout advancement, and P(0,0);
- the timing model distributes the grid's goals over time and states but
  never contradicts its marginals (see src/models/sequence.py);
- player props allocate the same team xG — no market gets its own private
  estimate of team strength.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.models.artifact import ArtifactBundle
from src.models.gbm import OUTCOME_CLASSES
from src.models.player_props import allocate_player_props
from src.models.score_grid import (
    btts,
    knockout_advance,
    outcome_probs,
    over_under,
    score_grid,
    top_scorelines,
)
from src.models.suggestions import Driver, MarketQuote, make_suggestions


def compose_prediction(
    bundle: ArtifactBundle,
    features_row: pd.DataFrame,
    *,
    home_players: pd.DataFrame | None = None,
    away_players: pd.DataFrame | None = None,
    quotes: list[MarketQuote] | None = None,
    drivers: dict[str, list[Driver]] | None = None,
    knockout: bool = False,
    top_n_scores: int = 5,
    ev_threshold: float = 0.03,
    kelly_fraction: float = 0.25,
) -> dict[str, Any]:
    """One match in, the full layered prediction out. ``features_row`` is a
    single-row frame matching the bundle's training schema (schema mismatch
    raises FeatureSchemaError upstream)."""
    raw = bundle.outcome.predict_proba(features_row)
    cal_probs, conf_sets = bundle.head.predict(raw)
    p_home, p_draw, p_away = (float(x) for x in cal_probs[0])
    conformal_set = [OUTCOME_CLASSES[i] for i in conf_sets[0]]

    mu_home, mu_away = bundle.xg.predict(features_row)
    mu_h, mu_a = float(mu_home[0]), float(mu_away[0])
    grid = score_grid(mu_h, mu_a, bundle.rho)

    result: dict[str, Any] = {
        "model_version": bundle.version,
        "match_outcome": {
            "home": p_home, "draw": p_draw, "away": p_away,
            "conformal_set": conformal_set,
            "conformal_alpha": bundle.card.get("conformal_alpha", 0.1),
        },
        "expected_goals": {"home": mu_h, "away": mu_a},
        "exact_score": {
            "top_scorelines": top_scorelines(grid, n=top_n_scores),
            "over_under_2_5": over_under(grid, 2.5),
            "btts": btts(grid),
            "grid_outcome_probs": outcome_probs(grid),
        },
        "event_sequence": {
            "first_scorer": bundle.timing.first_scorer(
                mu_h, mu_a, p_zero_zero=float(grid[0, 0])
            ),
            "goals_by_band": bundle.timing.expected_goals_by_band(mu_h, mu_a),
            "next_goal_from_kickoff": bundle.timing.next_goal(
                mu_h, mu_a, minute=0.0, score_home=0, score_away=0
            ),
        },
    }

    if knockout:
        result["knockout"] = {
            "advance": knockout_advance(mu_h, mu_a, bundle.rho)
        }

    if home_players is not None or away_players is not None:
        props: dict[str, list[dict]] = {}
        for side, frame, mu in [
            ("home", home_players, mu_h), ("away", away_players, mu_a)
        ]:
            if frame is None:
                continue
            alloc = allocate_player_props(frame, team_mu=mu)
            props[side] = alloc[
                ["player", "p_anytime_scorer", "p_assist",
                 "goal_lambda", "assist_lambda"]
            ].to_dict(orient="records")
        result["player_props"] = props

    if quotes:
        # the model side of every quote comes from THIS prediction — callers
        # supply market prices only, so edges can never use stale model probs
        model_prob_map = {
            ("h2h", "home"): p_home, ("h2h", "draw"): p_draw,
            ("h2h", "away"): p_away,
            ("totals_2.5", "over"): result["exact_score"]["over_under_2_5"]["over"],
            ("totals_2.5", "under"): result["exact_score"]["over_under_2_5"]["under"],
            ("btts", "yes"): result["exact_score"]["btts"]["yes"],
            ("btts", "no"): result["exact_score"]["btts"]["no"],
        }
        quotes = [
            MarketQuote(
                market=q.market, selection=q.selection,
                model_prob=model_prob_map.get((q.market, q.selection), q.model_prob),
                market_prob=q.market_prob, decimal_odds=q.decimal_odds,
            )
            for q in quotes
        ]
        suggestions = make_suggestions(
            quotes,
            ev_threshold=ev_threshold,
            kelly_fraction=kelly_fraction,
            h2h_conformal_set=conformal_set,
            drivers=drivers,
        )
        result["market_comparison"] = [
            {
                "market": s.market, "selection": s.selection,
                "model_prob": s.model_prob, "market_prob": s.market_prob,
                "decimal_odds": s.decimal_odds, "edge": s.edge, "ev": s.ev,
            }
            for s in suggestions
        ]
        result["suggestions"] = [
            {
                "market": s.market, "selection": s.selection,
                "edge": s.edge, "ev": s.ev, "kelly_stake": s.kelly_stake,
                "tier": s.tier, "rationale": s.rationale,
            }
            for s in suggestions if s.flagged
        ]

    return result

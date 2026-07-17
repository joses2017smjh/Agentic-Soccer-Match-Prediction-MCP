"""Assemble the per-match modeling frame: decayed team form + de-vigged odds.

Downstream steps (news availability, sentiment, tournament context) append
columns to the frame produced here; the GBM trains on the result.
"""

from __future__ import annotations

import pandas as pd

from src.features.odds_features import (
    consensus_market_probabilities,
    h2h_odds_features,
    odds_at_cutoff,
)
from src.features.team_form import match_level_form, team_form_features


def build_match_features(
    team_matches: pd.DataFrame,
    odds_snapshots: pd.DataFrame,
    *,
    form_window: int = 10,
    form_half_life: float = 5.0,
    cutoff_minutes: int = 60,
    devig_method: str = "power",
) -> pd.DataFrame:
    """One row per match: home_*/away_* form features + vig-free odds anchors.

    Both inputs pass through their own leakage guards (strictly-prior form
    matches; odds snapshots at or before kickoff − cutoff). Matches without
    any usable pre-cutoff odds keep NaN anchors rather than being dropped —
    the model treats missing odds as its own signal.
    """
    form_long = team_form_features(
        team_matches,
        window=form_window,
        half_life=form_half_life,
        cutoff_minutes=cutoff_minutes,
    )
    matches = match_level_form(form_long)

    schedule = matches[["match_id", "kickoff_utc"]].drop_duplicates()
    latest = odds_at_cutoff(odds_snapshots, schedule, cutoff_minutes=cutoff_minutes)
    market_probs = consensus_market_probabilities(latest, method=devig_method)
    anchors = h2h_odds_features(market_probs)

    return matches.merge(anchors, on="match_id", how="left")

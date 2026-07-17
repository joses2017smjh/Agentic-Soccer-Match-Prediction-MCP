"""Rolling, recency-decayed team form features.

For each (match, team) row, summarizes the team's previous ``window`` matches
with exponential decay (most recent match weighted highest). The current match
is never included — features for match i are computed from matches strictly
before it, and only from stat records published before the prediction cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.interfaces import TEAM_MATCH_COLUMNS, validate_frame
from src.features.leakage import assert_all_before, prediction_cutoff

FORM_STATS: list[str] = [
    "xg_for",
    "xg_against",
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "possession",
]


def decay_weights(n: int, half_life: float) -> np.ndarray:
    """Weights for n prior matches ordered oldest→newest; newest has weight 1."""
    ages = np.arange(n - 1, -1, -1, dtype=float)  # newest match: age 0
    return 0.5 ** (ages / half_life)


def _team_form(group: pd.DataFrame, window: int, half_life: float) -> pd.DataFrame:
    group = group.sort_values("kickoff_utc")
    values = group[FORM_STATS].to_numpy(dtype=float)
    n_rows = len(group)
    out = np.full((n_rows, len(FORM_STATS)), np.nan)
    counts = np.zeros(n_rows, dtype=int)

    for i in range(n_rows):
        start = max(0, i - window)
        past = values[start:i]  # strictly before match i — the leakage guard
        if past.size == 0:
            continue
        w = decay_weights(len(past), half_life)
        out[i] = np.average(past, axis=0, weights=w)
        counts[i] = len(past)

    form = pd.DataFrame(
        out, columns=[f"form_{s}" for s in FORM_STATS], index=group.index
    )
    form["form_n_matches"] = counts
    form["rest_days"] = (
        group["kickoff_utc"].diff().dt.total_seconds() / 86400.0
    )
    return form


def team_form_features(
    team_matches: pd.DataFrame,
    *,
    window: int = 10,
    half_life: float = 5.0,
    cutoff_minutes: int = 60,
) -> pd.DataFrame:
    """Return team_matches with form_* columns appended.

    Leakage guards:
    1. Form for match i uses only matches strictly earlier for that team.
    2. Every contributing stat record must have been published
       (``record_time_utc``) before the *next* match's prediction cutoff —
       violated rows fail hard rather than silently leaking.
    """
    df = validate_frame(team_matches, TEAM_MATCH_COLUMNS, "TEAM_MATCH").copy()
    df = df.sort_values(["team", "kickoff_utc"]).reset_index(drop=True)

    # Guard 2: a match's stats must be published before the team's next kickoff cutoff.
    check = df[["team", "kickoff_utc", "record_time_utc"]].copy()
    check["next_cutoff"] = prediction_cutoff(
        check.groupby("team")["kickoff_utc"].shift(-1), cutoff_minutes
    )
    prior = check[check["next_cutoff"].notna()]
    assert_all_before(prior, "record_time_utc", "next_cutoff", context="team_form stats")

    form = (
        df.groupby("team", group_keys=False)
        .apply(_team_form, window=window, half_life=half_life, include_groups=False)
        .sort_index()
    )
    return pd.concat([df, form], axis=1)


def match_level_form(team_form_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long team-form frame to one row per match (home_*/away_* columns)."""
    feature_cols = [c for c in team_form_df.columns if c.startswith("form_")] + [
        "rest_days"
    ]
    keys = ["match_id", "kickoff_utc", "competition", "season", "stage", "neutral_venue"]

    home = team_form_df[team_form_df["is_home"]].set_index("match_id")
    away = team_form_df[~team_form_df["is_home"]].set_index("match_id")
    merged = home[keys[1:] + ["team"] + feature_cols].join(
        away[["team"] + feature_cols], lsuffix="_home", rsuffix="_away", how="inner"
    )
    return merged.rename(
        columns={"team_home": "home_team", "team_away": "away_team"}
    ).reset_index()

"""Pipeline stages as plain functions over frames.

Each stage reuses the existing loader and feature code -- there is one
implementation of "raw CSV to canonical frame" and one of "form features" in
this repo, and the pipeline calls them rather than restating them. What the
stages add is the accounting around those calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.football_data_uk import closing_odds_frame, team_match_frame
from src.data.interfaces import TEAM_MATCH_COLUMNS, validate_frame
from src.features.team_form import match_level_form, team_form_features
from src.pipeline.quality import (
    RAW_REQUIRED,
    PipelineError,
    RejectLog,
    gate_odds,
    gate_team_matches,
    read_raw_csv,
)

_CLOSE_RENAME = {
    "odds_home": "close_odds_home", "odds_draw": "close_odds_draw",
    "odds_away": "close_odds_away", "odds_imp_home": "close_imp_home",
    "odds_imp_draw": "close_imp_draw", "odds_imp_away": "close_imp_away",
}


@dataclass
class SeasonFrames:
    """Canonical frames for one raw season file, plus what was rejected."""

    team_matches: pd.DataFrame
    odds_pre_close: pd.DataFrame
    odds_close: pd.DataFrame
    rejects: RejectLog
    odds_rejects: RejectLog


def ingest_season(path: Path, competition: str, start_year: int) -> SeasonFrames:
    """One raw season CSV -> gated canonical frames."""
    raw, malformed = read_raw_csv(path)
    missing_columns = [c for c in (*RAW_REQUIRED, "Date") if c not in raw.columns]
    if missing_columns:
        raise PipelineError(f"{path.name}: raw file has no column(s) {missing_columns}")

    rejects = RejectLog(checked=len(raw) + malformed)
    rejects.add("malformed_line", malformed)
    complete = raw.dropna(subset=list(RAW_REQUIRED))
    rejects.add("missing_result_or_team", len(raw) - len(complete))

    season = f"{start_year}-{start_year + 1}"
    if complete.empty:
        team = pd.DataFrame(columns=TEAM_MATCH_COLUMNS)
    else:
        team = team_match_frame(complete, competition, season)
    # the builder drops rows whose date cannot be parsed; two rows per match
    rejects.add("unparseable_kickoff", len(complete) - len(team) // 2)

    if not team.empty:
        team, gate_log = gate_team_matches(team, start_year)
        for reason, count in gate_log.reasons.items():
            rejects.add(reason, count)
        validate_frame(team, TEAM_MATCH_COLUMNS, "TEAM_MATCH")

    valid_ids = set(team["match_id"]) if not team.empty else set()
    odds_rejects = RejectLog()
    odds: dict[str, pd.DataFrame] = {}
    for stage in ("pre_close", "close"):
        frame = closing_odds_frame(complete, odds_stage=stage) if not complete.empty else pd.DataFrame()
        frame, log = gate_odds(frame, valid_ids)
        odds_rejects.merge(log)
        odds[stage] = frame
    return SeasonFrames(team, odds["pre_close"], odds["close"], rejects, odds_rejects)


def assemble_match_dataset(
    team_matches: pd.DataFrame,
    pre_close: pd.DataFrame,
    close: pd.DataFrame,
    *,
    form_window: int = 10,
    form_half_life: float = 5.0,
    cutoff_minutes: int = 60,
) -> pd.DataFrame:
    """One row per match: leakage-guarded form features, the result, the
    pre-close market anchor and the closing-line benchmark.

    Pre-close prices are the model's anchor; closing prices are what it is
    judged against. Keeping them in separate, clearly named columns is the
    B0 correction -- they must never be confused again.
    """
    form = team_form_features(
        team_matches, window=form_window, half_life=form_half_life,
        cutoff_minutes=cutoff_minutes,
    )
    matches = match_level_form(form)

    home_rows = (
        team_matches[team_matches["is_home"]]
        .set_index("match_id")[["goals_for", "goals_against"]]
    )
    matches = matches.merge(home_rows, left_on="match_id", right_index=True, how="inner")
    matches = matches.merge(pre_close, on="match_id", how="inner")
    close_cols = close.rename(columns=_CLOSE_RENAME)[["match_id", *_CLOSE_RENAME.values()]]
    return matches.merge(close_cols, on="match_id", how="inner")


def check_features(features: pd.DataFrame, matches_in: int) -> dict[str, int]:
    """Post-conditions on the processed frame; returns coverage counts."""
    if features["match_id"].duplicated().any():
        raise PipelineError("features: duplicate match_id in the processed frame")
    for prefix in ("odds_imp_", "close_imp_"):
        cols = [f"{prefix}{o}" for o in ("home", "draw", "away")]
        probs = features[cols]
        if not probs.empty and not ((probs > 0) & (probs < 1)).all().all():
            raise PipelineError(f"features: {prefix}* outside (0, 1)")
    return {"matches_in": int(matches_in), "rows_out": int(len(features)),
            "without_both_odds": int(matches_in - len(features))}

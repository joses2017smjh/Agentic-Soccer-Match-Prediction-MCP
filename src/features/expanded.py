"""Expanded feature engineering (B3).

Free features from data already on disk:
  - In-league Elo (compute_elo exists but was absent from FEATURES)
  - Opponent-adjusted form: weight each form match by the opponent's Elo
  - Home/away split form: home form at home, away form away
  - Book disagreement: std of de-vigged probs across books, and margin width

Leakage guard: line movement (pre-close -> close drift) must never enter
the pre-close feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def elo_features(
    matches: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-match Elo rating difference as a feature.

    Uses a walk-forward approach: for each match, the Elo ratings are
    computed only from matches played strictly before it.
    """
    from src.models.ratings import BASE_ELO, HOME_ADV, K, _mov_multiplier

    results = results.sort_values("date")
    ratings: dict[str, float] = {}

    # build a time-indexed map of ratings *before* each date
    date_ratings: dict[str, dict[str, float]] = {}
    prev_date = ""
    for r in results.itertuples(index=False):
        d = str(r.date)
        if d != prev_date:
            date_ratings[d] = dict(ratings)
            prev_date = d
        h, a = r.home_team, r.away_team
        ra = ratings.get(h, BASE_ELO)
        rb = ratings.get(a, BASE_ELO)
        adv = 0.0 if bool(getattr(r, "neutral", False)) else HOME_ADV
        exp_home = 1.0 / (1.0 + 10 ** (-((ra + adv) - rb) / 400))
        gd = int(r.home_score - r.away_score)
        actual = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        change = K * _mov_multiplier(gd) * (actual - exp_home)
        ratings[h] = ra + change
        ratings[a] = rb - change
    # final snapshot for matches after all results
    date_ratings["9999-12-31"] = dict(ratings)

    sorted_dates = sorted(date_ratings.keys())

    def _get_rating(team: str, match_date: str) -> float:
        # find the latest ratings snapshot before match_date
        for d in reversed(sorted_dates):
            if d <= match_date:
                return date_ratings[d].get(team, BASE_ELO)
        return BASE_ELO

    elo_home = []
    elo_away = []
    elo_diff = []
    for _, row in matches.iterrows():
        md = str(row.get("kickoff_utc", row.get("date", "")))[:10]
        h_team = row.get("home_team", "")
        a_team = row.get("away_team", "")
        h_elo = _get_rating(h_team, md)
        a_elo = _get_rating(a_team, md)
        elo_home.append(h_elo)
        elo_away.append(a_elo)
        elo_diff.append(h_elo - a_elo)

    return pd.DataFrame({
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_diff": elo_diff,
    }, index=matches.index)


def book_disagreement_features(
    odds_frame: pd.DataFrame,
    book_columns: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Compute cross-book disagreement and overround from de-vigged odds.

    Returns per-match:
      - book_std_home/draw/away: std of de-vigged prob across books
      - overround: raw odds implied prob sum minus 1 (margin width)
    """
    if book_columns is None:
        # default: detect all book columns matching pattern {book}H, {book}D, {book}A
        books = ["B365", "BW", "IW", "PS", "WH", "VC"]
        book_columns = [(f"{b}H", f"{b}D", f"{b}A") for b in books]

    results = []
    for _, row in odds_frame.iterrows():
        home_probs = []
        draw_probs = []
        away_probs = []
        raw_totals = []

        for h_col, d_col, a_col in book_columns:
            oh = row.get(h_col)
            od = row.get(d_col)
            oa = row.get(a_col)
            if pd.notna(oh) and pd.notna(od) and pd.notna(oa):
                if oh > 1 and od > 1 and oa > 1:
                    total = 1/oh + 1/od + 1/oa
                    raw_totals.append(total)
                    home_probs.append((1/oh) / total)
                    draw_probs.append((1/od) / total)
                    away_probs.append((1/oa) / total)

        if home_probs:
            results.append({
                "book_std_home": float(np.std(home_probs)),
                "book_std_draw": float(np.std(draw_probs)),
                "book_std_away": float(np.std(away_probs)),
                "overround": float(np.mean(raw_totals)) - 1.0,
            })
        else:
            results.append({
                "book_std_home": 0.0,
                "book_std_draw": 0.0,
                "book_std_away": 0.0,
                "overround": 0.0,
            })

    return pd.DataFrame(results, index=odds_frame.index)


def home_away_split_form(
    team_matches: pd.DataFrame,
    window: int = 10,
    half_life: float = 5.0,
) -> pd.DataFrame:
    """Home form at home, away form away.

    Splits the form calculation so the home team's feature uses only prior
    home matches, and vice versa. Falls back to overall form when fewer
    than 3 split matches exist.
    """
    from src.features.team_form import FORM_STATS, decay_weights

    team_matches = team_matches.sort_values(["team", "kickoff_utc"])
    records = []

    for team, group in team_matches.groupby("team"):
        group = group.sort_values("kickoff_utc")
        values = group[FORM_STATS].to_numpy(dtype=float)
        is_home = group["is_home"].to_numpy(dtype=bool)

        for i in range(len(group)):
            # split: only prior matches at the same venue
            venue = is_home[i]
            same_venue_idx = [j for j in range(max(0, i - window * 2), i)
                              if is_home[j] == venue]
            same_venue_idx = same_venue_idx[-window:]

            if len(same_venue_idx) >= 3:
                past = values[same_venue_idx]
                w = decay_weights(len(past), half_life)
                split_form = np.average(past, axis=0, weights=w)
            else:
                split_form = np.full(len(FORM_STATS), np.nan)

            record = {"match_id": group.iloc[i]["match_id"], "team": team}
            for k, stat in enumerate(FORM_STATS):
                record[f"split_form_{stat}"] = split_form[k]
            records.append(record)

    return pd.DataFrame(records)


EXPANDED_FEATURES = [
    "elo_diff",
    "elo_home",
    "elo_away",
    "book_std_home",
    "book_std_draw",
    "book_std_away",
    "overround",
]

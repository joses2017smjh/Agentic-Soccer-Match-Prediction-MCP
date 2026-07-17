"""football-data.co.uk provider — free, no key, real results + closing odds.

One HTTP GET per (division, season) returns a CSV with full-time results,
shots / shots-on-target, and closing odds from several bookmakers (Bet365,
Pinnacle, ...). CSVs are cached under data/raw/football_data_uk/ so repeat
runs and tests are offline.

Two outputs, both canonical:
- ``team_match_frame``   TEAM_MATCH long frame (src/data/interfaces.py)
- ``closing_odds_frame`` one row per match with payable decimal odds and
                         de-vigged (power method) implied probabilities

xG proxy: this source has no xG. We use a documented shot-quality proxy
    xg_proxy = 0.30 * shots_on_target + 0.03 * (shots − shots_on_target)
(league-average conversion ≈ 30% for on-target, ≈ 3% for the rest). It is a
crude stand-in until an event provider (StatsBomb/FBref) is wired; the
column keeps the canonical name so downstream code is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.data.interfaces import TEAM_MATCH_COLUMNS, validate_frame
from src.features.odds_features import remove_overround

BASE_URL = "https://www.football-data.co.uk/mmz4281"
DEFAULT_CACHE = Path("data/raw/football_data_uk")

# bookmaker column prefixes, in preference order for the consensus close
_BOOKS = ["PS", "B365", "WH", "BW"]  # Pinnacle first: sharpest close


def season_code(start_year: int) -> str:
    """2024 → '2425' (the 2024-25 season)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def fetch_season_csv(
    division: str, start_year: int, cache_dir: Path = DEFAULT_CACHE,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Download (or read cached) one season CSV."""
    code = season_code(start_year)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{division}_{code}.csv"
    if not path.exists():
        url = f"{BASE_URL}/{code}/{division}.csv"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")


def _kickoff_utc(raw: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
    two_digit = date.isna()
    if two_digit.any():
        date.loc[two_digit] = pd.to_datetime(
            raw.loc[two_digit, "Date"], format="%d/%m/%y", errors="coerce"
        )
    time = raw.get("Time", pd.Series("15:00", index=raw.index)).fillna("15:00")
    return pd.to_datetime(
        date.dt.strftime("%Y-%m-%d") + " " + time, errors="coerce"
    ).dt.tz_localize("UTC")


def _match_ids(raw: pd.DataFrame, kickoff: pd.Series) -> pd.Series:
    slug = (
        raw["HomeTeam"].str.replace(r"\W", "", regex=True)
        + "-" + raw["AwayTeam"].str.replace(r"\W", "", regex=True)
    )
    return slug + "-" + kickoff.dt.strftime("%Y-%m-%d")


def team_match_frame(
    raw: pd.DataFrame, competition: str, season: str
) -> pd.DataFrame:
    """CSV → canonical TEAM_MATCH long frame (one row per match per team)."""
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    kickoff = _kickoff_utc(raw)
    match_id = _match_ids(raw, kickoff)

    shots_h = raw.get("HS", pd.Series(float("nan"), index=raw.index))
    shots_a = raw.get("AS", pd.Series(float("nan"), index=raw.index))
    sot_h = raw.get("HST", pd.Series(float("nan"), index=raw.index))
    sot_a = raw.get("AST", pd.Series(float("nan"), index=raw.index))
    xg_h = 0.30 * sot_h + 0.03 * (shots_h - sot_h)
    xg_a = 0.30 * sot_a + 0.03 * (shots_a - sot_a)

    def _side(is_home: bool) -> pd.DataFrame:
        us, them = ("HomeTeam", "AwayTeam") if is_home else ("AwayTeam", "HomeTeam")
        gf, ga = ("FTHG", "FTAG") if is_home else ("FTAG", "FTHG")
        return pd.DataFrame({
            "match_id": match_id,
            "kickoff_utc": kickoff,
            "competition": competition,
            "season": season,
            "stage": "group",          # league play: no knockout rounds
            "team": raw[us],
            "opponent": raw[them],
            "is_home": is_home,
            "neutral_venue": False,
            "goals_for": raw[gf].astype(float),
            "goals_against": raw[ga].astype(float),
            "xg_for": xg_h if is_home else xg_a,
            "xg_against": xg_a if is_home else xg_h,
            "shots_for": (shots_h if is_home else shots_a).astype(float),
            "shots_against": (shots_a if is_home else shots_h).astype(float),
            "possession": float("nan"),  # not published by this source
            "record_time_utc": kickoff + pd.Timedelta(hours=3),
        })

    frame = pd.concat(
        [_side(True), _side(False)], ignore_index=True
    ).dropna(subset=["kickoff_utc"])
    return validate_frame(frame, TEAM_MATCH_COLUMNS, "TEAM_MATCH")


def closing_odds_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-match closing odds: payable prices + de-vigged probabilities.

    Uses the first bookmaker (Pinnacle-first order) with all three prices
    present. These are CLOSING odds — the strongest market benchmark; as a
    model feature they proxy the pre-cutoff price and that approximation is
    disclosed wherever results are reported.
    """
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    kickoff = _kickoff_utc(raw)
    rows = []
    for idx in raw.index:
        rec = raw.loc[idx]
        for book in _BOOKS:
            cols = [f"{book}H", f"{book}D", f"{book}A"]
            if all(c in raw.columns and pd.notna(rec[c]) and rec[c] > 1.0
                   for c in cols):
                h, d, a = (float(rec[c]) for c in cols)
                ph, pd_, pa = remove_overround(
                    pd.array([h, d, a], dtype=float).to_numpy(), method="power"
                )
                rows.append({
                    "match_id": _match_ids(raw.loc[[idx]], kickoff.loc[[idx]]).iloc[0],
                    "book": book,
                    "odds_home": h, "odds_draw": d, "odds_away": a,
                    "odds_imp_home": float(ph), "odds_imp_draw": float(pd_),
                    "odds_imp_away": float(pa),
                })
                break
    return pd.DataFrame(rows)


def load_seasons(
    division: str, start_years: list[int], *,
    competition: str = "EPL", cache_dir: Path = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(TEAM_MATCH frame, closing-odds frame) across several seasons."""
    team_frames, odds_frames = [], []
    for year in start_years:
        raw = fetch_season_csv(division, year, cache_dir)
        season = f"{year}-{year + 1}"
        team_frames.append(team_match_frame(raw, competition, season))
        odds_frames.append(closing_odds_frame(raw))
    return (
        pd.concat(team_frames, ignore_index=True),
        pd.concat(odds_frames, ignore_index=True),
    )

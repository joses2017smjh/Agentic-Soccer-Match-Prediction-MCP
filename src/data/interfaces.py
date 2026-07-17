"""Provider-agnostic data contracts and ingestion interfaces.

Feature and model code depends only on the canonical frames defined here.
Concrete clients (StatsBomb, FBref, API-Football, The Odds API, RSS/NewsAPI)
each implement one Protocol and return these frames — swapping a provider
never touches feature code.

Canonical frames (column contracts, enforced by `validate_frame`):

TEAM_MATCH — one row per (match, team), long format:
    match_id, kickoff_utc, competition, season, stage, team, opponent,
    is_home, neutral_venue, goals_for, goals_against, xg_for, xg_against,
    shots_for, shots_against, possession, record_time_utc

ODDS_SNAPSHOT — one row per (match, bookmaker, market, outcome, snapshot):
    match_id, bookmaker, market, outcome, decimal_odds, snapshot_time_utc

Every canonical row carries the UTC timestamp at which the information became
available; the leakage guards in src/features/leakage.py key off these columns.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

TEAM_MATCH_COLUMNS: list[str] = [
    "match_id",
    "kickoff_utc",
    "competition",
    "season",
    "stage",
    "team",
    "opponent",
    "is_home",
    "neutral_venue",
    "goals_for",
    "goals_against",
    "xg_for",
    "xg_against",
    "shots_for",
    "shots_against",
    "possession",
    "record_time_utc",
]

ODDS_SNAPSHOT_COLUMNS: list[str] = [
    "match_id",
    "bookmaker",
    "market",
    "outcome",
    "decimal_odds",
    "snapshot_time_utc",
]

_TIMESTAMP_COLUMNS = {"kickoff_utc", "record_time_utc", "snapshot_time_utc"}


def validate_frame(df: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    """Check a provider frame against its contract; return it unchanged."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} frame missing columns: {missing}")
    for col in _TIMESTAMP_COLUMNS & set(columns):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise TypeError(f"{name}.{col} must be datetime64 (UTC), got {df[col].dtype}")
        if df[col].dt.tz is None:
            raise TypeError(f"{name}.{col} must be timezone-aware UTC")
    return df


class MatchStatsProvider(Protocol):
    """Historical team-match statistics (FBref, football-data.org, ...)."""

    def team_matches(
        self, competitions: list[str], seasons: list[str]
    ) -> pd.DataFrame:
        """Return a TEAM_MATCH frame for the given competitions/seasons."""
        ...


class OddsProvider(Protocol):
    """Betting odds, live and historical (The Odds API, Football-Data.co.uk)."""

    def odds_snapshots(
        self, match_ids: list[str], markets: list[str]
    ) -> pd.DataFrame:
        """Return an ODDS_SNAPSHOT frame; every row timestamped at capture."""
        ...


class FixtureProvider(Protocol):
    """Upcoming fixtures and confirmed lineups (API-Football)."""

    def fixtures(self, competition: str, from_utc: datetime, to_utc: datetime) -> pd.DataFrame:
        ...

    def lineups(self, match_id: str) -> pd.DataFrame:
        """Confirmed lineups once released (~1h before kickoff), timestamped."""
        ...


class EventProvider(Protocol):
    """Play-by-play event timelines (StatsBomb Open Data)."""

    def match_events(self, match_id: str) -> pd.DataFrame:
        ...


class NewsProvider(Protocol):
    """Raw news items for the availability parser and sentiment scorer."""

    def articles(self, teams: list[str], from_utc: datetime, to_utc: datetime) -> pd.DataFrame:
        """Return columns: team, title, body, source, published_utc."""
        ...

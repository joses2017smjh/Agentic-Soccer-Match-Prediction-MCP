"""Chronological replay of real EPL fixtures from football-data.co.uk.

Loads cached CSVs and yields gameweeks (groups of matches on consecutive
dates) for the environment to step through.  Pre-close and closing odds
from Pinnacle (fallback Bet365) are both available per match.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "football_data_uk"


@dataclass(frozen=True)
class Fixture:
    date: date
    match_id: str
    home: str
    away: str
    ftr: str  # H/D/A
    odds_h: float
    odds_d: float
    odds_a: float
    close_h: float
    close_d: float
    close_a: float
    odds_source: str

    @property
    def closing_odds(self) -> dict[str, float]:
        return {"H": self.close_h, "D": self.close_d, "A": self.close_a}

    @property
    def pre_close_odds(self) -> dict[str, float]:
        return {"H": self.odds_h, "D": self.odds_d, "A": self.odds_a}


@dataclass
class Gameweek:
    number: int
    dates: list[date]
    fixtures: list[Fixture]


def _float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_season(season_file: str | Path) -> list[Fixture]:
    """Load one season CSV into a list of Fixture, sorted by date."""
    fixtures: list[Fixture] = []
    with open(season_file, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            date_str = row.get("Date", "")
            try:
                d = datetime.strptime(date_str, "%d/%m/%Y").date()
            except (ValueError, TypeError):
                continue
            home = row.get("HomeTeam", "")
            away = row.get("AwayTeam", "")
            fixtures.append(Fixture(
                date=d,
                match_id=f"{home}-{away}-{d.isoformat()}",
                home=home,
                away=away,
                ftr=row.get("FTR", ""),
                odds_h=_float(row.get("PSH") or row.get("B365H")),
                odds_d=_float(row.get("PSD") or row.get("B365D")),
                odds_a=_float(row.get("PSA") or row.get("B365A")),
                close_h=_float(row.get("PSCH") or row.get("B365CH")),
                close_d=_float(row.get("PSCD") or row.get("B365CD")),
                close_a=_float(row.get("PSCA") or row.get("B365CA")),
                odds_source="pinnacle" if row.get("PSCH") else "bet365",
            ))
    fixtures.sort(key=lambda f: f.date)
    return fixtures


def load_all_seasons() -> list[Fixture]:
    """Load all cached EPL seasons, sorted chronologically."""
    all_fixtures: list[Fixture] = []
    for csv_path in sorted(_DATA_DIR.glob("E0_*.csv")):
        all_fixtures.extend(load_season(csv_path))
    all_fixtures.sort(key=lambda f: f.date)
    return all_fixtures


def iter_gameweeks(fixtures: list[Fixture]) -> Iterator[Gameweek]:
    """Group fixtures into gameweeks (clusters of dates within 3 days)."""
    if not fixtures:
        return

    gw_num = 1
    current_dates: list[date] = [fixtures[0].date]
    current_fixtures: list[Fixture] = [fixtures[0]]

    for fix in fixtures[1:]:
        gap = (fix.date - current_dates[-1]).days
        if gap <= 3 and len(current_fixtures) < 12:
            if fix.date not in current_dates:
                current_dates.append(fix.date)
            current_fixtures.append(fix)
        else:
            yield Gameweek(number=gw_num, dates=current_dates, fixtures=current_fixtures)
            gw_num += 1
            current_dates = [fix.date]
            current_fixtures = [fix]

    if current_fixtures:
        yield Gameweek(number=gw_num, dates=current_dates, fixtures=current_fixtures)

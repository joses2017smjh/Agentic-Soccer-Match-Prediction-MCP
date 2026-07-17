"""Shared synthetic fixtures for feature tests."""

from __future__ import annotations

import pandas as pd
import pytest

UTC = "UTC"


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


@pytest.fixture
def team_matches() -> pd.DataFrame:
    """Three matches for team A (vs B, C, then B again), long format both sides."""
    rows = []
    fixtures = [
        ("m1", "2024-06-01 15:00", "A", "B", 2, 1, 1.8, 0.9, 14, 8, 60.0),
        ("m2", "2024-06-05 15:00", "A", "C", 0, 0, 1.2, 1.1, 10, 11, 52.0),
        ("m3", "2024-06-09 15:00", "A", "B", 1, 3, 0.7, 2.4, 6, 18, 45.0),
    ]
    for mid, ko, home, away, gh, ga, xgh, xga, sh, sa, poss in fixtures:
        kickoff = _ts(ko)
        common = dict(
            match_id=mid,
            kickoff_utc=kickoff,
            competition="TEST",
            season="2024",
            stage="group",
            neutral_venue=False,
            record_time_utc=kickoff + pd.Timedelta(hours=3),
        )
        rows.append(
            common
            | dict(team=home, opponent=away, is_home=True, goals_for=gh,
                   goals_against=ga, xg_for=xgh, xg_against=xga, shots_for=sh,
                   shots_against=sa, possession=poss)
        )
        rows.append(
            common
            | dict(team=away, opponent=home, is_home=False, goals_for=ga,
                   goals_against=gh, xg_for=xga, xg_against=xgh, shots_for=sa,
                   shots_against=sh, possession=100 - poss)
        )
    return pd.DataFrame(rows)


@pytest.fixture
def odds_snapshots() -> pd.DataFrame:
    """h2h snapshots for m3 (kickoff 2024-06-09 15:00): two before cutoff, one after."""
    rows = []
    for t, h, d, a in [
        ("2024-06-09 10:00", 2.10, 3.40, 3.60),
        ("2024-06-09 13:30", 2.30, 3.40, 3.20),  # latest usable (cutoff 14:00)
        ("2024-06-09 14:30", 2.60, 3.40, 2.90),  # after cutoff — must be ignored
    ]:
        for outcome, odds in [("home", h), ("draw", d), ("away", a)]:
            rows.append(
                dict(match_id="m3", bookmaker="bookA", market="h2h",
                     outcome=outcome, decimal_odds=odds, snapshot_time_utc=_ts(t))
            )
    return pd.DataFrame(rows)

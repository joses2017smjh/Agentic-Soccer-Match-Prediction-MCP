"""Unit tests: decayed team form and its leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.leakage import LeakageError
from src.features.team_form import (
    decay_weights,
    match_level_form,
    team_form_features,
)


def test_decay_weights_newest_heaviest() -> None:
    w = decay_weights(3, half_life=1.0)
    assert w[-1] == 1.0  # newest prior match
    np.testing.assert_allclose(w, [0.25, 0.5, 1.0])


def test_first_match_has_no_form(team_matches: pd.DataFrame) -> None:
    out = team_form_features(team_matches, window=10, half_life=5.0)
    first_a = out[(out["team"] == "A") & (out["match_id"] == "m1")].iloc[0]
    assert np.isnan(first_a["form_xg_for"])
    assert first_a["form_n_matches"] == 0


def test_current_match_excluded_from_form(team_matches: pd.DataFrame) -> None:
    """Form before m2 must equal m1's stats exactly — m2 itself not included."""
    out = team_form_features(team_matches, window=10, half_life=5.0)
    a_m2 = out[(out["team"] == "A") & (out["match_id"] == "m2")].iloc[0]
    assert a_m2["form_xg_for"] == pytest.approx(1.8)
    assert a_m2["form_goals_for"] == pytest.approx(2.0)
    assert a_m2["form_n_matches"] == 1


def test_decayed_average_two_matches(team_matches: pd.DataFrame) -> None:
    """Form before m3 = decayed mean of m1, m2 with half_life=1 → weights 0.5, 1."""
    out = team_form_features(team_matches, window=10, half_life=1.0)
    a_m3 = out[(out["team"] == "A") & (out["match_id"] == "m3")].iloc[0]
    expected = (0.5 * 1.8 + 1.0 * 1.2) / 1.5
    assert a_m3["form_xg_for"] == pytest.approx(expected)
    assert a_m3["form_n_matches"] == 2
    assert a_m3["rest_days"] == pytest.approx(4.0)


def test_late_published_stats_raise(team_matches: pd.DataFrame) -> None:
    """Stats published after the next match's cutoff must hard-fail, not leak."""
    bad = team_matches.copy()
    m1 = bad["match_id"] == "m1"
    bad.loc[m1, "record_time_utc"] = pd.Timestamp("2024-06-05 14:30", tz="UTC")
    with pytest.raises(LeakageError):
        team_form_features(bad, window=10, half_life=5.0)


def test_match_level_pivot(team_matches: pd.DataFrame) -> None:
    long = team_form_features(team_matches, window=10, half_life=5.0)
    wide = match_level_form(long)
    assert len(wide) == 3
    m3 = wide[wide["match_id"] == "m3"].iloc[0]
    assert m3["home_team"] == "A" and m3["away_team"] == "B"
    assert m3["form_n_matches_home"] == 2  # A played m1, m2
    assert m3["form_n_matches_away"] == 1  # B only played m1
    assert m3["form_xg_against_away"] == pytest.approx(1.8)  # B conceded A's 1.8 xG in m1

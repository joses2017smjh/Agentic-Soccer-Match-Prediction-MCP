"""Unit tests: leakage guard helpers and the end-to-end feature assembly."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_features import build_match_features
from src.features.leakage import (
    LeakageError,
    assert_all_before,
    merge_asof_guarded,
    prediction_cutoff,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_prediction_cutoff() -> None:
    assert prediction_cutoff(_ts("2024-06-09 15:00"), 60) == _ts("2024-06-09 14:00")


def test_assert_all_before_raises_with_context() -> None:
    df = pd.DataFrame({"ts": [_ts("2024-01-01 12:00")], "cut": [_ts("2024-01-01 11:00")]})
    with pytest.raises(LeakageError, match="unit-test"):
        assert_all_before(df, "ts", "cut", context="unit-test")


def test_merge_asof_guarded_takes_latest_prior() -> None:
    left = pd.DataFrame({"key": ["x"], "cutoff": [_ts("2024-01-01 14:00")]})
    right = pd.DataFrame(
        {
            "key": ["x", "x", "x"],
            "event_ts": [_ts("2024-01-01 10:00"), _ts("2024-01-01 13:59"),
                         _ts("2024-01-01 14:01")],
            "value": [1, 2, 3],
        }
    )
    out = merge_asof_guarded(
        left, right, left_cutoff="cutoff", right_ts="event_ts", by="key"
    )
    assert out["value"].tolist() == [2]  # 13:59 row; 14:01 never eligible


def test_build_match_features_end_to_end(
    team_matches: pd.DataFrame, odds_snapshots: pd.DataFrame
) -> None:
    features = build_match_features(
        team_matches, odds_snapshots, form_window=10, form_half_life=5.0
    )
    assert len(features) == 3
    m3 = features[features["match_id"] == "m3"].iloc[0]
    # form and odds both present for m3
    assert m3["form_n_matches_home"] == 2
    assert m3["odds_imp_home"] + m3["odds_imp_draw"] + m3["odds_imp_away"] == pytest.approx(1.0)
    # m1/m2 had no odds snapshots — anchors stay NaN, matches are kept
    m1 = features[features["match_id"] == "m1"].iloc[0]
    assert pd.isna(m1["odds_imp_home"])

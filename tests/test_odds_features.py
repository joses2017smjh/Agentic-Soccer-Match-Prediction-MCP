"""Unit tests: vig removal and timestamp-aligned odds selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.odds_features import (
    consensus_market_probabilities,
    h2h_odds_features,
    implied_probabilities,
    odds_at_cutoff,
    remove_overround,
)


def test_implied_probabilities_include_overround() -> None:
    raw = implied_probabilities(np.array([2.0, 3.5, 3.5]))
    assert raw.sum() > 1.0


def test_implied_probabilities_reject_bad_odds() -> None:
    with pytest.raises(ValueError):
        implied_probabilities(np.array([1.0, 3.5]))


@pytest.mark.parametrize("method", ["proportional", "power"])
def test_devig_sums_to_one_and_preserves_order(method: str) -> None:
    odds = np.array([1.5, 4.2, 7.0])
    probs = remove_overround(odds, method=method)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert probs[0] > probs[1] > probs[2]


def test_power_devig_shaves_longshots_harder() -> None:
    """Power method should assign the longshot less probability than proportional."""
    odds = np.array([1.3, 5.0, 12.0])
    prop = remove_overround(odds, method="proportional")
    power = remove_overround(odds, method="power")
    assert power[2] < prop[2]
    assert power[0] > prop[0]


def test_odds_at_cutoff_ignores_post_cutoff_snapshot(
    odds_snapshots: pd.DataFrame,
) -> None:
    schedule = pd.DataFrame(
        {"match_id": ["m3"], "kickoff_utc": [pd.Timestamp("2024-06-09 15:00", tz="UTC")]}
    )
    latest = odds_at_cutoff(odds_snapshots, schedule, cutoff_minutes=60)
    # cutoff is 14:00 → the 13:30 snapshot wins; the 14:30 one must be invisible
    home = latest[latest["outcome"] == "home"].iloc[0]
    assert home["decimal_odds"] == pytest.approx(2.30)
    assert (latest["snapshot_time_utc"] <= pd.Timestamp("2024-06-09 14:00", tz="UTC")).all()


def test_h2h_anchor_features(odds_snapshots: pd.DataFrame) -> None:
    schedule = pd.DataFrame(
        {"match_id": ["m3"], "kickoff_utc": [pd.Timestamp("2024-06-09 15:00", tz="UTC")]}
    )
    latest = odds_at_cutoff(odds_snapshots, schedule, cutoff_minutes=60)
    probs = consensus_market_probabilities(latest, method="proportional")
    wide = h2h_odds_features(probs)
    row = wide.iloc[0]
    total = row["odds_imp_home"] + row["odds_imp_draw"] + row["odds_imp_away"]
    assert total == pytest.approx(1.0, abs=1e-9)
    # 13:30 snapshot: 2.30 / 3.40 / 3.20 → home favourite after de-vig
    assert row["odds_imp_home"] > row["odds_imp_away"] > row["odds_imp_draw"]

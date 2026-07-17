"""Unit tests: goal-timing model, grid reconciliation, in-play next-goal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.score_grid import score_grid
from src.models.sequence import BAND_LABELS, GoalTimingModel


@pytest.fixture
def fitted() -> GoalTimingModel:
    """Train on synthetic goals skewed late, trailing teams scoring more."""
    rng = np.random.default_rng(7)
    minutes = rng.uniform(0, 90, 600) ** 1.15 % 90  # mild late skew
    states = rng.choice(["level", "trailing", "leading"], 600, p=[0.5, 0.3, 0.2])
    return GoalTimingModel().fit(
        pd.DataFrame({"minute": minutes, "scorer_state": states})
    )


def test_band_multipliers_mean_one(fitted: GoalTimingModel) -> None:
    assert fitted.band_multipliers.mean() == pytest.approx(1.0)
    assert (fitted.band_multipliers > 0).all()


def test_expected_goals_by_band_sums_to_mu(fitted: GoalTimingModel) -> None:
    bands = fitted.expected_goals_by_band(1.6, 1.1)
    assert [b["band"] for b in bands] == BAND_LABELS
    assert sum(b["home"] for b in bands) == pytest.approx(1.6)
    assert sum(b["away"] for b in bands) == pytest.approx(1.1)


def test_first_scorer_reconciled_with_grid(fitted: GoalTimingModel) -> None:
    """P(no goals) must equal the Dixon-Coles grid's P(0,0) exactly, and the
    home/away split must fill the remainder in mu ratio."""
    mu_h, mu_a, rho = 1.6, 1.1, -0.12
    p00 = float(score_grid(mu_h, mu_a, rho)[0, 0])
    fs = fitted.first_scorer(mu_h, mu_a, p_zero_zero=p00)
    assert fs["no_goals"] == pytest.approx(p00)
    assert sum(fs.values()) == pytest.approx(1.0)
    assert fs["home_first"] / fs["away_first"] == pytest.approx(mu_h / mu_a)


def test_next_goal_probs_and_state_effects(fitted: GoalTimingModel) -> None:
    base = fitted.next_goal(1.5, 1.5, minute=60, score_home=0, score_away=0)
    assert base["home"] + base["away"] + base["no_more_goals"] == pytest.approx(1.0)
    assert base["home"] == pytest.approx(base["away"])

    # equal-strength teams, home trailing → trailing boost favours home
    behind = fitted.next_goal(1.5, 1.5, minute=60, score_home=0, score_away=1)
    if fitted.trailing_boost > fitted.leading_damp:
        assert behind["home"] > behind["away"]


def test_next_goal_at_ninety_is_certainly_none(fitted: GoalTimingModel) -> None:
    end = fitted.next_goal(1.5, 1.2, minute=90, score_home=1, score_away=0)
    assert end["no_more_goals"] == pytest.approx(1.0)


def test_roundtrip_persistence(fitted: GoalTimingModel, tmp_path) -> None:
    fitted.save(tmp_path)
    loaded = GoalTimingModel.load(tmp_path)
    np.testing.assert_allclose(loaded.band_multipliers, fitted.band_multipliers)
    assert loaded.trailing_boost == fitted.trailing_boost

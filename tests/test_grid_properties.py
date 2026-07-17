"""Property tests: every market derived from one grid stays consistent.

The central architecture claim is "one grid, no market contradicts another."
These tests recompute each derived market by brute force from the same grid
across random (mu_home, mu_away, rho) draws and assert agreement — if a
refactor ever splits the sources of truth, this fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.score_grid import (
    btts,
    knockout_advance,
    outcome_probs,
    over_under,
    score_grid,
)
from src.models.sequence import GoalTimingModel

RNG = np.random.default_rng(7)
CASES = [
    (float(RNG.uniform(0.3, 3.5)), float(RNG.uniform(0.3, 3.5)),
     float(RNG.uniform(-0.15, 0.1)))
    for _ in range(25)
]


@pytest.mark.parametrize("mu_h,mu_a,rho", CASES)
def test_grid_markets_agree_with_brute_force(mu_h, mu_a, rho) -> None:
    grid = score_grid(mu_h, mu_a, rho)
    n = grid.shape[0]
    assert grid.sum() == pytest.approx(1.0, abs=1e-9)

    brute_home = sum(grid[i, j] for i in range(n) for j in range(n) if i > j)
    brute_draw = sum(grid[i, i] for i in range(n))
    probs = outcome_probs(grid)
    assert probs["home"] == pytest.approx(brute_home, abs=1e-12)
    assert probs["draw"] == pytest.approx(brute_draw, abs=1e-12)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)

    brute_over = sum(grid[i, j] for i in range(n) for j in range(n) if i + j > 2.5)
    ou = over_under(grid, 2.5)
    assert ou["over"] == pytest.approx(brute_over, abs=1e-12)
    assert ou["over"] + ou["under"] == pytest.approx(1.0)

    brute_btts = sum(grid[i, j] for i in range(1, n) for j in range(1, n))
    assert btts(grid)["yes"] == pytest.approx(brute_btts, abs=1e-12)


@pytest.mark.parametrize("mu_h,mu_a,rho", CASES[:10])
def test_sequence_and_knockout_share_the_grid(mu_h, mu_a, rho) -> None:
    grid = score_grid(mu_h, mu_a, rho)
    fs = GoalTimingModel().first_scorer(mu_h, mu_a, p_zero_zero=float(grid[0, 0]))
    assert fs["no_goals"] == pytest.approx(float(grid[0, 0]), abs=1e-12)
    assert sum(fs.values()) == pytest.approx(1.0, abs=1e-9)

    adv = knockout_advance(mu_h, mu_a, rho)
    assert adv["home"] + adv["away"] == pytest.approx(1.0, abs=1e-9)
    # advancing can never be less likely than winning in 90 minutes
    assert adv["home"] >= outcome_probs(grid)["home"] - 1e-9

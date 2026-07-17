"""Unit tests: Dixon-Coles grid, derived markets, rho fit, knockout advance."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.score_grid import (
    btts,
    fit_rho,
    knockout_advance,
    outcome_probs,
    over_under,
    score_grid,
    top_scorelines,
)


def test_grid_sums_to_one_and_favours_stronger_team() -> None:
    grid = score_grid(2.0, 0.8, rho=-0.1)
    assert grid.sum() == pytest.approx(1.0)
    probs = outcome_probs(grid)
    assert probs["home"] > probs["away"]
    assert sum(probs.values()) == pytest.approx(1.0)


def test_negative_rho_inflates_low_draws() -> None:
    """rho < 0 must raise P(0-0) and P(1-1) relative to independence."""
    dc = score_grid(1.4, 1.1, rho=-0.15)
    indep = score_grid(1.4, 1.1, rho=0.0)
    assert dc[0, 0] > indep[0, 0]
    assert dc[1, 1] > indep[1, 1]
    assert dc[1, 0] < indep[1, 0]


def test_derived_markets_internally_consistent() -> None:
    grid = score_grid(1.5, 1.2, rho=-0.1)
    ou = over_under(grid, 2.5)
    assert ou["over"] + ou["under"] == pytest.approx(1.0)
    b = btts(grid)
    assert b["yes"] + b["no"] == pytest.approx(1.0)
    # BTTS-no must equal P(home=0) + P(away=0) - P(0-0)
    p_h0, p_a0 = grid[0, :].sum(), grid[:, 0].sum()
    assert b["no"] == pytest.approx(p_h0 + p_a0 - grid[0, 0])


def test_top_scorelines_ranked() -> None:
    top = top_scorelines(score_grid(1.5, 1.2, rho=-0.1), n=3)
    assert len(top) == 3
    assert top[0]["prob"] >= top[1]["prob"] >= top[2]["prob"]


def test_fit_rho_recovers_sign_on_synthetic_data() -> None:
    """Sample from a DC grid with rho=-0.15; MLE should land clearly negative."""
    rng = np.random.default_rng(42)
    true_rho, mu_h, mu_a = -0.15, 1.3, 1.1
    grid = score_grid(mu_h, mu_a, rho=true_rho, max_goals=8)
    flat_idx = rng.choice(grid.size, size=4000, p=grid.ravel())
    gh, ga = np.unravel_index(flat_idx, grid.shape)
    est = fit_rho(gh.astype(float), ga.astype(float),
                  np.full(len(gh), mu_h), np.full(len(ga), mu_a))
    assert est == pytest.approx(true_rho, abs=0.06)
    assert est < 0


def test_knockout_advance() -> None:
    adv = knockout_advance(1.8, 0.9, rho=-0.1)
    assert adv["home"] + adv["away"] == pytest.approx(1.0)
    assert adv["home"] > 0.6  # clear favourite advances more often
    # equal teams with fair pens → 50/50
    even = knockout_advance(1.2, 1.2, rho=-0.1, pens_home=0.5)
    assert even["home"] == pytest.approx(0.5, abs=1e-9)

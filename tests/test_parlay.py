"""Property tests: correlated-parlay pricer agrees with grid fundamentals.

Core invariants:
1. A single-leg "parlay" has correlation factor exactly 1.0
2. Joint probability is always <= min(individual marginals) (no cell-set
   intersection can exceed the smallest component)
3. Over 2.5 + Under 2.5 has joint probability 0.0 (mutually exclusive)
4. Home win + Home win = Home win (idempotent intersection)
5. Same-match correlation factors differ from 1.0 for classically correlated
   pairs (home win + over 2.5 should be positively correlated)
6. Cross-match legs are independent (correlation factor ~1.0)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.parlay import ParlayLeg, price_parlay
from src.models.score_grid import btts, outcome_probs, over_under, score_grid

RNG = np.random.default_rng(42)
CASES = [
    (float(RNG.uniform(0.5, 3.0)), float(RNG.uniform(0.5, 3.0)),
     float(RNG.uniform(-0.15, 0.05)))
    for _ in range(15)
]


@pytest.mark.parametrize("mu_h,mu_a,rho", CASES)
def test_single_leg_correlation_is_one(mu_h, mu_a, rho) -> None:
    """A one-leg parlay has correlation factor 1.0 by definition."""
    leg = ParlayLeg(match_id="A", leg_type="outcome", selection="home")
    result = price_parlay([leg], {"A": (mu_h, mu_a)}, rho=rho)
    assert result.correlation_factor == pytest.approx(1.0, abs=1e-12)
    assert result.joint_probability == pytest.approx(
        result.independent_probability, abs=1e-12
    )


@pytest.mark.parametrize("mu_h,mu_a,rho", CASES)
def test_joint_leq_min_marginal(mu_h, mu_a, rho) -> None:
    """Joint probability cannot exceed the smallest individual marginal."""
    legs = [
        ParlayLeg(match_id="A", leg_type="outcome", selection="home"),
        ParlayLeg(match_id="A", leg_type="over", selection="over", line=2.5),
    ]
    result = price_parlay(legs, {"A": (mu_h, mu_a)}, rho=rho)
    min_marg = min(lr.marginal_prob for lr in result.legs)
    assert result.joint_probability <= min_marg + 1e-9


@pytest.mark.parametrize("mu_h,mu_a,rho", CASES)
def test_mutually_exclusive_legs_are_zero(mu_h, mu_a, rho) -> None:
    """Over 2.5 + Under 2.5 is mutually exclusive -> joint = 0."""
    legs = [
        ParlayLeg(match_id="A", leg_type="over", selection="over", line=2.5),
        ParlayLeg(match_id="A", leg_type="under", selection="under", line=2.5),
    ]
    result = price_parlay(legs, {"A": (mu_h, mu_a)}, rho=rho)
    assert result.joint_probability == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("mu_h,mu_a,rho", CASES)
def test_idempotent_intersection(mu_h, mu_a, rho) -> None:
    """home win + home win = home win (same mask, joint = marginal)."""
    legs = [
        ParlayLeg(match_id="A", leg_type="outcome", selection="home"),
        ParlayLeg(match_id="A", leg_type="outcome", selection="home"),
    ]
    result = price_parlay(legs, {"A": (mu_h, mu_a)}, rho=rho)
    grid = score_grid(mu_h, mu_a, rho)
    expected = outcome_probs(grid)["home"]
    assert result.joint_probability == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("mu_h,mu_a,rho", CASES[:5])
def test_marginals_agree_with_grid_functions(mu_h, mu_a, rho) -> None:
    """Per-leg marginals from the parlay pricer match score_grid helpers."""
    legs = [
        ParlayLeg(match_id="M", leg_type="outcome", selection="home"),
        ParlayLeg(match_id="M", leg_type="over", selection="over", line=2.5),
        ParlayLeg(match_id="M", leg_type="btts", selection="yes"),
    ]
    result = price_parlay(legs, {"M": (mu_h, mu_a)}, rho=rho)
    grid = score_grid(mu_h, mu_a, rho)

    assert result.legs[0].marginal_prob == pytest.approx(
        outcome_probs(grid)["home"], abs=1e-9
    )
    assert result.legs[1].marginal_prob == pytest.approx(
        over_under(grid, 2.5)["over"], abs=1e-9
    )
    assert result.legs[2].marginal_prob == pytest.approx(
        btts(grid)["yes"], abs=1e-9
    )


def test_home_win_plus_over_positively_correlated() -> None:
    """Classic positive correlation: if the home team wins, total goals are
    likely above 2.5 because the home team scored.  The correlation factor
    should be >1 for reasonable match parameters."""
    mu_h, mu_a, rho = 1.8, 1.2, -0.05
    legs = [
        ParlayLeg(match_id="A", leg_type="outcome", selection="home"),
        ParlayLeg(match_id="A", leg_type="over", selection="over", line=2.5),
    ]
    result = price_parlay(legs, {"A": (mu_h, mu_a)}, rho=rho)
    assert result.correlation_factor > 1.0, (
        f"expected positive correlation, got {result.correlation_factor}"
    )


def test_home_win_plus_under_negatively_correlated() -> None:
    """If the home team wins *and* under 2.5, the home team won 1-0 or 2-0
    (small set).  Joint is less than independent -> negative correlation."""
    mu_h, mu_a, rho = 1.8, 1.2, -0.05
    legs = [
        ParlayLeg(match_id="A", leg_type="outcome", selection="home"),
        ParlayLeg(match_id="A", leg_type="under", selection="under", line=2.5),
    ]
    result = price_parlay(legs, {"A": (mu_h, mu_a)}, rho=rho)
    assert result.correlation_factor < 1.0, (
        f"expected negative correlation, got {result.correlation_factor}"
    )


def test_cross_match_independence() -> None:
    """Legs from different matches should have correlation factor ~1.0."""
    legs = [
        ParlayLeg(match_id="A", leg_type="outcome", selection="home"),
        ParlayLeg(match_id="B", leg_type="outcome", selection="away"),
    ]
    mus = {"A": (1.5, 1.2), "B": (1.0, 1.8)}
    result = price_parlay(legs, mus, rho=-0.05)
    # overall correlation = product of within-match correlations, each is 1.0
    assert result.correlation_factor == pytest.approx(1.0, abs=1e-9)


def test_player_scorer_leg() -> None:
    """A player with a non-zero xG share has positive anytime-scorer probability."""
    legs = [
        ParlayLeg(
            match_id="A", leg_type="anytime_scorer", selection="Saka",
            team_side="home", xg_share=0.18,
        ),
    ]
    result = price_parlay(legs, {"A": (1.8, 1.0)}, rho=-0.05)
    assert 0 < result.joint_probability < 1.0
    assert result.correlation_factor == pytest.approx(1.0, abs=1e-12)


def test_scorer_plus_over_positively_correlated() -> None:
    """Player scores + over 2.5 should be positively correlated -- more goals
    means more chances for the player to score."""
    legs = [
        ParlayLeg(
            match_id="A", leg_type="anytime_scorer", selection="Saka",
            team_side="home", xg_share=0.18,
        ),
        ParlayLeg(match_id="A", leg_type="over", selection="over", line=2.5),
    ]
    result = price_parlay(legs, {"A": (1.8, 1.0)}, rho=-0.05)
    assert result.correlation_factor > 1.0


@pytest.mark.parametrize("mu_h,mu_a,rho", CASES[:5])
def test_exact_score_is_grid_cell(mu_h, mu_a, rho) -> None:
    """An exact-score leg marginal should match the grid cell directly."""
    leg = ParlayLeg(match_id="A", leg_type="exact_score", selection="1-1")
    result = price_parlay([leg], {"A": (mu_h, mu_a)}, rho=rho)
    grid = score_grid(mu_h, mu_a, rho)
    assert result.legs[0].marginal_prob == pytest.approx(
        float(grid[1, 1]), abs=1e-9
    )


def test_three_leg_same_match() -> None:
    """Home win + Over 2.5 + BTTS yes -- a triple from one match."""
    legs = [
        ParlayLeg(match_id="A", leg_type="outcome", selection="home"),
        ParlayLeg(match_id="A", leg_type="over", selection="over", line=2.5),
        ParlayLeg(match_id="A", leg_type="btts", selection="yes"),
    ]
    result = price_parlay(legs, {"A": (2.0, 1.5)}, rho=-0.05)
    # joint must be positive and < min marginal
    assert result.joint_probability > 0.0
    assert result.joint_probability <= min(lr.marginal_prob for lr in result.legs) + 1e-9
    # three positively-correlated legs -> overall correlation > 1
    assert result.match_groups[0].correlation_factor > 1.0


def test_sportsbook_odds_comparison() -> None:
    """When decimal odds are provided, the pricer computes edge and EV."""
    legs = [
        ParlayLeg(match_id="A", leg_type="outcome", selection="home", decimal_odds=2.10),
        ParlayLeg(match_id="A", leg_type="over", selection="over", line=2.5, decimal_odds=1.85),
    ]
    result = price_parlay(legs, {"A": (1.8, 1.2)}, rho=-0.05)
    assert result.sportsbook_parlay_odds == pytest.approx(2.10 * 1.85, abs=1e-6)
    assert result.model_fair_odds > 0
    # edge should be non-zero (model and sportsbook differ)
    assert result.edge != 0.0

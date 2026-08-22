"""Tests for the population tournament runner."""

from __future__ import annotations

import numpy as np
import pytest

from envs.real_market import Gameweek, iter_gameweeks, load_season
from envs.tournament import (
    TournamentReport,
    _bootstrap_ci,
    is_distinguishable,
    run_tournament,
)
from pathlib import Path
from training.evaluate import AbstainerPolicy, FavouritePolicy, RandomPolicy

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "football_data_uk"


@pytest.fixture(scope="module")
def gameweeks():
    fixtures = load_season(_DATA_DIR / "E0_2425.csv")
    return list(iter_gameweeks(fixtures))[:5]


def test_bootstrap_ci_contains_mean():
    """Bootstrap CI must contain the sample mean."""
    values = list(np.random.default_rng(0).normal(0, 1, 50))
    mean, lo, hi = _bootstrap_ci(values)
    assert lo <= mean <= hi


def test_bootstrap_ci_empty():
    """Empty list produces zero CI."""
    mean, lo, hi = _bootstrap_ci([])
    assert mean == 0.0


def test_is_distinguishable_identical():
    """Two identical samples should not be distinguishable."""
    values = [0.01] * 50
    result = is_distinguishable(values, values)
    assert result.distinguishable is False


def test_is_distinguishable_different():
    """Well-separated distributions should be distinguishable."""
    a = [0.10 + x * 0.001 for x in range(50)]
    b = [-0.10 + x * 0.001 for x in range(50)]
    result = is_distinguishable(a, b)
    assert result.distinguishable is True
    assert result.p_value < 0.05


def test_tournament_leaderboard(gameweeks):
    """Tournament must produce a leaderboard sorted by mean CLV."""
    policies = [
        ("Favourite", FavouritePolicy()),
        ("Abstainer", AbstainerPolicy()),
    ]
    report = run_tournament(policies, gameweeks, n_seeds=5, max_gw_per_seed=3, n_bootstrap=500)
    assert len(report.leaderboard) == 2
    assert report.leaderboard[0].mean_clv >= report.leaderboard[1].mean_clv


def test_tournament_pairwise(gameweeks):
    """Tournament must produce pairwise significance results."""
    policies = [
        ("Favourite", FavouritePolicy()),
        ("Abstainer", AbstainerPolicy()),
    ]
    report = run_tournament(policies, gameweeks, n_seeds=5, max_gw_per_seed=3, n_bootstrap=500)
    assert len(report.pairwise) == 1
    assert report.pairwise[0].policy_a in ("Favourite", "Abstainer")
    assert report.pairwise[0].policy_b in ("Favourite", "Abstainer")


def test_tournament_table_format(gameweeks):
    """Leaderboard table must contain all required fields."""
    policies = [
        ("Favourite", FavouritePolicy()),
        ("Abstainer", AbstainerPolicy()),
    ]
    report = run_tournament(policies, gameweeks, n_seeds=5, max_gw_per_seed=2, n_bootstrap=500)
    table = report.leaderboard_table()
    assert len(table) == 2
    required = {"rank", "policy", "mean_clv", "clv_95ci", "mean_reward", "reward_95ci", "total_bets", "n_runs"}
    for row in table:
        assert required.issubset(row.keys())


def test_significance_matrix(gameweeks):
    """Significance matrix must be symmetric and complete."""
    policies = [
        ("A", AbstainerPolicy()),
        ("B", FavouritePolicy()),
        ("C", RandomPolicy(seed=0)),
    ]
    report = run_tournament(policies, gameweeks, n_seeds=5, max_gw_per_seed=2, n_bootstrap=500)
    matrix = report.significance_matrix()
    names = [s.name for s in report.leaderboard]
    for name in names:
        assert name in matrix
        assert matrix[name][name] == "-"


def test_tournament_n_seeds(gameweeks):
    """Each policy must be evaluated n_seeds times."""
    policies = [("Abstainer", AbstainerPolicy())]
    report = run_tournament(policies, gameweeks, n_seeds=10, max_gw_per_seed=2, n_bootstrap=500)
    assert report.leaderboard[0].n_runs == 10
    assert report.n_seeds == 10

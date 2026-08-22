"""Tests for the tunable-efficiency simulated market generator."""

from __future__ import annotations

import numpy as np
import pytest

from envs.sim_market import (
    MarketConfig,
    generate_fixture_batch,
    generate_season,
    sim_clv_distribution,
)


def test_generate_season_shape():
    """Season must have the configured number of gameweeks."""
    cfg = MarketConfig(eta=0.85, seed=0, n_matches_per_gw=10)
    season = generate_season(config=cfg, n_gameweeks=38)
    assert len(season.gameweeks) == 38
    assert len(season.true_probs) == 38


def test_generate_season_fixtures_per_gw():
    """Each gameweek must have the configured number of fixtures."""
    cfg = MarketConfig(eta=0.85, seed=0, n_matches_per_gw=10)
    season = generate_season(config=cfg, n_gameweeks=5)
    for gw in season.gameweeks:
        assert len(gw.fixtures) == 10


def test_fixture_odds_range():
    """All odds must be in [1.01, 100]."""
    fixtures = generate_fixture_batch(50, eta=0.5, seed=0)
    for fix in fixtures:
        for odds in [fix.odds_h, fix.odds_d, fix.odds_a,
                     fix.close_h, fix.close_d, fix.close_a]:
            assert 1.01 <= odds <= 100.0, f"Odds out of range: {odds}"


def test_fixture_outcomes_valid():
    """Outcomes must be H, D, or A."""
    fixtures = generate_fixture_batch(100, eta=0.7, seed=1)
    for fix in fixtures:
        assert fix.ftr in ("H", "D", "A")


def test_eta_1_closing_near_truth():
    """At eta=1.0, closing odds should closely reflect true probabilities."""
    cfg = MarketConfig(eta=1.0, seed=42, n_matches_per_gw=100)
    season = generate_season(config=cfg, n_gameweeks=1)
    for fix, true_p in zip(season.gameweeks[0].fixtures, season.true_probs[0]):
        close_implied = np.array([1/fix.close_h, 1/fix.close_d, 1/fix.close_a])
        close_implied = close_implied / close_implied.sum()
        np.testing.assert_allclose(close_implied, true_p, atol=0.08)


def test_high_eta_closing_more_informative():
    """Higher eta means closing is closer to truth, so |CLV| is larger."""
    dist_high = sim_clv_distribution(eta=1.0, n_matches=500, seed=42)
    dist_low = sim_clv_distribution(eta=0.0, n_matches=500, seed=42)
    assert dist_high["mean_abs_clv"] > dist_low["mean_abs_clv"]


def test_clv_distribution_keys():
    """CLV distribution must contain all required keys."""
    dist = sim_clv_distribution(eta=0.85, n_matches=100, seed=0)
    required = {"eta", "mean_clv", "std_clv", "mean_abs_clv", "median_clv", "pct_positive", "n"}
    assert required.issubset(dist.keys())


def test_deterministic_with_seed():
    """Same seed must produce identical fixtures."""
    a = generate_fixture_batch(10, eta=0.85, seed=42)
    b = generate_fixture_batch(10, eta=0.85, seed=42)
    for fa, fb in zip(a, b):
        assert fa.odds_h == fb.odds_h
        assert fa.ftr == fb.ftr


def test_different_seeds_differ():
    """Different seeds must produce different fixtures."""
    a = generate_fixture_batch(10, eta=0.85, seed=1)
    b = generate_fixture_batch(10, eta=0.85, seed=2)
    any_differ = any(fa.odds_h != fb.odds_h for fa, fb in zip(a, b))
    assert any_differ


def test_sim_source_tag():
    """Simulated fixtures must be tagged with odds_source='sim'."""
    fixtures = generate_fixture_batch(5, eta=0.85, seed=0)
    for fix in fixtures:
        assert fix.odds_source == "sim"


def test_flb_compresses_probabilities():
    """Favourite-longshot bias should compress odds toward equal."""
    cfg_no_flb = MarketConfig(eta=0.85, flb_strength=0.0, seed=42, n_matches_per_gw=50)
    cfg_flb = MarketConfig(eta=0.85, flb_strength=0.5, seed=42, n_matches_per_gw=50)
    season_no = generate_season(config=cfg_no_flb, n_gameweeks=1)
    season_flb = generate_season(config=cfg_flb, n_gameweeks=1)

    var_no = np.var([f.odds_h for f in season_no.gameweeks[0].fixtures])
    var_flb = np.var([f.odds_h for f in season_flb.gameweeks[0].fixtures])
    assert var_flb < var_no

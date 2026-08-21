"""Tests for the sim-to-real transfer measurement module."""

from __future__ import annotations

import numpy as np
import pytest

from evals.sim_to_real import (
    CalibrationSnapshot,
    _calibration_snapshot,
    _fixtures_to_arrays,
    measure_market_transfer,
)
from envs.real_market import load_season
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "football_data_uk"


# ── Fixture-to-array conversion ──────────────────────────────────────

def test_fixtures_to_arrays_shape():
    """Probability and label arrays must have consistent shapes."""
    fixtures = load_season(_DATA_DIR / "E0_2425.csv")
    probs, labels = _fixtures_to_arrays(fixtures)
    assert probs.shape[0] == labels.shape[0]
    assert probs.shape[1] == 3
    assert labels.shape[0] > 0


def test_fixtures_to_arrays_probs_sum_to_one():
    """De-vigged probabilities must sum to 1 for each match."""
    fixtures = load_season(_DATA_DIR / "E0_2425.csv")
    probs, _ = _fixtures_to_arrays(fixtures)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-7)


def test_fixtures_to_arrays_labels_valid():
    """Labels must be in {0, 1, 2} (home, draw, away)."""
    fixtures = load_season(_DATA_DIR / "E0_2425.csv")
    _, labels = _fixtures_to_arrays(fixtures)
    assert all(l in (0, 1, 2) for l in labels)


# ── Calibration snapshot ─────────────────────────────────────────────

def test_calibration_snapshot_metrics():
    """Snapshot metrics must be finite and within valid ranges."""
    fixtures = load_season(_DATA_DIR / "E0_2425.csv")
    snap = _calibration_snapshot("test", fixtures)
    assert snap.n_matches > 0
    assert 0 < snap.log_loss < 5.0
    assert 0 < snap.brier_score < 2.0
    assert 0 <= snap.ece <= 1.0


def test_calibration_snapshot_empty():
    """Empty fixture list produces zeroed metrics."""
    snap = _calibration_snapshot("empty", [])
    assert snap.n_matches == 0
    assert snap.log_loss == 0.0


# ── Market transfer ─────────────────────────────────────────────────

def test_market_transfer_report():
    """Market transfer report must compare sim vs real calibration."""
    report = measure_market_transfer(n_sim_seasons=4)
    assert report.sim.n_matches > 0
    assert report.real.n_matches > 0
    assert np.isfinite(report.ece_drift)
    assert np.isfinite(report.brier_drift)


def test_market_transfer_drift_direction():
    """Drift values must be the difference real - sim."""
    report = measure_market_transfer(n_sim_seasons=4)
    expected_ece_drift = report.real.ece - report.sim.ece
    assert abs(report.ece_drift - expected_ece_drift) < 1e-8


def test_market_transfer_summary():
    """Summary dict must contain all required keys."""
    report = measure_market_transfer(n_sim_seasons=4)
    summary = report.summary()
    required = {"sim_ece", "real_ece", "ece_drift", "sim_brier", "real_brier", "brier_drift"}
    assert required.issubset(summary.keys())


# ── Environment-level transfer ───────────────────────────────────────

def test_environment_transfer_with_abstainer():
    """Abstainer policy should have transfer ratio near 1.0."""
    from evals.sim_to_real import measure_environment_transfer
    from training.evaluate import AbstainerPolicy

    report = measure_environment_transfer(
        policies=[("Abstainer", AbstainerPolicy())],
        n_sim_seasons=4,
        max_gw=3,
    )

    assert len(report.policies) == 1
    p = report.policies[0]
    assert p.sim_n_bets == 0
    assert p.real_n_bets == 0
    assert abs(p.transfer_ratio - 1.0) < 0.1


def test_environment_transfer_with_favourite():
    """Favourite policy should produce a finite transfer ratio."""
    from evals.sim_to_real import measure_environment_transfer
    from training.evaluate import FavouritePolicy

    report = measure_environment_transfer(
        policies=[("Favourite", FavouritePolicy())],
        n_sim_seasons=4,
        max_gw=3,
    )

    assert len(report.policies) == 1
    p = report.policies[0]
    assert p.sim_n_bets > 0
    assert p.real_n_bets > 0
    assert np.isfinite(p.transfer_ratio)


def test_environment_transfer_summary_table():
    """Summary table must have one row per policy."""
    from evals.sim_to_real import measure_environment_transfer
    from training.evaluate import AbstainerPolicy, FavouritePolicy

    report = measure_environment_transfer(
        policies=[
            ("Abstainer", AbstainerPolicy()),
            ("Favourite", FavouritePolicy()),
        ],
        n_sim_seasons=4,
        max_gw=2,
    )

    table = report.summary_table()
    assert len(table) == 2
    assert table[0]["policy"] == "Abstainer"
    assert table[1]["policy"] == "Favourite"

"""Unit tests: splits, metrics, walk-forward backtest, ROI simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.backtest import run_backtest, simulate_roi
from src.eval.metrics import (
    brier,
    conformal_coverage,
    expected_calibration_error,
    log_loss,
    reliability_table,
    rps,
)
from src.eval.splits import walk_forward_folds


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# ------------------------------------------------------------------- splits

def test_walk_forward_ordering_and_no_overlap() -> None:
    df = pd.DataFrame({
        "kickoff_utc": [_ts("2021-08-01"), _ts("2022-08-01"), _ts("2023-08-01")],
        "season": ["21", "22", "23"],
    })
    folds = list(walk_forward_folds(df))
    assert [f.group for f in folds] == ["22", "23"]
    assert len(folds[1].train_idx) == 2  # expanding window


def test_walk_forward_rejects_overlap() -> None:
    df = pd.DataFrame({
        # season 21 runs into June 2022; season 22 starts January 2022
        "kickoff_utc": [_ts("2021-08-01"), _ts("2022-06-01"), _ts("2022-01-01")],
        "season": ["21", "21", "22"],
    })
    with pytest.raises(ValueError, match="temporal overlap"):
        list(walk_forward_folds(df))


# ------------------------------------------------------------------- metrics

def test_log_loss_and_brier_known_values() -> None:
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]])
    y = np.array([0, 1])
    assert log_loss(probs, y) == pytest.approx(-(np.log(0.8) + np.log(0.5)) / 2)
    expected_brier = ((0.04 + 0.01 + 0.01) + (0.04 + 0.25 + 0.09)) / 2
    assert brier(probs, y) == pytest.approx(expected_brier)


def test_rps_rewards_mass_near_outcome() -> None:
    y = np.array([0])
    near = np.array([[0.6, 0.3, 0.1]])
    far = np.array([[0.6, 0.1, 0.3]])  # same top prob, mass further away
    assert rps(near, y) < rps(far, y)


def test_reliability_and_ece_on_calibrated_forecasts() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    outcome = (rng.uniform(size=p.size) < p).astype(float)
    table = reliability_table(p, outcome)
    assert (np.abs(table["mean_pred"] - table["realized"]) < 0.03).all()
    assert expected_calibration_error(p, outcome) < 0.02


def test_conformal_coverage_helper() -> None:
    assert conformal_coverage([[0], [0, 1], [2]], np.array([0, 1, 1])) == pytest.approx(2 / 3)


# ------------------------------------------------------------------ backtest

@pytest.fixture
def toy_history() -> tuple[pd.DataFrame, pd.Series]:
    """Three seasons where the market prob column is almost perfect."""
    rng = np.random.default_rng(5)
    n_per, seasons = 120, ["21", "22", "23"]
    frames, labels = [], []
    for i, s in enumerate(seasons):
        true_home = rng.uniform(0.25, 0.65, n_per)
        draw = np.full(n_per, 0.25)
        away = 1 - true_home - draw
        y_idx = np.array([rng.choice(3, p=[h, d, a])
                          for h, d, a in zip(true_home, draw, away)])
        frames.append(pd.DataFrame({
            "kickoff_utc": pd.date_range(f"20{s}-08-01", periods=n_per,
                                         freq="D", tz="UTC"),
            "season": s,
            "signal": true_home + rng.normal(0, 0.05, n_per),
            "odds_imp_home": true_home, "odds_imp_draw": draw, "odds_imp_away": away,
        }))
        labels.append(pd.Series(np.array(["home", "draw", "away"])[y_idx]))
    return (pd.concat(frames, ignore_index=True),
            pd.concat(labels, ignore_index=True))


def _naive_fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Logistic-ish toy model on the signal column."""
    h = np.clip(test["signal"].to_numpy(), 0.05, 0.9)
    d = np.full(len(test), 0.25)
    return np.column_stack([h, d, np.clip(1 - h - d, 0.01, None)])


def test_backtest_market_beats_noisy_model(toy_history) -> None:
    features, y = toy_history
    report = run_backtest(features, y, _naive_fit_predict)
    assert [f.group for f in report.folds] == ["22", "23"]
    summary = report.summary()
    # market column is the truth here; it must win on log loss
    assert summary.loc["market", "logloss"] <= summary.loc["model", "logloss"]
    assert summary.loc["market", "logloss"] <= summary.loc["baseline", "logloss"]


def test_simulate_roi_settles_known_bets() -> None:
    """Two matches, one clear +EV home edge each; home wins one, loses one."""
    preds = pd.DataFrame({
        "model_home": [0.60, 0.60], "model_draw": [0.20, 0.20], "model_away": [0.20, 0.20],
        "market_home": [0.45, 0.45], "market_draw": [0.28, 0.28], "market_away": [0.27, 0.27],
        "odds_home": [2.10, 2.10], "odds_draw": [3.4, 3.4], "odds_away": [3.6, 3.6],
        "y_idx": [0, 2],  # home wins match 1, away wins match 2
    })
    res = simulate_roi(preds, ev_threshold=0.03)
    assert res["n_bets"] == 2               # only the home edges clear EV
    assert res["pnl"] == pytest.approx(1.10 - 1.0)
    assert res["roi"] == pytest.approx(0.05)
    assert res["hit_rate"] == pytest.approx(0.5)

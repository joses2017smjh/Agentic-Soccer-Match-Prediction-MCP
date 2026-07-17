"""Evaluation metrics: log loss, Brier, RPS, reliability, conformal coverage.

All probability arrays are (n, k) with columns in a fixed class order
(home/draw/away for outcomes); ``y_idx`` holds integer class labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def log_loss(probs: np.ndarray, y_idx: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y_idx)), y_idx], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Multiclass Brier: mean squared distance to the one-hot outcome."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def rps(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Ranked probability score over *ordered* classes.

    For match outcomes use the order (home, draw, away); for scoreline-derived
    totals use ascending goal counts. Lower is better; RPS punishes putting
    mass far from the outcome in rank space, which log loss ignores.
    """
    k = probs.shape[1]
    cum_p = np.cumsum(probs, axis=1)[:, : k - 1]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    cum_o = np.cumsum(onehot, axis=1)[:, : k - 1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (k - 1)))


def reliability_table(
    prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Binary reliability curve data: per-bin mean forecast vs realized rate.

    Perfectly calibrated forecasts have mean_pred ≈ realized in every bin.
    """
    bins = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        rows.append({
            "bin_low": b / n_bins,
            "bin_high": (b + 1) / n_bins,
            "mean_pred": float(prob[mask].mean()),
            "realized": float(outcome[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(
    prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10
) -> float:
    table = reliability_table(prob, outcome, n_bins)
    weights = table["count"] / table["count"].sum()
    return float((weights * (table["mean_pred"] - table["realized"]).abs()).sum())


def conformal_coverage(sets: list[list[int]], y_idx: np.ndarray) -> float:
    """Fraction of outcomes inside their prediction set; report next to 1-α."""
    return float(np.mean([y in s for y, s in zip(y_idx, sets)]))

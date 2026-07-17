"""Probability calibration and split-conformal prediction sets.

``IsotonicCalibrator`` — per-class isotonic regression (one-vs-rest) fit on a
temporally held-out calibration slice, renormalized to sum to 1. Isotonic is
preferred over Platt here because GBM miscalibration is not sigmoidal and the
calibration slice is large enough to avoid isotonic's small-sample overfit.

``ConformalWrapper`` — split conformal prediction (Angelopoulos & Bates 2023).
Nonconformity score s_i = 1 − p̂(true class). With calibration scores
s_(1..n), the quantile q̂ at level ⌈(n+1)(1−α)⌉/n yields prediction sets
{k : p̂_k ≥ 1 − q̂} with distribution-free marginal coverage ≥ 1 − α.
A set like {home, draw} tells the agent (Phase B) the model cannot separate
those outcomes at the requested confidence — that uncertainty must be
surfaced, not overclaimed.

Both objects are part of the versioned model artifact.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class IsotonicCalibrator:
    classes: tuple[str, ...] = ("home", "draw", "away")
    _models: list[IsotonicRegression] = field(default_factory=list)

    def fit(self, raw_probs: np.ndarray, y_idx: np.ndarray) -> "IsotonicCalibrator":
        """raw_probs: (n, k) model outputs; y_idx: integer class labels."""
        self._models = []
        for k in range(raw_probs.shape[1]):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw_probs[:, k], (y_idx == k).astype(float))
            self._models.append(iso)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        assert self._models, "call fit() first"
        cal = np.column_stack(
            [m.predict(raw_probs[:, k]) for k, m in enumerate(self._models)]
        )
        cal = np.clip(cal, 1e-6, 1.0)
        return cal / cal.sum(axis=1, keepdims=True)


@dataclass
class ConformalWrapper:
    alpha: float = 0.1
    q_hat: float | None = None

    def fit(self, cal_probs: np.ndarray, y_idx: np.ndarray) -> "ConformalWrapper":
        """cal_probs must come from data unseen by both GBM and calibrator."""
        n = len(y_idx)
        scores = 1.0 - cal_probs[np.arange(n), y_idx]
        level = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        self.q_hat = float(np.quantile(scores, min(level, 1.0), method="higher"))
        return self

    def prediction_set(self, probs: np.ndarray) -> list[list[int]]:
        """Class indices whose probability clears the conformal threshold.
        Never empty: the argmax class is always included."""
        assert self.q_hat is not None, "call fit() first"
        sets: list[list[int]] = []
        for row in np.atleast_2d(probs):
            included = [k for k, p in enumerate(row) if p >= 1.0 - self.q_hat]
            if not included:
                included = [int(np.argmax(row))]
            sets.append(included)
        return sets

    def empirical_coverage(self, probs: np.ndarray, y_idx: np.ndarray) -> float:
        sets = self.prediction_set(probs)
        return float(np.mean([y in s for y, s in zip(y_idx, sets)]))


@dataclass
class CalibratedOutcomeHead:
    """Calibrator + conformal wrapper bundled as one artifact component."""

    calibrator: IsotonicCalibrator
    conformal: ConformalWrapper

    def predict(self, raw_probs: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
        cal = self.calibrator.transform(raw_probs)
        return cal, self.conformal.prediction_set(cal)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "outcome_head.pkl").write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "CalibratedOutcomeHead":
        return pickle.loads((path / "outcome_head.pkl").read_bytes())

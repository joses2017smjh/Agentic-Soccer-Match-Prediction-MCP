"""Probability calibration and split-conformal prediction sets.

Calibration ladder (B2):
``IsotonicCalibrator`` — per-class isotonic regression (incumbent)
``TemperatureCalibrator`` — 1-parameter softmax rescaling (Guo et al. 2017)
``VectorCalibrator`` — per-class scale + shift (6 params, Guo et al. 2017)
``DirichletCalibrator`` — ODIR-regularized Dirichlet (Kull et al. 2019)
``select_calibrator`` — pick the best on a validation fold by log loss

Conformal prediction sets (B7):
``ConformalWrapper`` — split conformal (Angelopoulos & Bates 2023)
``AdaptiveConformal`` — ACI with online alpha adjustment (Gibbs & Candes 2021)

Both calibrator and conformal wrapper are part of the versioned model artifact.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# Calibrator protocol
# ---------------------------------------------------------------------------

class Calibrator(Protocol):
    name: str

    def fit(self, raw_probs: np.ndarray, y_idx: np.ndarray) -> "Calibrator": ...
    def transform(self, raw_probs: np.ndarray) -> np.ndarray: ...


def _normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-8, None)
    return probs / probs.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# B2: Calibration ladder
# ---------------------------------------------------------------------------


@dataclass
class IsotonicCalibrator:
    name: str = "isotonic"
    classes: tuple[str, ...] = ("home", "draw", "away")
    _models: list[IsotonicRegression] = field(default_factory=list)

    def fit(self, raw_probs: np.ndarray, y_idx: np.ndarray) -> "IsotonicCalibrator":
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
        return _normalize(cal)


@dataclass
class TemperatureCalibrator:
    """Single-parameter temperature scaling (Guo et al. 2017).
    Divides logits by T before softmax; T > 1 softens, T < 1 sharpens."""

    name: str = "temperature"
    temperature: float = 1.0

    def fit(self, raw_probs: np.ndarray, y_idx: np.ndarray) -> "TemperatureCalibrator":
        logits = np.log(np.clip(raw_probs, 1e-8, None))

        def nll(t: np.ndarray) -> float:
            scaled = logits / max(t[0], 1e-4)
            exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            softmax = exp_s / exp_s.sum(axis=1, keepdims=True)
            return float(-np.mean(np.log(np.clip(
                softmax[np.arange(len(y_idx)), y_idx], 1e-12, None,
            ))))

        result = minimize(nll, x0=[1.0], bounds=[(0.01, 10.0)], method="L-BFGS-B")
        self.temperature = float(result.x[0])
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(raw_probs, 1e-8, None))
        scaled = logits / max(self.temperature, 1e-4)
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return _normalize(exp_s)


@dataclass
class VectorCalibrator:
    """Per-class scale and shift (6 params for 3 classes; Guo et al. 2017)."""

    name: str = "vector"
    _w: np.ndarray | None = None
    _b: np.ndarray | None = None

    def fit(self, raw_probs: np.ndarray, y_idx: np.ndarray) -> "VectorCalibrator":
        k = raw_probs.shape[1]
        logits = np.log(np.clip(raw_probs, 1e-8, None))

        def nll(params: np.ndarray) -> float:
            w, b = params[:k], params[k:]
            scaled = logits * w + b
            exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            softmax = exp_s / exp_s.sum(axis=1, keepdims=True)
            return float(-np.mean(np.log(np.clip(
                softmax[np.arange(len(y_idx)), y_idx], 1e-12, None,
            ))))

        x0 = np.concatenate([np.ones(k), np.zeros(k)])
        result = minimize(nll, x0=x0, method="L-BFGS-B")
        self._w = result.x[:k]
        self._b = result.x[k:]
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        assert self._w is not None, "call fit() first"
        logits = np.log(np.clip(raw_probs, 1e-8, None))
        scaled = logits * self._w + self._b
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return _normalize(exp_s)


@dataclass
class DirichletCalibrator:
    """Dirichlet calibration with ODIR regularization (Kull et al. 2019).
    Full k x k matrix + bias (12 params for 3 classes)."""

    name: str = "dirichlet"
    reg_lambda: float = 1e-3
    _W: np.ndarray | None = None
    _b: np.ndarray | None = None

    def fit(self, raw_probs: np.ndarray, y_idx: np.ndarray) -> "DirichletCalibrator":
        k = raw_probs.shape[1]
        logits = np.log(np.clip(raw_probs, 1e-8, None))

        def nll(params: np.ndarray) -> float:
            W = params[: k * k].reshape(k, k)
            b = params[k * k:]
            scaled = logits @ W + b
            exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            softmax = exp_s / exp_s.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(np.clip(
                softmax[np.arange(len(y_idx)), y_idx], 1e-12, None,
            )))
            # ODIR: off-diagonal L2 regularization
            off_diag = W.copy()
            np.fill_diagonal(off_diag, 0.0)
            loss += self.reg_lambda * np.sum(off_diag ** 2)
            return float(loss)

        x0 = np.concatenate([np.eye(k).ravel(), np.zeros(k)])
        result = minimize(nll, x0=x0, method="L-BFGS-B")
        self._W = result.x[: k * k].reshape(k, k)
        self._b = result.x[k * k:]
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        assert self._W is not None, "call fit() first"
        logits = np.log(np.clip(raw_probs, 1e-8, None))
        scaled = logits @ self._W + self._b
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return _normalize(exp_s)


def select_calibrator(
    raw_probs: np.ndarray,
    y_idx: np.ndarray,
    val_probs: np.ndarray,
    val_y_idx: np.ndarray,
) -> Calibrator:
    """Fit all calibrators on train, select the one with best val log loss."""
    from src.eval.metrics import log_loss

    candidates: list[Calibrator] = [
        IsotonicCalibrator(),
        TemperatureCalibrator(),
        VectorCalibrator(),
        DirichletCalibrator(),
    ]
    best: Calibrator | None = None
    best_ll = float("inf")
    for cal in candidates:
        cal.fit(raw_probs, y_idx)
        cal_val = cal.transform(val_probs)
        ll = log_loss(cal_val, val_y_idx)
        if ll < best_ll:
            best_ll = ll
            best = cal
    assert best is not None
    return best


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
class AdaptiveConformal:
    """Adaptive Conformal Inference (Gibbs & Candes 2021).

    Maintains a running alpha_t that adjusts online: when the prediction set
    covers, alpha_t increases (tighter sets); when it misses, alpha_t decreases
    (wider sets). Over time this corrects for temporal drift that breaks the
    exchangeability assumption static conformal relies on.

    gamma controls the adaptation speed (learning rate for alpha updates).
    """

    alpha: float = 0.1
    gamma: float = 0.005
    q_hat: float | None = None
    _alpha_t: float | None = None
    _cal_scores: np.ndarray | None = None

    def fit(self, cal_probs: np.ndarray, y_idx: np.ndarray) -> "AdaptiveConformal":
        n = len(y_idx)
        self._cal_scores = 1.0 - cal_probs[np.arange(n), y_idx]
        self._alpha_t = self.alpha
        level = math.ceil((n + 1) * (1.0 - self._alpha_t)) / n
        self.q_hat = float(np.quantile(
            self._cal_scores, min(level, 1.0), method="higher",
        ))
        return self

    def prediction_set(self, probs: np.ndarray) -> list[list[int]]:
        assert self.q_hat is not None, "call fit() first"
        sets: list[list[int]] = []
        for row in np.atleast_2d(probs):
            included = [k for k, p in enumerate(row) if p >= 1.0 - self.q_hat]
            if not included:
                included = [int(np.argmax(row))]
            sets.append(included)
        return sets

    def update(self, prob_row: np.ndarray, y: int) -> None:
        """Online update: adjust alpha_t based on whether we covered."""
        assert self._alpha_t is not None and self._cal_scores is not None
        s = self.prediction_set(np.atleast_2d(prob_row))
        err = 1.0 if y not in s[0] else 0.0
        self._alpha_t = self._alpha_t + self.gamma * (self.alpha - err)
        self._alpha_t = float(np.clip(self._alpha_t, 0.001, 0.999))
        n = len(self._cal_scores)
        level = math.ceil((n + 1) * (1.0 - self._alpha_t)) / n
        self.q_hat = float(np.quantile(
            self._cal_scores, min(level, 1.0), method="higher",
        ))

    def update_batch(self, probs: np.ndarray, y_idx: np.ndarray) -> float:
        """Process a batch sequentially (temporal order matters). Returns
        the empirical coverage over this batch."""
        covered = 0
        for i in range(len(y_idx)):
            s = self.prediction_set(probs[i:i+1])
            if y_idx[i] in s[0]:
                covered += 1
            self.update(probs[i], y_idx[i])
        return covered / max(len(y_idx), 1)

    def empirical_coverage(self, probs: np.ndarray, y_idx: np.ndarray) -> float:
        sets = self.prediction_set(probs)
        return float(np.mean([y in s for y, s in zip(y_idx, sets)]))


@dataclass
class CalibratedOutcomeHead:
    """Calibrator + conformal wrapper bundled as one artifact component."""

    calibrator: IsotonicCalibrator | TemperatureCalibrator | VectorCalibrator | DirichletCalibrator
    conformal: ConformalWrapper | AdaptiveConformal

    def predict(self, raw_probs: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
        cal = self.calibrator.transform(raw_probs)
        return cal, self.conformal.prediction_set(cal)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "outcome_head.pkl").write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "CalibratedOutcomeHead":
        return pickle.loads((path / "outcome_head.pkl").read_bytes())

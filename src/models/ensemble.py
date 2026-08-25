"""Log-opinion pool ensemble (B6).

Combines multiple probability forecasters in log space with learned weights.
Members: market prior, residual GBM, grid-derived (Dixon-Coles from Elo xG).
Weights are fit on a validation fold by minimizing log loss.

The log-opinion pool multiplies probability vectors element-wise (in log space)
with exponent weights, then renormalizes. This is the natural combination rule
for probability forecasters and preserves the sharpness of confident members
better than linear pooling.

    p_ensemble(k) = prod_m p_m(k)^w_m / Z
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def log_opinion_pool(
    member_probs: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Combine member probability arrays via log-opinion pool.

    Args:
        member_probs: list of (n, k) probability arrays, one per member
        weights: (m,) weight vector (sums to 1 or gets normalized)

    Returns:
        (n, k) combined probabilities
    """
    assert len(member_probs) == len(weights)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    log_combined = np.zeros_like(member_probs[0], dtype=float)
    for w, probs in zip(weights, member_probs):
        log_combined += w * np.log(np.clip(probs, 1e-12, None))

    # numerically stable softmax normalization
    log_combined -= log_combined.max(axis=1, keepdims=True)
    combined = np.exp(log_combined)
    combined = np.clip(combined, 1e-12, None)
    return combined / combined.sum(axis=1, keepdims=True)


def fit_ensemble_weights(
    member_probs: list[np.ndarray],
    y_idx: np.ndarray,
) -> np.ndarray:
    """Learn log-opinion pool weights by minimizing validation log loss.

    Uses L-BFGS-B with simplex-like bounds (weights in [0, 1], normalized
    inside the objective).

    Args:
        member_probs: list of (n, k) probability arrays
        y_idx: (n,) integer class labels

    Returns:
        (m,) optimal weight vector (sums to 1)
    """
    m = len(member_probs)

    def neg_log_likelihood(raw_w: np.ndarray) -> float:
        w = np.exp(raw_w)
        w = w / w.sum()
        combined = log_opinion_pool(member_probs, w)
        ll = np.mean(np.log(np.clip(
            combined[np.arange(len(y_idx)), y_idx], 1e-12, None,
        )))
        return -float(ll)

    x0 = np.zeros(m)
    result = minimize(neg_log_likelihood, x0=x0, method="L-BFGS-B")
    w = np.exp(result.x)
    return w / w.sum()


class LogOpinionPool:
    """Fitted ensemble that stores member names and weights."""

    def __init__(
        self,
        member_names: list[str],
        weights: np.ndarray | None = None,
    ) -> None:
        self.member_names = member_names
        self.weights = weights if weights is not None else np.ones(len(member_names)) / len(member_names)

    def fit(
        self,
        member_probs: list[np.ndarray],
        y_idx: np.ndarray,
    ) -> "LogOpinionPool":
        self.weights = fit_ensemble_weights(member_probs, y_idx)
        return self

    def predict(self, member_probs: list[np.ndarray]) -> np.ndarray:
        return log_opinion_pool(member_probs, self.weights)

    def summary(self) -> dict[str, float]:
        return dict(zip(self.member_names, self.weights.tolist()))

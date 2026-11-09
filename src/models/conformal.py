"""Drift-aware conformal prediction: score functions × adaptation strategies.

The static split-conformal wrapper (src/models/calibration.py) undercovers on
real EPL data — 0.888 vs the 0.90 target (docs/backtest_epl.md) — because
teams change between seasons and exchangeability bends. This module holds the
study grid that quantifies whether the current literature's drift-aware
methods restore the guarantee, and at what set-size cost
(scripts/conformal_study.py runs it; docs/conformal_study.md reports it).

Score functions — a per-class nonconformity s(x, k); the prediction set is
{k : s(x, k) ≤ q}, so every strategy below composes with every score:

- ``LACScore``   s = 1 − p̂_k (Sadinle et al. 2019). Smallest sets, weakest
                 conditional coverage — identical sets to ``ConformalWrapper``.
- ``APSScore``   cumulative sorted probability mass down through class k
                 (Romano, Sesia & Candès 2020). Deterministic variant (no
                 tie-breaking randomization), which is mildly conservative.
- ``RAPSScore``  APS + λ·max(0, rank_k − k_reg) (Angelopoulos et al. 2021):
                 the rank penalty discourages bloated sets.

Adaptation strategies — how the threshold responds to temporal drift:

- ``StaticConformal``    split-conformal quantile; exchangeability assumed.
                         The baseline every comparison is anchored to.
- ``WeightedConformal``  weighted quantile with exponential time decay
                         (Barber, Candès, Ramdas & Tibshirani 2023,
                         "Conformal prediction beyond exchangeability"): the
                         coverage gap is bounded by the weighted drift, so
                         down-weighting stale matches buys robustness to
                         gradual change. Offline — no feedback loop needed.
- ``ACIConformal``       Adaptive Conformal Inference (Gibbs & Candès 2021):
                         after each settled match, α_t ← α_t + γ(α − err_t).
                         Long-run coverage converges to 1−α with NO
                         exchangeability assumption, at the price of needing
                         outcomes fed back in (here: after full-time, i.e.
                         the same settlement loop /reflect already runs).

Sets are never empty — the argmax class is always included, matching
``ConformalWrapper``'s convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import math

import numpy as np


class ScoreFunction(Protocol):
    @property
    def name(self) -> str: ...

    def per_class(self, probs: np.ndarray) -> np.ndarray:
        """(n, k) probabilities → (n, k) nonconformity scores (higher = worse)."""
        ...


@dataclass(frozen=True)
class LACScore:
    name: str = "lac"

    def per_class(self, probs: np.ndarray) -> np.ndarray:
        return 1.0 - np.atleast_2d(probs)


@dataclass(frozen=True)
class APSScore:
    name: str = "aps"

    def per_class(self, probs: np.ndarray) -> np.ndarray:
        probs = np.atleast_2d(probs)
        order = np.argsort(-probs, axis=1)
        cum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
        scores = np.empty_like(probs)
        np.put_along_axis(scores, order, cum, axis=1)
        return scores


@dataclass(frozen=True)
class RAPSScore:
    lam: float = 0.10
    k_reg: int = 1
    name: str = "raps"

    def per_class(self, probs: np.ndarray) -> np.ndarray:
        probs = np.atleast_2d(probs)
        order = np.argsort(-probs, axis=1)
        cum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
        ranks = np.arange(1, probs.shape[1] + 1, dtype=float)
        penalized = cum + self.lam * np.clip(ranks - self.k_reg, 0.0, None)
        scores = np.empty_like(probs)
        np.put_along_axis(scores, order, penalized, axis=1)
        return scores


def _true_scores(score_fn: ScoreFunction, probs: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    return score_fn.per_class(probs)[np.arange(len(y_idx)), y_idx]


def _sets_from_threshold(
    score_fn: ScoreFunction, probs: np.ndarray, q: float
) -> list[list[int]]:
    """{k : s(x,k) ≤ q}, argmax fallback so sets are never empty."""
    probs = np.atleast_2d(probs)
    scores = score_fn.per_class(probs)
    sets: list[list[int]] = []
    for row_s, row_p in zip(scores, probs):
        included = [k for k, s in enumerate(row_s) if s <= q]
        if not included:
            included = [int(np.argmax(row_p))]
        sets.append(included)
    return sets


def _corrected_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected empirical quantile at ⌈(n+1)(1−α)⌉/n.

    Returns +inf when the corrected level exceeds 1 (n too small for α) —
    downstream that yields the full outcome set, the honest answer.
    """
    n = len(scores)
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class StaticConformal:
    """Split conformal — the ``ConformalWrapper`` recipe, any score function."""

    score_fn: ScoreFunction = field(default_factory=LACScore)
    alpha: float = 0.1
    q_hat: float | None = None

    def fit(self, cal_probs: np.ndarray, cal_y: np.ndarray) -> "StaticConformal":
        self.q_hat = _corrected_quantile(
            _true_scores(self.score_fn, cal_probs, cal_y), self.alpha
        )
        return self

    def prediction_sets(self, probs: np.ndarray) -> list[list[int]]:
        assert self.q_hat is not None, "call fit() first"
        return _sets_from_threshold(self.score_fn, probs, self.q_hat)


@dataclass
class WeightedConformal:
    """Nonexchangeable conformal with exponential time-decay weights.

    Calibration matches are weighted w_i = ρ^age, ρ = 0.5^(1/half_life), age
    measured in matches back from the end of the calibration window. Following
    Barber et al. 2023, the quantile adds a point mass at +∞ with the test
    point's own (unit) weight — so as the test match recedes from the
    calibration data (``offset`` matches later), every calibration weight
    shrinks by ρ^offset, the +∞ atom gains relative mass, and the threshold
    drifts conservative: exactly the "trust stale data less" behavior the
    static quantile lacks.

    ``half_life=None`` means uniform weights, which recovers the canonical
    ⌈(n+1)(1−α)⌉-th order statistic of split conformal (np.quantile's
    "higher" interpolation in the static wrapper can sit one order statistic
    above it — same guarantee, marginally more conservative).
    """

    score_fn: ScoreFunction = field(default_factory=LACScore)
    alpha: float = 0.1
    half_life: float | None = 250.0
    _scores: np.ndarray | None = None   # temporally ordered, oldest first
    _ages: np.ndarray | None = None     # age in matches at calibration end

    def fit(self, cal_probs: np.ndarray, cal_y: np.ndarray) -> "WeightedConformal":
        """``cal_probs`` rows must be in kickoff order, oldest first."""
        self._scores = _true_scores(self.score_fn, cal_probs, cal_y)
        self._ages = np.arange(len(self._scores) - 1, -1, -1, dtype=float)
        return self

    def _quantile(self, offset: float) -> float:
        assert self._scores is not None and self._ages is not None, "call fit() first"
        if self.half_life is None:
            weights = np.ones_like(self._scores)
        else:
            rho = 0.5 ** (1.0 / self.half_life)
            weights = rho ** (self._ages + offset)
        order = np.argsort(self._scores, kind="stable")
        cum = np.cumsum(weights[order])
        # +∞ atom carries the test point's unit weight
        total = cum[-1] + 1.0
        hit = np.nonzero(cum / total >= 1.0 - self.alpha)[0]
        if len(hit) == 0:
            return float("inf")
        return float(self._scores[order][hit[0]])

    def prediction_sets(
        self, probs: np.ndarray, offsets: np.ndarray | None = None
    ) -> list[list[int]]:
        """``offsets[i]``: matches elapsed between calibration end and test row i."""
        probs = np.atleast_2d(probs)
        if offsets is None:
            offsets = np.zeros(len(probs))
        sets: list[list[int]] = []
        for row, off in zip(probs, offsets):
            sets.extend(_sets_from_threshold(self.score_fn, row[None, :], self._quantile(float(off))))
        return sets


@dataclass
class ACIConformal:
    """Adaptive Conformal Inference (Gibbs & Candès 2021).

    Maintains an effective level α_t updated online after each settled match:
    err_t = 1{y_t ∉ Γ_t}, then α_{t+1} = α_t + γ(α − err_t). Misses widen the
    sets, streaks of hits tighten them; long-run coverage → 1−α regardless of
    drift. Settled scores join the calibration pool (sliding ``window`` keeps
    it season-fresh; None = expanding).

    α_t clamps: α_t ≤ 0 → full set (threshold +∞); α_t ≥ 1 → argmax only.
    """

    score_fn: ScoreFunction = field(default_factory=LACScore)
    alpha: float = 0.1
    gamma: float = 0.01
    window: int | None = None
    _pool: list[float] = field(default_factory=list)
    alpha_t: float | None = None

    def fit(self, cal_probs: np.ndarray, cal_y: np.ndarray) -> "ACIConformal":
        self._pool = list(_true_scores(self.score_fn, cal_probs, cal_y))
        self.alpha_t = self.alpha
        return self

    def _threshold(self) -> float:
        assert self.alpha_t is not None, "call fit() first"
        if self.alpha_t <= 0.0:
            return float("inf")
        if self.alpha_t >= 1.0:
            return float("-inf")
        scores = np.asarray(self._pool[-self.window:] if self.window else self._pool)
        return _corrected_quantile(scores, self.alpha_t)

    def step(self, probs_row: np.ndarray) -> list[int]:
        """Prediction set for one match at the current α_t (no update)."""
        return _sets_from_threshold(self.score_fn, probs_row[None, :], self._threshold())[0]

    def update(self, probs_row: np.ndarray, y: int, pred_set: list[int]) -> None:
        """Feed back a settled outcome: α_t update + pool append."""
        err = float(y not in pred_set)
        if self.alpha_t is None:
            raise RuntimeError("call fit() before update()")
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err)
        self._pool.append(float(_true_scores(self.score_fn, probs_row[None, :], np.array([y]))[0]))

    def run(
        self, probs: np.ndarray, y_idx: np.ndarray
    ) -> tuple[list[list[int]], np.ndarray]:
        """Sequential predict-then-settle over a temporally ordered test slice.

        Returns (prediction sets, α_t trajectory as of each prediction).
        """
        probs = np.atleast_2d(probs)
        sets: list[list[int]] = []
        trajectory = np.empty(len(probs))
        for t, (row, y) in enumerate(zip(probs, y_idx)):
            trajectory[t] = self.alpha_t
            s = self.step(row)
            sets.append(s)
            self.update(row, int(y), s)
        return sets, trajectory

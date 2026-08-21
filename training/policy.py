"""Small MLP staking policy implemented from scratch in numpy.

Architecture: input(8) -> dense(64) -> ReLU -> dense(32) -> ReLU -> dense(4) -> softmax
Actions:      {0: skip, 1: bet_H, 2: bet_D, 3: bet_A}

The policy maps per-fixture feature vectors to a categorical distribution
over actions.  Weights are updated via REINFORCE gradients computed by
``backward()``, which uses the cached forward pass for backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class MLPPolicy:
    """Two-hidden-layer MLP with analytical backprop for REINFORCE."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden1: int = 64,
        hidden2: int = 32,
        n_actions: int = 4,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.n_actions = n_actions

        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden1))
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden1), (hidden1, hidden2))
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.normal(0, np.sqrt(2.0 / hidden2), (hidden2, n_actions))
        self.b3 = np.zeros(n_actions)

        self._cache: dict[str, np.ndarray] = {}

    # ── forward ──────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return action probabilities (n, n_actions)."""
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, z2)
        logits = h2 @ self.W3 + self.b3
        probs = _softmax(logits)

        self._cache = {
            "X": X, "z1": z1, "h1": h1,
            "z2": z2, "h2": h2, "probs": probs,
        }
        return probs

    # ── sampling ─────────────────────────────────────────────────────

    def sample(
        self, X: np.ndarray, rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample actions from the policy.

        Returns (actions, log_probs), both shape (n,).
        """
        rng = rng or np.random.default_rng()
        probs = self.forward(X)
        actions = np.array([rng.choice(self.n_actions, p=p) for p in probs])
        log_probs = np.log(probs[np.arange(len(actions)), actions] + 1e-10)
        return actions, log_probs

    def greedy(self, X: np.ndarray) -> np.ndarray:
        """Deterministic argmax actions for evaluation."""
        probs = self.forward(X)
        return probs.argmax(axis=1)

    # ── backward ─────────────────────────────────────────────────────

    def backward(
        self, actions: np.ndarray, advantages: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """REINFORCE gradients via backprop on the cached forward pass.

        The gradient of the policy-gradient loss
            L = -(1/n) sum_i advantage_i * log pi(a_i | s_i)
        w.r.t. the logits is:
            dL/d_logits[i] = advantage_i * (pi[i] - one_hot(a_i)) / n

        These are then backpropagated through the MLP.
        """
        c = self._cache
        X, z1, h1, z2, h2, probs = (
            c["X"], c["z1"], c["h1"], c["z2"], c["h2"], c["probs"],
        )
        n = len(actions)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), actions] = 1.0
        d_logits = (advantages[:, None] * (probs - one_hot)) / n

        # layer 3
        dW3 = h2.T @ d_logits
        db3 = d_logits.sum(axis=0)
        dh2 = d_logits @ self.W3.T

        # ReLU 2
        dz2 = dh2 * (z2 > 0).astype(np.float64)

        # layer 2
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T

        # ReLU 1
        dz1 = dh1 * (z1 > 0).astype(np.float64)

        # layer 1
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        return {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3,
        }

    # ── parameter access ─────────────────────────────────────────────

    @property
    def param_names(self) -> list[str]:
        return ["W1", "b1", "W2", "b2", "W3", "b3"]

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {n: getattr(self, n) for n in self.param_names}

    def n_params(self) -> int:
        return sum(p.size for p in self.params.values())

    def save(self, path: str) -> None:
        np.savez(path, **self.params)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> MLPPolicy:
        data = np.load(path)
        p = cls(**kwargs)
        for name in p.param_names:
            setattr(p, name, data[name])
        return p

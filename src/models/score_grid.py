"""Dixon–Coles scoreline grid and every market derived from it.

Dixon & Coles (1997): independent Poissons for home/away goals with a
low-score dependency correction τ (rho < 0 in practice — 0-0 and 1-1 occur
more often than independence implies). The team means (mu_home, mu_away) come
from TeamXGGBM, so the outcome model, scorelines, over/under, BTTS and
first-team-to-score all share one pair of xG estimates and cannot contradict
each other. rho is fit once on historical scores by MLE and stored in the
artifact.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson


def _tau(x: np.ndarray, y: np.ndarray, mu_h: float, mu_a: float, rho: float) -> np.ndarray:
    """Dixon–Coles low-score correction; 1.0 outside {0,1}x{0,1}."""
    t = np.ones_like(x, dtype=float)
    t = np.where((x == 0) & (y == 0), 1.0 - mu_h * mu_a * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + mu_h * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + mu_a * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return t


def score_grid(
    mu_home: float, mu_away: float, rho: float = -0.1, max_goals: int = 10
) -> np.ndarray:
    """(max_goals+1, max_goals+1) matrix, P[i, j] = P(home=i, away=j), sums to 1."""
    goals = np.arange(max_goals + 1)
    hx, ay = np.meshgrid(goals, goals, indexing="ij")
    grid = (
        _tau(hx, ay, mu_home, mu_away, rho)
        * poisson.pmf(hx, mu_home)
        * poisson.pmf(ay, mu_away)
    )
    grid = np.clip(grid, 0.0, None)
    return grid / grid.sum()  # renormalize away truncation + extreme-rho clipping


def fit_rho(
    goals_home: np.ndarray, goals_away: np.ndarray,
    mu_home: np.ndarray, mu_away: np.ndarray,
) -> float:
    """MLE for rho over historical matches with known (predicted) means."""

    def neg_ll(rho: float) -> float:
        tau = np.array([
            _tau(np.array(h), np.array(a), mh, ma, rho)
            for h, a, mh, ma in zip(goals_home, goals_away, mu_home, mu_away)
        ], dtype=float)
        if (tau <= 0).any():
            return 1e12
        ll = (
            np.log(tau)
            + poisson.logpmf(goals_home, mu_home)
            + poisson.logpmf(goals_away, mu_away)
        )
        return -float(ll.sum())

    res = minimize_scalar(neg_ll, bounds=(-0.35, 0.35), method="bounded")
    return float(res.x)


def outcome_probs(grid: np.ndarray) -> dict[str, float]:
    home = float(np.tril(grid, -1).sum())   # i > j
    away = float(np.triu(grid, 1).sum())    # j > i
    draw = float(np.trace(grid))
    return {"home": home, "draw": draw, "away": away}


def top_scorelines(grid: np.ndarray, n: int = 5) -> list[dict]:
    flat = [
        {"score": f"{i}-{j}", "prob": float(grid[i, j])}
        for i in range(grid.shape[0]) for j in range(grid.shape[1])
    ]
    return sorted(flat, key=lambda d: -d["prob"])[:n]


def over_under(grid: np.ndarray, line: float = 2.5) -> dict[str, float]:
    totals = np.add.outer(
        np.arange(grid.shape[0]), np.arange(grid.shape[1])
    )
    over = float(grid[totals > line].sum())
    return {"over": over, "under": 1.0 - over}


def btts(grid: np.ndarray) -> dict[str, float]:
    yes = float(grid[1:, 1:].sum())
    return {"yes": yes, "no": 1.0 - yes}


def knockout_advance(
    mu_home: float, mu_away: float, rho: float = -0.1,
    max_goals: int = 10, pens_home: float = 0.5,
) -> dict[str, float]:
    """P(advance) for a single-match knockout tie.

    Extra time is modeled as a 30-minute continuation: the same Dixon–Coles
    machinery with means scaled by 30/90. If ET is also level, penalties
    decide at ``pens_home`` (default 0.5; override with a shootout-skill
    estimate if you have one).
    """
    reg = outcome_probs(score_grid(mu_home, mu_away, rho, max_goals))
    et_grid = score_grid(mu_home / 3.0, mu_away / 3.0, rho, max_goals)
    et = outcome_probs(et_grid)
    home_adv = reg["home"] + reg["draw"] * (et["home"] + et["draw"] * pens_home)
    return {"home": home_adv, "away": 1.0 - home_adv}

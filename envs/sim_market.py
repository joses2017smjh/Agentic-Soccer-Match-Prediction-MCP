"""Tunable-efficiency synthetic market generator.

The market efficiency parameter eta in [0, 1] controls how far the closing
line travels toward the true probability:

    eta = 1.0 -- perfectly efficient.  Closing line = truth.  No CLV
                 available to anyone.  Correct policy: abstain.
    eta = 0.0 -- closing line is noisy walk from opening.  Low CLV.
    eta ~ 0.85-0.95 -- realistic range calibrated against observed CLV
                       distributions from football-data.co.uk.

This enables:
  - Unlimited episodes (no cap from real data)
  - Controllable difficulty / curriculum
  - Ablatable market structure (favourite-longshot bias on/off)
  - The fidelity ladder: does sim performance predict real performance?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterator

import numpy as np

from envs.real_market import Fixture, Gameweek, load_all_seasons


@dataclass(frozen=True)
class MarketConfig:
    """Configuration for the synthetic market generator."""
    eta: float = 0.85
    margin: float = 0.05
    flb_strength: float = 0.0
    n_matches_per_gw: int = 10
    opening_noise_std: float = 0.08
    line_movement_std: float = 0.05
    seed: int | None = None


@dataclass
class SimSeason:
    gameweeks: list[Gameweek]
    config: MarketConfig
    true_probs: list[list[np.ndarray]] = field(default_factory=list)


def _dirichlet_probs(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n probability vectors from a Dirichlet that mimics EPL."""
    alpha = np.array([2.5, 1.8, 1.5])
    return rng.dirichlet(alpha, size=n)


def _apply_flb(probs: np.ndarray, strength: float) -> np.ndarray:
    """Apply favourite-longshot bias: compress probabilities toward 1/3."""
    if strength <= 0.0:
        return probs
    uniform = np.ones(3) / 3.0
    biased = probs * (1.0 - strength) + uniform * strength
    return biased / biased.sum()


def _odds_from_probs(
    probs: np.ndarray,
    margin: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Convert true probs to opening odds with margin and noise."""
    noisy = probs + rng.normal(0, noise_std, size=3)
    noisy = np.clip(noisy, 0.02, 0.96)
    noisy = noisy / noisy.sum()
    fair_odds = 1.0 / noisy
    inflated = fair_odds / (1.0 + margin)
    return np.clip(inflated, 1.01, 100.0)


def _closing_odds(
    true_probs: np.ndarray,
    opening_odds: np.ndarray,
    eta: float,
    margin: float,
    movement_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute closing odds: eta-weighted blend of truth and random walk."""
    opening_implied = 1.0 / opening_odds
    opening_implied = opening_implied / opening_implied.sum()

    walk = opening_implied + rng.normal(0, movement_std, size=3)
    walk = np.clip(walk, 0.02, 0.96)
    walk = walk / walk.sum()

    closing_implied = eta * true_probs + (1.0 - eta) * walk
    closing_implied = np.clip(closing_implied, 0.02, 0.96)
    closing_implied = closing_implied / closing_implied.sum()

    fair_close = 1.0 / closing_implied
    inflated_close = fair_close / (1.0 + margin * 0.5)
    return np.clip(inflated_close, 1.01, 100.0)


def _sample_outcome(probs: np.ndarray, rng: np.random.Generator) -> str:
    idx = rng.choice(3, p=probs)
    return ["H", "D", "A"][idx]


def generate_season(
    config: MarketConfig | None = None,
    n_gameweeks: int = 38,
) -> SimSeason:
    """Generate a full synthetic season."""
    cfg = config or MarketConfig()
    rng = np.random.default_rng(cfg.seed)

    base_date = date(2024, 8, 17)
    all_gws: list[Gameweek] = []
    all_probs: list[list[np.ndarray]] = []

    for gw_num in range(1, n_gameweeks + 1):
        gw_date = base_date + timedelta(weeks=gw_num - 1)
        probs_batch = _dirichlet_probs(rng, cfg.n_matches_per_gw)
        fixtures: list[Fixture] = []
        gw_probs: list[np.ndarray] = []

        for i, true_p in enumerate(probs_batch):
            if cfg.flb_strength > 0:
                market_p = _apply_flb(true_p, cfg.flb_strength)
            else:
                market_p = true_p

            opening = _odds_from_probs(market_p, cfg.margin, cfg.opening_noise_std, rng)
            closing = _closing_odds(
                true_p, opening, cfg.eta, cfg.margin, cfg.line_movement_std, rng,
            )
            outcome = _sample_outcome(true_p, rng)

            home = f"Team_{chr(65 + i % 20)}"
            away = f"Team_{chr(65 + (i + 10) % 20)}"

            fixtures.append(Fixture(
                date=gw_date,
                match_id=f"SIM-{gw_num:02d}-{i:02d}",
                home=home,
                away=away,
                ftr=outcome,
                odds_h=round(float(opening[0]), 2),
                odds_d=round(float(opening[1]), 2),
                odds_a=round(float(opening[2]), 2),
                close_h=round(float(closing[0]), 2),
                close_d=round(float(closing[1]), 2),
                close_a=round(float(closing[2]), 2),
                odds_source="sim",
            ))
            gw_probs.append(true_p)

        all_gws.append(Gameweek(number=gw_num, dates=[gw_date], fixtures=fixtures))
        all_probs.append(gw_probs)

    return SimSeason(gameweeks=all_gws, config=cfg, true_probs=all_probs)


def generate_fixture_batch(
    n: int,
    eta: float = 0.85,
    seed: int | None = None,
    **kwargs,
) -> list[Fixture]:
    """Generate n standalone fixtures (no gameweek grouping)."""
    cfg = MarketConfig(eta=eta, seed=seed, n_matches_per_gw=n, **kwargs)
    season = generate_season(config=cfg, n_gameweeks=1)
    return season.gameweeks[0].fixtures


def calibrate_eta(
    target_mean_abs_clv: float = 0.02,
    n_trials: int = 1000,
    seed: int = 42,
) -> float:
    """Find eta that produces a target mean |CLV| for a naive favourite strategy.

    Returns the eta value that best matches the target CLV distribution.
    """
    best_eta = 0.85
    best_error = float("inf")

    for eta_candidate in np.linspace(0.5, 0.99, 50):
        cfg = MarketConfig(eta=eta_candidate, seed=seed, n_matches_per_gw=n_trials)
        season = generate_season(config=cfg, n_gameweeks=1)

        clvs = []
        for fix in season.gameweeks[0].fixtures:
            fav_sel = min(
                [("H", fix.odds_h), ("D", fix.odds_d), ("A", fix.odds_a)],
                key=lambda x: x[1],
            )
            close_map = {"H": fix.close_h, "D": fix.close_d, "A": fix.close_a}
            clv = (fav_sel[1] / close_map[fav_sel[0]]) - 1.0
            clvs.append(abs(clv))

        mean_abs = np.mean(clvs)
        error = abs(mean_abs - target_mean_abs_clv)
        if error < best_error:
            best_error = error
            best_eta = eta_candidate

    return round(best_eta, 3)


def sim_clv_distribution(
    eta: float,
    n_matches: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute CLV distribution statistics for a given eta."""
    fixtures = generate_fixture_batch(n_matches, eta=eta, seed=seed)

    clvs = []
    for fix in fixtures:
        fav = min(fix.odds_h, fix.odds_d, fix.odds_a)
        if fav == fix.odds_h:
            clv = (fix.odds_h / fix.close_h) - 1.0
        elif fav == fix.odds_d:
            clv = (fix.odds_d / fix.close_d) - 1.0
        else:
            clv = (fix.odds_a / fix.close_a) - 1.0
        clvs.append(clv)

    arr = np.array(clvs)
    return {
        "eta": eta,
        "mean_clv": round(float(arr.mean()), 6),
        "std_clv": round(float(arr.std()), 6),
        "mean_abs_clv": round(float(np.abs(arr).mean()), 6),
        "median_clv": round(float(np.median(arr)), 6),
        "pct_positive": round(float((arr > 0).mean()), 4),
        "n": len(clvs),
    }

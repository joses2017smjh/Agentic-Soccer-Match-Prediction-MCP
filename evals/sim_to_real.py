"""Sim-to-real transfer measurement.

Measures whether a policy's performance and the market's calibration
transfer from historical (sim) data to held-out (real) data -- directly
analogous to Halluminate's July 2026 sim-to-real study, which found
transfer collapsed from 25% to 5% when moving from Westworld to a real
website.

Two levels of transfer measurement:

1. **Market-level**: do the pre-close odds remain well-calibrated across
   seasons?  Measured by ECE and reliability curves on training vs
   held-out data.

2. **Environment-level**: does a staking policy trained on historical
   gameweeks perform comparably on unseen gameweeks?  Measured by the
   transfer ratio: performance_real / performance_sim.

The honest expectation: transfer will degrade, and reporting the gap is
the finding.  A model that reports a 0.72 transfer ratio is more
trustworthy than one that claims 1.0 without measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from envs.real_market import Fixture, Gameweek, iter_gameweeks, load_season
from src.eval.metrics import (
    brier,
    expected_calibration_error,
    log_loss,
    reliability_table,
)

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "football_data_uk"

SEASON_FILES = [
    "E0_1920.csv", "E0_2021.csv", "E0_2122.csv",
    "E0_2223.csv", "E0_2324.csv", "E0_2425.csv",
]

OUTCOME_MAP = {"H": 0, "D": 1, "A": 2}


# ── Market-level transfer ────────────────────────────────────────────

@dataclass
class CalibrationSnapshot:
    """Calibration metrics for one data split."""
    name: str
    n_matches: int
    log_loss: float
    brier_score: float
    ece: float
    reliability: list[dict[str, Any]]


@dataclass
class MarketTransferReport:
    sim: CalibrationSnapshot
    real: CalibrationSnapshot
    ece_drift: float
    brier_drift: float
    log_loss_drift: float

    def summary(self) -> dict[str, Any]:
        return {
            "sim_ece": self.sim.ece,
            "real_ece": self.real.ece,
            "ece_drift": round(self.ece_drift, 4),
            "sim_brier": self.sim.brier_score,
            "real_brier": self.real.brier_score,
            "brier_drift": round(self.brier_drift, 4),
            "sim_n": self.sim.n_matches,
            "real_n": self.real.n_matches,
        }


def _fixtures_to_arrays(
    fixtures: list[Fixture],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert fixtures to (probs, labels) arrays for calibration measurement.

    Uses de-vigged pre-close odds as the "model" probability.
    """
    probs_list = []
    labels = []
    for fix in fixtures:
        if fix.odds_h <= 0 or fix.odds_d <= 0 or fix.odds_a <= 0:
            continue
        if fix.ftr not in OUTCOME_MAP:
            continue
        raw = np.array([1.0 / fix.odds_h, 1.0 / fix.odds_d, 1.0 / fix.odds_a])
        total = raw.sum()
        if total <= 0:
            continue
        probs_list.append(raw / total)
        labels.append(OUTCOME_MAP[fix.ftr])

    return np.array(probs_list), np.array(labels, dtype=int)


def _calibration_snapshot(
    name: str, fixtures: list[Fixture],
) -> CalibrationSnapshot:
    probs, labels = _fixtures_to_arrays(fixtures)
    n = len(labels)
    if n == 0:
        return CalibrationSnapshot(
            name=name, n_matches=0, log_loss=0.0,
            brier_score=0.0, ece=0.0, reliability=[],
        )

    ll = log_loss(probs, labels)
    bs = brier(probs, labels)

    home_probs = probs[:, 0]
    home_outcomes = (labels == 0).astype(float)
    ece = expected_calibration_error(home_probs, home_outcomes, n_bins=10)
    rel = reliability_table(home_probs, home_outcomes, n_bins=10)

    return CalibrationSnapshot(
        name=name,
        n_matches=n,
        log_loss=round(ll, 4),
        brier_score=round(bs, 4),
        ece=round(ece, 4),
        reliability=rel.to_dict("records") if len(rel) > 0 else [],
    )


def measure_market_transfer(
    n_sim_seasons: int = 4,
) -> MarketTransferReport:
    """Compare market calibration between training and held-out seasons."""
    sim_fixtures: list[Fixture] = []
    for f in SEASON_FILES[:n_sim_seasons]:
        sim_fixtures.extend(load_season(_DATA_DIR / f))

    real_fixtures: list[Fixture] = []
    for f in SEASON_FILES[n_sim_seasons:]:
        real_fixtures.extend(load_season(_DATA_DIR / f))

    sim_snap = _calibration_snapshot("sim (seasons 1-4)", sim_fixtures)
    real_snap = _calibration_snapshot("real (seasons 5-6)", real_fixtures)

    return MarketTransferReport(
        sim=sim_snap,
        real=real_snap,
        ece_drift=real_snap.ece - sim_snap.ece,
        brier_drift=real_snap.brier_score - sim_snap.brier_score,
        log_loss_drift=real_snap.log_loss - sim_snap.log_loss,
    )


# ── Environment-level transfer ───────────────────────────────────────

@dataclass
class PolicyTransferReport:
    """Transfer measurement for one policy."""
    policy_name: str
    sim_reward: float
    real_reward: float
    transfer_ratio: float
    sim_clv: float
    real_clv: float
    sim_n_bets: int
    real_n_bets: int
    sim_gameweeks: int
    real_gameweeks: int


@dataclass
class EnvironmentTransferReport:
    policies: list[PolicyTransferReport]
    market_transfer: MarketTransferReport

    def summary_table(self) -> list[dict[str, Any]]:
        return [
            {
                "policy": p.policy_name,
                "sim_reward": round(p.sim_reward, 6),
                "real_reward": round(p.real_reward, 6),
                "transfer_ratio": round(p.transfer_ratio, 4)
                if p.transfer_ratio is not None else "n/a",
                "sim_clv": round(p.sim_clv, 6),
                "real_clv": round(p.real_clv, 6),
            }
            for p in self.policies
        ]


def measure_environment_transfer(
    policies: list[tuple[str, Any]],
    n_sim_seasons: int = 4,
    bankroll: float = 1000.0,
    max_gw: int | None = None,
) -> EnvironmentTransferReport:
    """Measure staking policy transfer from sim to real gameweeks.

    Args:
        policies: list of (name, Policy) pairs.
        n_sim_seasons: first N seasons are "sim", rest are "real".
        bankroll: per-episode bankroll.
        max_gw: cap gameweeks per split (for faster testing).
    """
    from envs.market_env import run_episode

    sim_fixtures: list[Fixture] = []
    for f in SEASON_FILES[:n_sim_seasons]:
        sim_fixtures.extend(load_season(_DATA_DIR / f))
    sim_fixtures.sort(key=lambda f: f.date)

    real_fixtures: list[Fixture] = []
    for f in SEASON_FILES[n_sim_seasons:]:
        real_fixtures.extend(load_season(_DATA_DIR / f))
    real_fixtures.sort(key=lambda f: f.date)

    sim_gws = list(iter_gameweeks(sim_fixtures))
    real_gws = list(iter_gameweeks(real_fixtures))

    if max_gw:
        sim_gws = sim_gws[:max_gw]
        real_gws = real_gws[:max_gw]

    policy_reports: list[PolicyTransferReport] = []

    for name, policy in policies:
        sim_results = [run_episode(gw, policy, bankroll=bankroll) for gw in sim_gws]
        real_results = [run_episode(gw, policy, bankroll=bankroll) for gw in real_gws]

        sim_rewards = [r.reward for r in sim_results]
        real_rewards = [r.reward for r in real_results]
        sim_clvs = [r.mean_clv for r in sim_results if r.n_bets > 0]
        real_clvs = [r.mean_clv for r in real_results if r.n_bets > 0]

        sim_r = float(np.mean(sim_rewards)) if sim_rewards else 0.0
        real_r = float(np.mean(real_rewards)) if real_rewards else 0.0

        if sim_r != 0 and abs(sim_r) > 1e-9:
            ratio = real_r / sim_r
        else:
            ratio = 1.0 if abs(real_r) < 1e-9 else float("inf")

        policy_reports.append(PolicyTransferReport(
            policy_name=name,
            sim_reward=sim_r,
            real_reward=real_r,
            transfer_ratio=ratio,
            sim_clv=float(np.mean(sim_clvs)) if sim_clvs else 0.0,
            real_clv=float(np.mean(real_clvs)) if real_clvs else 0.0,
            sim_n_bets=sum(r.n_bets for r in sim_results),
            real_n_bets=sum(r.n_bets for r in real_results),
            sim_gameweeks=len(sim_gws),
            real_gameweeks=len(real_gws),
        ))

    market = measure_market_transfer(n_sim_seasons)

    return EnvironmentTransferReport(
        policies=policy_reports,
        market_transfer=market,
    )

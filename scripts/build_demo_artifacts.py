"""Train a small demo artifact bundle on synthetic data.

Lets the ML inference MCP server (and the whole Phase B stack) run
end-to-end before real training data is wired in. Real training uses the
same ArtifactBundle contract, so nothing downstream changes.

Usage: .venv/bin/python -m scripts.build_demo_artifacts [version]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.artifact import ArtifactBundle
from src.models.calibration import (
    CalibratedOutcomeHead,
    ConformalWrapper,
    IsotonicCalibrator,
)
from src.models.gbm import OUTCOME_CLASSES, OutcomeGBM, TeamXGGBM
from src.models.score_grid import fit_rho
from src.models.sequence import GoalTimingModel

DEMO_FEATURES: list[str] = [
    "odds_imp_home", "odds_imp_draw", "odds_imp_away",
    "form_xg_for_home", "form_xg_against_home",
    "form_xg_for_away", "form_xg_against_away",
    "availability_home", "availability_away",
    "sentiment_home", "sentiment_away",
    "neutral_venue",
]

ARTIFACT_ROOT = Path(__file__).resolve().parent.parent / "data" / "artifacts"


def synthesize(n: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 0.6, n)
    avail_h, avail_a = rng.uniform(0.7, 1.0, n), rng.uniform(0.7, 1.0, n)
    mu_h = np.clip(1.45 * np.exp(0.35 * strength) * avail_h, 0.2, 4.5)
    mu_a = np.clip(1.15 * np.exp(-0.35 * strength) * avail_a, 0.2, 4.5)
    gh, ga = rng.poisson(mu_h), rng.poisson(mu_a)
    y = pd.Series(np.where(gh > ga, "home", np.where(gh < ga, "away", "draw")))

    p_h = 1 / (1 + np.exp(-0.9 * strength))
    x = pd.DataFrame({
        "odds_imp_home": 0.75 * p_h + 0.05,
        "odds_imp_draw": np.full(n, 0.24),
        "odds_imp_away": np.clip(0.71 - 0.75 * p_h, 0.03, None),
        "form_xg_for_home": mu_h + rng.normal(0, 0.25, n),
        "form_xg_against_home": mu_a + rng.normal(0, 0.3, n),
        "form_xg_for_away": mu_a + rng.normal(0, 0.25, n),
        "form_xg_against_away": mu_h + rng.normal(0, 0.3, n),
        "availability_home": avail_h,
        "availability_away": avail_a,
        "sentiment_home": rng.uniform(-0.5, 0.5, n),
        "sentiment_away": rng.uniform(-0.5, 0.5, n),
        "neutral_venue": rng.integers(0, 2, n).astype(float),
    })[DEMO_FEATURES]
    return x, y, mu_h, mu_a, gh, ga


def build(version: str = "v0-demo", seed: int = 42) -> Path:
    x, y, mu_h, mu_a, gh, ga = synthesize(seed=seed)
    n = len(x)
    i_train, i_cal, i_conf = (
        slice(0, int(n * 0.6)), slice(int(n * 0.6), int(n * 0.8)),
        slice(int(n * 0.8), n),
    )

    outcome = OutcomeGBM(seed=seed, params={"n_estimators": 150}).fit(
        x[i_train], y[i_train]
    )
    xg = TeamXGGBM(seed=seed, params={"n_estimators": 150}).fit(
        x[i_train], pd.Series(mu_h[i_train]), pd.Series(mu_a[i_train])
    )

    y_idx = pd.Categorical(y, categories=list(OUTCOME_CLASSES)).codes
    calibrator = IsotonicCalibrator().fit(
        outcome.predict_proba(x[i_cal]), y_idx[i_cal]
    )
    conformal = ConformalWrapper(alpha=0.1).fit(
        calibrator.transform(outcome.predict_proba(x[i_conf])), y_idx[i_conf]
    )
    head = CalibratedOutcomeHead(calibrator=calibrator, conformal=conformal)

    ph, pa = xg.predict(x[i_train])
    rho = fit_rho(gh[i_train].astype(float), ga[i_train].astype(float), ph, pa)

    rng = np.random.default_rng(seed)
    timing = GoalTimingModel().fit(pd.DataFrame({
        "minute": np.clip(rng.beta(1.6, 1.2, 3000) * 90, 0, 89.9),  # late skew
        "scorer_state": rng.choice(
            ["level", "trailing", "leading"], 3000, p=[0.5, 0.28, 0.22]
        ),
    }))

    bundle = ArtifactBundle(
        version=version, outcome=outcome, xg=xg, head=head, timing=timing,
        rho=rho,
        card=ArtifactBundle.build_card(
            version=version, training_window="synthetic-demo", rho=rho,
            conformal_alpha=0.1, feature_names=DEMO_FEATURES,
            notes="Synthetic demo bundle - NOT trained on real matches.",
        ),
    )
    path = bundle.save(ARTIFACT_ROOT)
    print(f"artifact bundle written to {path} (rho={rho:.3f})")
    return path


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "v0-demo")

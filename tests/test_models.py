"""Unit tests: GBM heads, calibration, conformal coverage, player allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.calibration import (
    CalibratedOutcomeHead,
    ConformalWrapper,
    IsotonicCalibrator,
)
from src.models.gbm import FeatureSchemaError, OutcomeGBM, TeamXGGBM
from src.models.player_props import allocate_player_props


def _synthetic_matches(n: int = 800, seed: int = 0):
    """Feature frame where odds anchors truly drive outcomes."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1, n)
    x = pd.DataFrame({
        "odds_imp_home": 1 / (1 + np.exp(-strength)),
        "form_xg_for_home": 1.3 + 0.4 * strength + rng.normal(0, 0.2, n),
        "form_xg_for_away": 1.2 - 0.3 * strength + rng.normal(0, 0.2, n),
    })
    mu_h = np.clip(1.35 + 0.5 * strength, 0.2, 4)
    mu_a = np.clip(1.15 - 0.4 * strength, 0.2, 4)
    gh, ga = rng.poisson(mu_h), rng.poisson(mu_a)
    y = pd.Series(np.where(gh > ga, "home", np.where(gh < ga, "away", "draw")))
    return x, y, mu_h, mu_a


def test_outcome_gbm_learns_and_orders_probs() -> None:
    x, y, *_ = _synthetic_matches()
    model = OutcomeGBM(params={"n_estimators": 60}).fit(x, y)
    probs = model.predict_proba(x)
    assert probs.shape == (len(x), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    # strongest home rows should carry high home probability
    top = x["odds_imp_home"] > 0.85
    assert probs[top.to_numpy(), 0].mean() > 0.5


def test_gbm_schema_guard(tmp_path) -> None:
    x, y, *_ = _synthetic_matches(200)
    model = OutcomeGBM(params={"n_estimators": 10}).fit(x, y)
    model.save(tmp_path)
    loaded = OutcomeGBM.load(tmp_path)
    with pytest.raises(FeatureSchemaError, match="odds_imp_home"):
        loaded.predict_proba(x.drop(columns=["odds_imp_home"]))


def test_team_xg_gbm_roundtrip(tmp_path) -> None:
    x, _, mu_h, mu_a = _synthetic_matches(400)
    model = TeamXGGBM(params={"n_estimators": 60}).fit(
        x, pd.Series(mu_h), pd.Series(mu_a)
    )
    model.save(tmp_path)
    ph, pa = TeamXGGBM.load(tmp_path).predict(x)
    assert (ph > 0).all() and (pa > 0).all()
    assert np.corrcoef(ph, mu_h)[0, 1] > 0.8


def test_isotonic_calibration_improves_and_normalizes() -> None:
    rng = np.random.default_rng(1)
    n = 3000
    true_p = rng.dirichlet([4, 3, 3], n)
    y_idx = np.array([rng.choice(3, p=p) for p in true_p])
    overconfident = true_p ** 2 / (true_p ** 2).sum(axis=1, keepdims=True)
    cal = IsotonicCalibrator().fit(overconfident[: n // 2], y_idx[: n // 2])
    out = cal.transform(overconfident[n // 2:])
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-6)

    def logloss(p):
        return -np.mean(np.log(p[np.arange(len(p)), y_idx[n // 2:]]))

    assert logloss(out) < logloss(overconfident[n // 2:])


def test_conformal_coverage_on_holdout() -> None:
    """Split-conformal sets must hit ≥ 1 - alpha coverage (within MC noise)."""
    rng = np.random.default_rng(2)
    n = 6000
    probs = rng.dirichlet([5, 3, 2], n)
    y_idx = np.array([rng.choice(3, p=p) for p in probs])
    conf = ConformalWrapper(alpha=0.1).fit(probs[: n // 2], y_idx[: n // 2])
    cov = conf.empirical_coverage(probs[n // 2:], y_idx[n // 2:])
    assert cov >= 0.88
    sets = conf.prediction_set(probs[n // 2:])
    assert all(len(s) >= 1 for s in sets)


def test_calibrated_head_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(3)
    probs = rng.dirichlet([4, 3, 3], 1000)
    y_idx = np.array([rng.choice(3, p=p) for p in probs])
    head = CalibratedOutcomeHead(
        calibrator=IsotonicCalibrator().fit(probs, y_idx),
        conformal=ConformalWrapper(alpha=0.1).fit(probs, y_idx),
    )
    head.save(tmp_path)
    cal, sets = CalibratedOutcomeHead.load(tmp_path).predict(probs[:5])
    assert cal.shape == (5, 3) and len(sets) == 5


@pytest.fixture
def squad_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "player": ["Striker", "Winger", "Mid", "Def"],
        "xg_share": [0.45, 0.30, 0.20, 0.05],
        "xa_share": [0.10, 0.40, 0.45, 0.05],
        "exp_minutes": [90, 90, 90, 90],
        "availability": [1.0, 1.0, 1.0, 1.0],
        "setpiece_mult": [1.2, 1.0, 1.0, 1.0],
    })


def test_player_allocation_conserves_team_mu(squad_frame: pd.DataFrame) -> None:
    out = allocate_player_props(squad_frame, team_mu=2.0)
    assert out["goal_lambda"].sum() == pytest.approx(2.0 * 0.98)
    assert out.iloc[0]["player"] == "Striker"
    assert ((out["p_anytime_scorer"] > 0) & (out["p_anytime_scorer"] < 1)).all()


def test_injured_striker_share_redistributes(squad_frame: pd.DataFrame) -> None:
    """Ruling the striker out zeroes his prop and lifts every teammate's."""
    base = allocate_player_props(squad_frame, team_mu=2.0).set_index("player")
    injured = squad_frame.copy()
    injured.loc[injured["player"] == "Striker", "availability"] = 0.0
    shocked = allocate_player_props(injured, team_mu=2.0).set_index("player")

    assert shocked.loc["Striker", "p_anytime_scorer"] == 0.0
    for p in ["Winger", "Mid", "Def"]:
        assert shocked.loc[p, "p_anytime_scorer"] > base.loc[p, "p_anytime_scorer"]
    assert shocked["goal_lambda"].sum() == pytest.approx(2.0 * 0.98)

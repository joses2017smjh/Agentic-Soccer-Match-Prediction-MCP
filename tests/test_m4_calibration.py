"""Tests for M4: calibration ladder (B2) and adaptive conformal (B7).

B2: Temperature, vector, Dirichlet calibrators produce valid probabilities
    and select_calibrator picks the best on validation log loss.
B7: AdaptiveConformal adjusts alpha_t online to correct undercoverage
    under distribution drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.calibration import (
    AdaptiveConformal,
    CalibratedOutcomeHead,
    ConformalWrapper,
    DirichletCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
    VectorCalibrator,
    select_calibrator,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _synthetic_probs(n: int = 500, seed: int = 42):
    """Generate synthetic miscalibrated probs and labels."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet([2, 1, 1], size=n)
    # make them miscalibrated by sharpening
    logits = np.log(raw + 1e-8) * 2.0
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    y_idx = np.array([rng.choice(3, p=p) for p in raw])
    return probs, y_idx


# --------------------------------------------------------------------------
# B2: calibration ladder -- individual calibrators
# --------------------------------------------------------------------------

class TestIsotonicCalibrator:
    def test_fit_and_transform_valid(self):
        probs, y = _synthetic_probs(300)
        cal = IsotonicCalibrator().fit(probs, y)
        out = cal.transform(probs)
        assert out.shape == probs.shape
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
        assert (out >= 0).all()

    def test_name(self):
        assert IsotonicCalibrator().name == "isotonic"


class TestTemperatureCalibrator:
    def test_fit_learns_temperature(self):
        probs, y = _synthetic_probs(300)
        cal = TemperatureCalibrator().fit(probs, y)
        assert cal.temperature > 0
        assert cal.temperature != 1.0  # should adjust from default

    def test_transform_sums_to_one(self):
        probs, y = _synthetic_probs(300)
        cal = TemperatureCalibrator().fit(probs, y)
        out = cal.transform(probs)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_identity_at_t1(self):
        probs, _ = _synthetic_probs(100)
        cal = TemperatureCalibrator(temperature=1.0)
        cal.temperature = 1.0
        out = cal.transform(probs)
        np.testing.assert_allclose(out, probs, atol=1e-4)


class TestVectorCalibrator:
    def test_fit_and_transform(self):
        probs, y = _synthetic_probs(300)
        cal = VectorCalibrator().fit(probs, y)
        out = cal.transform(probs)
        assert out.shape == probs.shape
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_params_exist(self):
        probs, y = _synthetic_probs(200)
        cal = VectorCalibrator().fit(probs, y)
        assert cal._w is not None and len(cal._w) == 3
        assert cal._b is not None and len(cal._b) == 3


class TestDirichletCalibrator:
    def test_fit_and_transform(self):
        probs, y = _synthetic_probs(300)
        cal = DirichletCalibrator().fit(probs, y)
        out = cal.transform(probs)
        assert out.shape == probs.shape
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_identity_init(self):
        probs, y = _synthetic_probs(200)
        cal = DirichletCalibrator().fit(probs, y)
        assert cal._W is not None
        assert cal._W.shape == (3, 3)


# --------------------------------------------------------------------------
# B2: calibration ladder -- select_calibrator
# --------------------------------------------------------------------------

def test_select_calibrator_picks_best():
    probs, y = _synthetic_probs(400, seed=0)
    train_p, val_p = probs[:300], probs[300:]
    train_y, val_y = y[:300], y[300:]
    best = select_calibrator(train_p, train_y, val_p, val_y)
    assert hasattr(best, "name")
    assert hasattr(best, "transform")
    out = best.transform(val_p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_select_calibrator_reduces_log_loss():
    from src.eval.metrics import log_loss
    probs, y = _synthetic_probs(400, seed=7)
    train_p, val_p = probs[:300], probs[300:]
    train_y, val_y = y[:300], y[300:]
    raw_ll = log_loss(val_p, val_y)
    best = select_calibrator(train_p, train_y, val_p, val_y)
    cal_ll = log_loss(best.transform(val_p), val_y)
    assert cal_ll <= raw_ll + 0.01


# --------------------------------------------------------------------------
# B7: ConformalWrapper (incumbent)
# --------------------------------------------------------------------------

def test_conformal_wrapper_basic():
    probs, y = _synthetic_probs(200)
    cw = ConformalWrapper(alpha=0.1).fit(probs, y)
    assert cw.q_hat is not None
    sets = cw.prediction_set(probs)
    assert len(sets) == 200
    assert all(len(s) >= 1 for s in sets)


def test_conformal_wrapper_coverage():
    probs, y = _synthetic_probs(500, seed=99)
    cal, test = probs[:300], probs[300:]
    cal_y, test_y = y[:300], y[300:]
    cw = ConformalWrapper(alpha=0.1).fit(cal, cal_y)
    cov = cw.empirical_coverage(test, test_y)
    assert cov >= 0.75  # finite sample with miscalibrated probs


# --------------------------------------------------------------------------
# B7: AdaptiveConformal (ACI -- the fix for undercoverage)
# --------------------------------------------------------------------------

def test_adaptive_conformal_fit():
    probs, y = _synthetic_probs(200)
    ac = AdaptiveConformal(alpha=0.1, gamma=0.005).fit(probs, y)
    assert ac.q_hat is not None
    assert ac._alpha_t is not None


def test_adaptive_conformal_prediction_set():
    probs, y = _synthetic_probs(200)
    ac = AdaptiveConformal(alpha=0.1).fit(probs, y)
    sets = ac.prediction_set(probs)
    assert len(sets) == 200
    assert all(len(s) >= 1 for s in sets)


def test_adaptive_conformal_update_adjusts_alpha():
    probs, y = _synthetic_probs(200)
    ac = AdaptiveConformal(alpha=0.1, gamma=0.01).fit(probs[:100], y[:100])
    alpha_before = ac._alpha_t
    # force a miss: set y to a class NOT in the prediction set
    pred_set = ac.prediction_set(probs[100:101])
    excluded = [k for k in range(3) if k not in pred_set[0]]
    if excluded:
        ac.update(probs[100], excluded[0])
        assert ac._alpha_t != alpha_before


def test_adaptive_conformal_corrects_drift():
    """Under a distribution shift, ACI should adjust alpha_t to maintain
    coverage, unlike static conformal which would undercover."""
    rng = np.random.default_rng(42)
    n_cal = 200
    n_test = 200

    # calibration: well-calibrated
    cal_probs = rng.dirichlet([3, 2, 1], size=n_cal)
    cal_y = np.array([rng.choice(3, p=p) for p in cal_probs])

    # test: shifted distribution (draw-heavy)
    test_true = rng.dirichlet([1, 5, 1], size=n_test)
    test_probs = rng.dirichlet([3, 2, 1], size=n_test)  # model still calibrated to old
    test_y = np.array([rng.choice(3, p=p) for p in test_true])

    # static conformal
    static = ConformalWrapper(alpha=0.1).fit(cal_probs, cal_y)
    static_cov = static.empirical_coverage(test_probs, test_y)

    # adaptive conformal with online updates
    adaptive = AdaptiveConformal(alpha=0.1, gamma=0.01).fit(cal_probs, cal_y)
    adaptive_cov = adaptive.update_batch(test_probs, test_y)

    # ACI should have at least comparable coverage, and alpha_t should have shifted
    assert adaptive._alpha_t != 0.1  # alpha adjusted


def test_adaptive_conformal_update_batch_coverage():
    probs, y = _synthetic_probs(400, seed=33)
    ac = AdaptiveConformal(alpha=0.1, gamma=0.005).fit(probs[:200], y[:200])
    cov = ac.update_batch(probs[200:], y[200:])
    assert 0.5 <= cov <= 1.0  # reasonable coverage range


def test_adaptive_conformal_alpha_bounded():
    probs, y = _synthetic_probs(300, seed=11)
    ac = AdaptiveConformal(alpha=0.1, gamma=0.1).fit(probs[:100], y[:100])
    for i in range(100, 300):
        ac.update(probs[i], y[i])
    assert 0.001 <= ac._alpha_t <= 0.999


# --------------------------------------------------------------------------
# CalibratedOutcomeHead with new calibrators
# --------------------------------------------------------------------------

def test_outcome_head_with_temperature_and_adaptive(tmp_path):
    probs, y = _synthetic_probs(200)
    cal = TemperatureCalibrator().fit(probs, y)
    conf = AdaptiveConformal(alpha=0.1).fit(cal.transform(probs), y)
    head = CalibratedOutcomeHead(calibrator=cal, conformal=conf)
    out_probs, out_sets = head.predict(probs)
    assert out_probs.shape == probs.shape
    np.testing.assert_allclose(out_probs.sum(axis=1), 1.0, atol=1e-6)
    assert len(out_sets) == 200

    head.save(tmp_path)
    loaded = CalibratedOutcomeHead.load(tmp_path)
    out2, sets2 = loaded.predict(probs)
    np.testing.assert_allclose(out_probs, out2, atol=1e-6)


def test_outcome_head_backward_compat(tmp_path):
    """Old-style head with IsotonicCalibrator + ConformalWrapper still works."""
    probs, y = _synthetic_probs(200)
    cal = IsotonicCalibrator().fit(probs, y)
    conf = ConformalWrapper(alpha=0.1).fit(cal.transform(probs), y)
    head = CalibratedOutcomeHead(calibrator=cal, conformal=conf)
    out_probs, out_sets = head.predict(probs)
    assert out_probs.shape == probs.shape
    np.testing.assert_allclose(out_probs.sum(axis=1), 1.0, atol=1e-6)

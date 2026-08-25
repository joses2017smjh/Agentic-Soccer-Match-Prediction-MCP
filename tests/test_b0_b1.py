"""Tests for B0 (odds_stage fix) and B1 (market-anchored residual learning).

B0: closing_odds_frame now distinguishes pre-close ({book}H) from close
    ({book}CH) columns, and load_seasons returns both.
B1: OutcomeGBM accepts base_margin so with zero trees it reproduces the
    market exactly; every tree must earn its reduction on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.football_data_uk import closing_odds_frame, load_seasons
from src.models.gbm import OutcomeGBM


# --------------------------------------------------------------------------
# B0: odds_stage
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def season_csv() -> pd.DataFrame:
    from src.data.football_data_uk import fetch_season_csv
    return fetch_season_csv("E0", 2024)


def test_pre_close_reads_psh_columns(season_csv: pd.DataFrame) -> None:
    frame = closing_odds_frame(season_csv, odds_stage="pre_close")
    assert len(frame) > 0
    assert (frame["odds_stage"] == "pre_close").all()
    assert {"odds_home", "odds_draw", "odds_away",
            "odds_imp_home", "odds_imp_draw", "odds_imp_away"}.issubset(frame.columns)


def test_close_reads_psch_columns(season_csv: pd.DataFrame) -> None:
    frame = closing_odds_frame(season_csv, odds_stage="close")
    assert len(frame) > 0
    assert (frame["odds_stage"] == "close").all()


def test_pre_close_and_close_differ(season_csv: pd.DataFrame) -> None:
    pre = closing_odds_frame(season_csv, odds_stage="pre_close")
    close = closing_odds_frame(season_csv, odds_stage="close")
    merged = pre.merge(close, on="match_id", suffixes=("_pre", "_close"))
    assert len(merged) > 100
    diffs = (merged["odds_home_pre"] - merged["odds_home_close"]).abs()
    assert diffs.sum() > 0, "pre-close and close should differ on at least some matches"


def test_invalid_odds_stage_raises(season_csv: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="odds_stage"):
        closing_odds_frame(season_csv, odds_stage="open")


def test_devigged_probs_sum_to_one(season_csv: pd.DataFrame) -> None:
    for stage in ("pre_close", "close"):
        frame = closing_odds_frame(season_csv, odds_stage=stage)
        totals = (frame["odds_imp_home"] + frame["odds_imp_draw"]
                  + frame["odds_imp_away"])
        np.testing.assert_allclose(totals, 1.0, atol=1e-8)


def test_load_seasons_returns_three_frames() -> None:
    team, pre_close, close = load_seasons("E0", [2024])
    assert len(team) > 0
    assert len(pre_close) > 0
    assert len(close) > 0
    assert (pre_close["odds_stage"] == "pre_close").all()
    assert (close["odds_stage"] == "close").all()


def test_no_close_leaks_into_features() -> None:
    """Close-price columns must never enter the feature matrix.

    The backtest's FEATURES list must not contain any close-price column
    names.  This is the leakage guard for B3/C3.
    """
    from scripts.backtest_epl import FEATURES
    close_patterns = ["close_imp", "close_odds", "PSCH", "PSCD", "PSCA"]
    for feat in FEATURES:
        for pat in close_patterns:
            assert pat not in feat, f"close-price leak: {feat!r} contains {pat!r}"


# --------------------------------------------------------------------------
# B1: market-anchored residual learning
# --------------------------------------------------------------------------

def _synthetic_with_market(n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1, n)
    true_home = 1 / (1 + np.exp(-strength))
    true_draw = np.full(n, 0.25)
    true_away = 1 - true_home - true_draw
    true_away = np.clip(true_away, 0.05, None)
    total = true_home + true_draw + true_away
    true_home /= total
    true_draw /= total
    true_away /= total

    market_probs = np.column_stack([true_home, true_draw, true_away])
    noise_feats = pd.DataFrame({
        "form_xg_for_home": 1.3 + 0.3 * strength + rng.normal(0, 0.2, n),
        "form_xg_for_away": 1.2 - 0.2 * strength + rng.normal(0, 0.2, n),
    })

    y_idx = np.array([rng.choice(3, p=p) for p in market_probs])
    y = pd.Series(np.array(["home", "draw", "away"])[y_idx])
    return noise_feats, y, market_probs


def test_base_margin_zero_trees_reproduces_market() -> None:
    x, y, market_probs = _synthetic_with_market(200)
    margin = np.log(market_probs)
    model = OutcomeGBM(params={"n_estimators": 0}).fit(x, y, base_margin=margin)
    preds = model.predict_proba(x, base_margin=margin)
    np.testing.assert_allclose(preds, market_probs, atol=1e-4)


def test_base_margin_model_beats_non_anchored() -> None:
    """A base-margin model should outperform a non-anchored model that gets
    the same features, because the market prior gives it a head start."""
    x, y, market_probs = _synthetic_with_market(600, seed=7)
    margin = np.log(market_probs)

    train_n = 400
    x_train, x_test = x.iloc[:train_n], x.iloc[train_n:]
    y_train, y_test = y.iloc[:train_n], y.iloc[train_n:]
    margin_train, margin_test = margin[:train_n], margin[train_n:]

    params = {"n_estimators": 50, "max_depth": 2, "min_child_weight": 10}
    anchored = OutcomeGBM(seed=42, params=params).fit(
        x_train, y_train, base_margin=margin_train,
    )
    plain = OutcomeGBM(seed=42, params=params).fit(x_train, y_train)

    from src.eval.metrics import log_loss
    from src.eval.backtest import OUTCOME_ORDER
    y_idx = pd.Categorical(y_test, categories=list(OUTCOME_ORDER)).codes.astype(int)

    anchored_ll = log_loss(
        anchored.predict_proba(x_test, base_margin=margin_test), y_idx,
    )
    plain_ll = log_loss(plain.predict_proba(x_test), y_idx)

    assert anchored_ll < plain_ll, (
        f"anchored ({anchored_ll:.4f}) should beat plain ({plain_ll:.4f})"
    )


def test_base_margin_save_load_roundtrip(tmp_path) -> None:
    x, y, market_probs = _synthetic_with_market(200)
    margin = np.log(market_probs)
    model = OutcomeGBM(params={"n_estimators": 10}).fit(x, y, base_margin=margin)
    model.save(tmp_path)

    loaded = OutcomeGBM.load(tmp_path)
    assert loaded._uses_base_margin is True
    preds_orig = model.predict_proba(x, base_margin=margin)
    preds_loaded = loaded.predict_proba(x, base_margin=margin)
    np.testing.assert_allclose(preds_orig, preds_loaded, atol=1e-6)


def test_without_base_margin_still_works() -> None:
    x, y, _ = _synthetic_with_market(200)
    model = OutcomeGBM(params={"n_estimators": 20}).fit(x, y)
    preds = model.predict_proba(x)
    assert preds.shape == (200, 3)
    np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-5)

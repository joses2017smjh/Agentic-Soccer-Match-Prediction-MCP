"""Walk-forward backtest on REAL data: EPL via football-data.co.uk.

Trains the outcome stack (XGBoost → isotonic → conformal) on real results
and real de-vigged closing odds, walking forward season by season. Reports,
per fold and overall:

- log loss / Brier / RPS for model vs de-vigged closing line vs naive
  baseline, on the identical match set;
- empirical conformal coverage vs the 1-α target, with mean set size;
- simulated ROI of the suggestion layer settled at payable closing odds.

The closing line is the benchmark to beat. If the model loses to it, this
script says so — that is the honest result the README promises.

Usage: .venv/bin/python -m scripts.backtest_epl [--out docs/backtest_epl.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.football_data_uk import load_seasons
from src.eval.backtest import OUTCOME_ORDER, run_backtest, simulate_roi
from src.models.calibration import ConformalWrapper, IsotonicCalibrator
from src.models.gbm import OutcomeGBM
from src.features.team_form import FORM_STATS, match_level_form, team_form_features

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
ALPHA = 0.10
SEED = 42

FEATURES = (
    [f"form_{s}_home" for s in FORM_STATS]
    + [f"form_{s}_away" for s in FORM_STATS]
    + ["form_n_matches_home", "form_n_matches_away",
       "rest_days_home", "rest_days_away",
       "odds_imp_home", "odds_imp_draw", "odds_imp_away"]
)


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    team_matches, odds = load_seasons("E0", SEASONS)
    form = team_form_features(team_matches, window=10, half_life=5.0)
    matches = match_level_form(form)

    # labels from the home side's goals
    home_rows = (
        team_matches[team_matches["is_home"]]
        .set_index("match_id")[["goals_for", "goals_against"]]
    )
    matches = matches.merge(
        home_rows, left_on="match_id", right_index=True, how="inner"
    ).merge(odds, on="match_id", how="inner")

    y = pd.Series(
        np.where(matches["goals_for"] > matches["goals_against"], "home",
                 np.where(matches["goals_for"] < matches["goals_against"],
                          "away", "draw")),
        index=matches.index,
    )
    return matches, y


def make_fit_predict(y: pd.Series, coverage_log: list[dict]):
    """Per fold: 75% GBM / 25% isotonic+conformal, temporally ordered."""

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        train = train.sort_values("kickoff_utc")
        cut = int(len(train) * 0.70)
        fit_idx, cal_idx = train.index[:cut], train.index[cut:]

        gbm = OutcomeGBM(seed=SEED, params={
            "n_estimators": 200, "max_depth": 3, "min_child_weight": 8,
        }).fit(train.loc[fit_idx, FEATURES], y.loc[fit_idx])
        y_codes = pd.Categorical(y, categories=list(OUTCOME_ORDER)).codes
        cal_raw = gbm.predict_proba(train.loc[cal_idx, FEATURES])

        # isotonic on a small slice overfits badly (step functions through
        # noise) — below 300 calibration rows, skip it and conformalize the
        # raw probabilities instead
        if len(cal_idx) >= 300:
            calibrator = IsotonicCalibrator().fit(
                cal_raw, y_codes[y.index.get_indexer(cal_idx)]
            )
            transform = calibrator.transform
        else:
            transform = lambda p: p  # noqa: E731

        conformal = ConformalWrapper(alpha=ALPHA).fit(
            transform(cal_raw), y_codes[y.index.get_indexer(cal_idx)]
        )

        test_probs = transform(gbm.predict_proba(test[FEATURES]))
        sets = conformal.prediction_set(test_probs)
        y_test = y_codes[y.index.get_indexer(test.index)]
        coverage_log.append({
            "season": str(test["season"].iloc[0]),
            "coverage": float(np.mean([t in s for t, s in zip(y_test, sets)])),
            "mean_set_size": float(np.mean([len(s) for s in sets])),
            "n": len(test),
        })
        return test_probs

    return fit_predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/backtest_epl.md"))
    args = parser.parse_args()

    matches, y = build_dataset()
    coverage_log: list[dict] = []
    report = run_backtest(
        matches, y, make_fit_predict(y, coverage_log),
        min_train_groups=2,   # first fold trains on two full seasons
    )
    summary = report.summary()

    preds = pd.concat([f.predictions for f in report.folds])
    preds = preds.join(matches[["odds_home", "odds_draw", "odds_away"]])
    roi_flat = simulate_roi(preds, ev_threshold=0.03)
    roi_kelly = simulate_roi(preds, ev_threshold=0.03, use_kelly=True)

    coverage = pd.DataFrame(coverage_log)
    overall_cov = float(np.average(coverage["coverage"], weights=coverage["n"]))
    n_total = int(sum(f.n for f in report.folds))
    model_ll = summary.loc["model", "logloss"]
    market_ll = summary.loc["market", "logloss"]
    verdict = (
        "the model BEATS the closing line"
        if model_ll < market_ll else
        "the closing line beats the model — as expected; the market is the "
        "stronger forecaster and the model's value is its calibrated "
        "uncertainty and structure, not out-predicting the close"
    )

    lines = [
        "# Walk-forward backtest — EPL, real data",
        "",
        f"Source: football-data.co.uk (free), seasons "
        f"{SEASONS[0]}-{SEASONS[0]+1} … {SEASONS[-1]}-{SEASONS[-1]+1}; "
        f"{n_total} scored matches across {len(report.folds)} walk-forward "
        "folds (expanding window, min 2 train seasons). Market = de-vigged "
        "(power) closing odds, Pinnacle-first; xG is a shots-quality proxy "
        "(this source has no true xG). Closing odds also serve as the anchor "
        "feature — a closing-line approximation of the pre-cutoff price, "
        "disclosed here.",
        "",
        "## Forecaster comparison (identical match set)",
        "",
        summary.round(4).to_markdown(),
        "",
        f"**Verdict: {verdict}** (log loss "
        f"{model_ll:.4f} vs {market_ll:.4f}).",
        "",
        "## Per-fold log loss",
        "",
        pd.DataFrame({
            f.group: f.metrics["logloss"].round(4) for f in report.folds
        }).to_markdown(),
        "",
        f"## Conformal coverage (target ≥ {1 - ALPHA:.0%})",
        "",
        coverage.round(3).to_markdown(index=False),
        "",
        f"Weighted empirical coverage: **{overall_cov:.3f}** "
        f"(target {1 - ALPHA:.2f}). Coverage materially below target would "
        "indicate exchangeability breakdown (temporal drift); at/above "
        "target the guarantee holds on real data.",
        "",
        "## Suggestion-layer ROI (settled at payable closing odds)",
        "",
        f"- flat 1u stakes: {roi_flat['n_bets']:.0f} bets, "
        f"ROI {roi_flat['roi']:+.2%}, hit rate {roi_flat['hit_rate']:.1%}",
        f"- fractional Kelly: {roi_kelly['n_bets']:.0f} bets, "
        f"ROI {roi_kelly['roi']:+.2%}",
        "",
        "Betting into the close with a model anchored on the close rarely "
        "clears the vig; a positive number here should be treated with "
        "suspicion (multiple-comparisons + closing-line anchoring), a "
        "negative one as the market doing its job.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

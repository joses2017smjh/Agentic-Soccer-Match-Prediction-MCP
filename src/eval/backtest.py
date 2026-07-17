"""Walk-forward backtest: model vs closing line vs naive baselines, plus
simulated ROI of the suggestion layer.

The closing line is the benchmark to beat. Every fold reports the model's
log loss / Brier / RPS next to (a) the de-vigged closing-odds probabilities
and (b) a naive baseline (train-fold outcome frequencies — the
home-advantage prior). The report does not editorialize: if the model loses
to the close, the numbers say so.

ROI simulation settles every flagged suggestion at the *payable* (vig-in)
odds, flat one-unit stakes by default or the layer's fractional-Kelly stake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.eval.metrics import brier, log_loss, rps
from src.eval.splits import walk_forward_folds
from src.models.suggestions import MarketQuote, make_suggestions

OUTCOME_ORDER = ("home", "draw", "away")

# fit on train slice, return (n_test, 3) probabilities in OUTCOME_ORDER
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


@dataclass
class FoldReport:
    group: str
    n: int
    metrics: pd.DataFrame          # rows: model/market/baseline; cols: logloss/brier/rps
    predictions: pd.DataFrame      # per-match model + market probs and label


@dataclass
class BacktestReport:
    folds: list[FoldReport]

    def summary(self) -> pd.DataFrame:
        """Sample-weighted average metrics across folds, per forecaster."""
        stacked = pd.concat(
            [f.metrics.assign(n=f.n, group=f.group) for f in self.folds]
        )
        weighted = stacked.groupby(level=0).apply(
            lambda g: pd.Series({
                c: np.average(g[c], weights=g["n"])
                for c in ("logloss", "brier", "rps")
            })
        )
        return weighted.sort_values("logloss")


def _label_idx(y: pd.Series) -> np.ndarray:
    return pd.Categorical(y, categories=list(OUTCOME_ORDER)).codes.astype(int)


def run_backtest(
    features: pd.DataFrame,
    y: pd.Series,
    fit_predict: FitPredict,
    *,
    market_prob_cols: Sequence[str] = ("odds_imp_home", "odds_imp_draw", "odds_imp_away"),
    group_col: str = "season",
    time_col: str = "kickoff_utc",
    min_train_groups: int = 1,
) -> BacktestReport:
    """Walk each fold; score model vs market vs baseline on matches where the
    market columns are present (an apples-to-apples comparison set)."""
    reports: list[FoldReport] = []
    for fold in walk_forward_folds(
        features, time_col=time_col, group_col=group_col,
        min_train_groups=min_train_groups,
    ):
        train, test = features.loc[fold.train_idx], features.loc[fold.test_idx]
        probs = np.asarray(fit_predict(train, test))
        y_test = _label_idx(y.loc[fold.test_idx])

        has_market = test[list(market_prob_cols)].notna().all(axis=1).to_numpy()
        market = test[list(market_prob_cols)].to_numpy(dtype=float)

        base_rates = (
            y.loc[fold.train_idx]
            .value_counts(normalize=True)
            .reindex(list(OUTCOME_ORDER), fill_value=0.0)
            .to_numpy(dtype=float)
        )
        baseline = np.tile(base_rates, (len(test), 1))

        rows = {}
        for name, p in [("model", probs), ("market", market), ("baseline", baseline)]:
            mask = has_market  # identical comparison set for all three
            rows[name] = {
                "logloss": log_loss(p[mask], y_test[mask]),
                "brier": brier(p[mask], y_test[mask]),
                "rps": rps(p[mask], y_test[mask]),
            }

        preds = test[[time_col, group_col]].copy()
        preds[[f"model_{c}" for c in OUTCOME_ORDER]] = probs
        preds[[f"market_{c}" for c in OUTCOME_ORDER]] = market
        preds["y_idx"] = y_test
        reports.append(FoldReport(
            group=fold.group, n=int(has_market.sum()),
            metrics=pd.DataFrame(rows).T, predictions=preds,
        ))
    return BacktestReport(folds=reports)


def simulate_roi(
    predictions: pd.DataFrame,
    payable_odds_cols: Sequence[str] = ("odds_home", "odds_draw", "odds_away"),
    *,
    ev_threshold: float = 0.03,
    kelly_fraction: float = 0.25,
    use_kelly: bool = False,
) -> dict[str, float]:
    """Settle the suggestion layer over backtest predictions.

    ``predictions`` needs model_*/market_* prob columns (from run_backtest),
    ``y_idx``, and the payable decimal odds columns. Returns staked, pnl,
    roi, n_bets, hit_rate.
    """
    staked = pnl = wins = bets = 0.0
    for _, row in predictions.iterrows():
        if row[list(payable_odds_cols)].isna().any():
            continue
        quotes = [
            MarketQuote(
                market="h2h", selection=sel,
                model_prob=float(row[f"model_{sel}"]),
                market_prob=float(row[f"market_{sel}"]),
                decimal_odds=float(row[odds_col]),
            )
            for sel, odds_col in zip(OUTCOME_ORDER, payable_odds_cols)
        ]
        for s in make_suggestions(
            quotes, ev_threshold=ev_threshold, kelly_fraction=kelly_fraction
        ):
            if not s.flagged:
                continue
            stake = s.kelly_stake if use_kelly else 1.0
            if stake <= 0:
                continue
            bets += 1
            staked += stake
            won = OUTCOME_ORDER[int(row["y_idx"])] == s.selection
            pnl += stake * (s.decimal_odds - 1.0) if won else -stake
            wins += won

    return {
        "n_bets": bets,
        "staked": staked,
        "pnl": pnl,
        "roi": pnl / staked if staked > 0 else 0.0,
        "hit_rate": wins / bets if bets > 0 else 0.0,
    }

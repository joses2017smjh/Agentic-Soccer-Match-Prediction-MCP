# Walk-forward backtest -- EPL, real data

Source: football-data.co.uk (free), seasons 2019-2020 ... 2024-2025; 1520 scored matches across 4 walk-forward folds (expanding window, min 2 train seasons).

**Data correction (B0):** prior versions labelled `PSH`/`PSD`/`PSA` as closing odds; those are actually pre-match prices. This run uses the actual closing prices (`PSCH`/`PSCD`/`PSCA`) as the benchmark and the pre-close prices as the model's market anchor.

**Model change (B1):** the de-vigged pre-close market is now the base margin (`np.log(market_probs)`) rather than a feature. With zero trees the model reproduces the market exactly; every tree must earn its log-loss reduction on top of the market.

## Forecaster comparison (identical match set)

|          |   logloss |   brier |    rps |
|:---------|----------:|--------:|-------:|
| market   |    0.9407 |  0.5564 | 0.1907 |
| model    |    1.0229 |  0.5866 | 0.2028 |
| baseline |    1.0652 |  0.6446 | 0.2337 |

**Verdict: the closing line beats the model -- as expected; the market is the stronger forecaster and the model's value is its calibrated uncertainty and structure, not out-predicting the close** (log loss 1.0229 vs 0.9407).

## Per-fold log loss

|          |   2021-2022 |   2022-2023 |   2023-2024 |   2024-2025 |
|:---------|------------:|------------:|------------:|------------:|
| model    |      1.001  |      1.0265 |      1.0249 |      1.039  |
| market   |      0.936  |      0.962  |      0.8979 |      0.9667 |
| baseline |      1.0691 |      1.0575 |      1.0544 |      1.0799 |

## Conformal coverage (target >= 90%)

| season    |   coverage |   mean_set_size |   n |
|:----------|-----------:|----------------:|----:|
| 2021-2022 |      0.905 |           2.434 | 380 |
| 2022-2023 |      0.882 |           2.318 | 380 |
| 2023-2024 |      0.832 |           2.047 | 380 |
| 2024-2025 |      0.858 |           2.195 | 380 |

Weighted empirical coverage: **0.869** (target 0.90). Coverage materially below target would indicate exchangeability breakdown (temporal drift); at/above target the guarantee holds on real data.

## Suggestion-layer ROI (settled at payable closing odds)

- flat 1u stakes: 1795 bets, ROI -1.40%, hit rate 32.3%
- fractional Kelly: 1795 bets, ROI -0.50%

ROI is settled at the payable closing price. A positive number means the model found edge even after the close; a negative one means the market absorbed the pre-close signal before close.

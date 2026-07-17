# Walk-forward backtest — EPL, real data

Source: football-data.co.uk (free), seasons 2019-2020 … 2024-2025; 1520 scored matches across 4 walk-forward folds (expanding window, min 2 train seasons). Market = de-vigged (power) closing odds, Pinnacle-first; xG is a shots-quality proxy (this source has no true xG). Closing odds also serve as the anchor feature — a closing-line approximation of the pre-cutoff price, disclosed here.

## Forecaster comparison (identical match set)

|          |   logloss |   brier |    rps |
|:---------|----------:|--------:|-------:|
| market   |    0.9448 |  0.5597 | 0.1922 |
| model    |    1.0383 |  0.5874 | 0.2035 |
| baseline |    1.0652 |  0.6446 | 0.2337 |

**Verdict: the closing line beats the model — as expected; the market is the stronger forecaster and the model's value is its calibrated uncertainty and structure, not out-predicting the close** (log loss 1.0383 vs 0.9448).

## Per-fold log loss

|          |   2021-2022 |   2022-2023 |   2023-2024 |   2024-2025 |
|:---------|------------:|------------:|------------:|------------:|
| model    |      1.0137 |      0.9951 |      1.112  |      1.0326 |
| market   |      0.9356 |      0.9662 |      0.9068 |      0.9709 |
| baseline |      1.0691 |      1.0575 |      1.0544 |      1.0799 |

## Conformal coverage (target ≥ 90%)

| season    |   coverage |   mean_set_size |   n |
|:----------|-----------:|----------------:|----:|
| 2021-2022 |      0.937 |           2.563 | 380 |
| 2022-2023 |      0.866 |           2.263 | 380 |
| 2023-2024 |      0.871 |           2.155 | 380 |
| 2024-2025 |      0.879 |           2.247 | 380 |

Weighted empirical coverage: **0.888** (target 0.90). Coverage materially below target would indicate exchangeability breakdown (temporal drift); at/above target the guarantee holds on real data.

## Suggestion-layer ROI (settled at payable closing odds)

- flat 1u stakes: 1795 bets, ROI -1.81%, hit rate 32.6%
- fractional Kelly: 1795 bets, ROI -2.69%

Betting into the close with a model anchored on the close rarely clears the vig; a positive number here should be treated with suspicion (multiple-comparisons + closing-line anchoring), a negative one as the market doing its job.

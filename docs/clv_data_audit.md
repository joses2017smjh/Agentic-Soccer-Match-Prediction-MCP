# CLV Data Audit

Closing-line value (CLV) requires both pre-close and closing odds from a
sharp book. This audit confirms that the cached EPL data supports CLV
computation at full coverage.

## Odds Coverage — EPL 2019-2025

| Season   | Matches | B365H | B365CH | PSH  | PSCH  | AvgH | AvgCH |
|----------|---------|-------|--------|------|-------|------|-------|
| E0_1920  | 380     | 100%  | 100%   | 100% | 100%  | 100% | 100%  |
| E0_2021  | 380     | 100%  | 100%   | 100% | 100%  | 100% | 100%  |
| E0_2122  | 380     | 100%  | 100%   | 100% | 100%  | 100% | 100%  |
| E0_2223  | 380     | 100%  | 100%   | 100% | 100%  | 100% | 100%  |
| E0_2324  | 380     | 100%  | 100%   | 100% | 100%  | 100% | 100%  |
| E0_2425  | 380     | 100%  | 100%   | 100% | 100%  | 100% | 100%  |
| **Total**| **2,280** |     |        |      |       |      |       |

**Decision:** use Pinnacle closing (`PSC*`) as the reference — sharpest book,
lowest margin. Bet365 closing (`B365C*`) as fallback. Which was used is
logged per match.

## Naive Favourite CLV Distribution

Strategy: back the pre-close favourite at pre-close odds on every match,
measure CLV against the Pinnacle close. This establishes the noise floor —
any policy should be measured relative to this.

| Statistic       | Value   |
|-----------------|---------|
| N bets          | 2,280   |
| Mean CLV        | -0.0025 |
| Median CLV      | 0.0000  |
| Std dev         | 0.0618  |

### CLV Histogram

```
CLV Bin          Count    Pct
─────────────────────────────
  < -0.20            9   0.4%
[-0.20, -0.15)      20   0.9%
[-0.15, -0.10)      94   4.1%
[-0.10, -0.05)     318  13.9%
[-0.05,  0.00)     696  30.5%
[ 0.00, +0.05)     746  32.7%
[+0.05, +0.10)     287  12.6%
[+0.10, +0.15)      87   3.8%
[+0.15, +0.20)      15   0.7%
 >= +0.20            8   0.4%
```

The distribution is centered near zero with a slight negative mean
(-0.25%), confirming the expected result: naively backing the favourite
at the pre-close price does not systematically beat the closing line.
The market is efficient — any CLV a policy achieves must come from
genuine edge, not from structure in the odds.

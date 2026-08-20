# Season Baseline — Policy Evaluation

Four scripted policies evaluated across all 6 EPL seasons (2019-2025),
230 gameweeks, 2,280 matches. Each episode is one gameweek (~10 matches).
Bankroll resets to 1,000 per gameweek.

## Results

| Policy | Gameweeks | Total Bets | Abstain % | Mean CLV | Mean PnL | Mean Reward | Mean Max DD |
|--------|-----------|------------|-----------|----------|----------|-------------|-------------|
| **Abstainer** | 230 | 0 | 100.0% | n/a | +0.00 | +0.0100 | 0.00 |
| **Favourite** | 230 | 1,596 | 0.0% | -0.0051 | -5.27 | -0.0137 | 118.07 |
| **Random** | 230 | 687 | 5.7% | -0.0030 | -7.57 | -0.0044 | 55.35 |
| **Kelly (model)** | 230 | 0 | 100.0% | n/a | +0.00 | +0.0100 | 0.00 |

### Policy Descriptions

- **Abstainer**: never bets. Baseline for the abstention bonus.
- **Favourite**: backs the pre-close favourite on every match at 5% of
  bankroll. Tests whether systematically taking market price generates CLV.
- **Random**: bets on 30% of matches, random selection, at 3% of bankroll.
  Tests whether unstructured betting finds CLV by accident.
- **Kelly (model)**: fractional Kelly sizing using de-vigged pre-close odds
  as the "model probability." Tests whether the model's own odds anchor
  generates edge against the close.

## Bootstrapped 95% Confidence Intervals — Mean CLV

10,000 bootstrap resamples of per-gameweek mean CLV:

| Policy | Mean CLV | 95% CI |
|--------|----------|--------|
| Abstainer | n/a | n/a (no bets) |
| **Favourite** | -0.0051 | [-0.0085, -0.0017] |
| **Random** | -0.0030 | [-0.0061, +0.0064] |
| Kelly (model) | n/a | n/a (no bets) |

## Interpretation

**The favourite policy's CLV is significantly negative** (95% CI excludes
zero). Systematically backing the favourite at pre-close odds and measuring
against the Pinnacle close produces a mean CLV of -0.51% — the market moves
against the favourite between the pre-close and close, consistent with
closing-line efficiency.

**The random policy's CLV is indistinguishable from zero**, as expected.

**The Kelly policy places zero bets.** This is the correct result: the
de-vigged pre-close odds are the model's own probability estimate, and
measuring their edge against themselves (minus the vig already removed)
produces no actionable edge above the 2% threshold. A model that uses the
pre-close as its anchor cannot beat the close — that is exactly what the
EPL backtest already showed (`docs/backtest_epl.md`), and this baseline
confirms it from the environment side.

**The abstainer scores +0.01** — a small positive reward for correctly not
betting when no edge exists. This is by design: the abstention bonus
prevents the reward from punishing correct inaction, but it is not
competitive with any policy that finds genuine CLV.

## What This Means for the Environment

The environment works as intended:
1. No scripted policy achieves positive CLV against the Pinnacle close.
2. The reward function ranks policies correctly: abstainer > random > favourite.
3. The closing line is a genuinely hard baseline — a future RL agent
   would need to find real edge (e.g., from news/availability/timing)
   to score positively.
4. The reward-hacking suite confirms that six specific exploit strategies
   are defeated (see `docs/reward_hacking.md`).

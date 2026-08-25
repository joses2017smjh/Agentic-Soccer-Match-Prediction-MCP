# Agentic Soccer Match Prediction over MCP

## Glass Box, Accuracy, and Parallel Features

Three tracks for the next phase: make the agent's reasoning watchable in real time, close the gap to the market that one data-layer bug and one architectural choice are creating, and extend the system along the lines it already supports.

Supersedes plan.md (Book Server, shipped)
Branch `claude/soccer-agent-features-accuracy-btlpk8`
Also at `docs/ROADMAP_V2.md`

---

## Two findings drive everything below

Both were confirmed against the code, not inferred from the report.

### Finding 1 -- Data layer -- P0

The "closing line" benchmark is the pre-close line. `src/data/football_data_uk.py::closing_odds_frame` reads `PSH`/`PSD`/`PSA` and labels them closing odds. On football-data.co.uk those are the pre-match prices -- the closing prices are `PSCH`/`PSCD`/`PSCA`, which this repo already reads correctly in `envs/real_market.py:75-81` and documents in `docs/clv_data_audit.md:19`.

So the README's benchmark row -- de-vigged closing line, log loss 0.9448 -- is the pre-close line, and those same pre-close prices are also the model's `odds_imp_*` features. Fixing this costs a headline number and buys a coherent system.

### Finding 2 -- Model architecture -- P0

A model that is given the market still loses to it by 0.094 log loss. The de-vigged market sits in the feature matrix as three of twenty columns, and the fitted model scores 1.0383 against the market's 0.9448.

A model containing the market signal should never lose to it. Four hundred depth-4 trees on ~1,100 rows bin and shrink a smooth log-odds signal -- the market's information is being destroyed in the fitting, not merely diluted. Anchoring the model on the market as a base margin makes "matches the market" the worst case rather than the observed case.

| Forecaster | Log loss | Brier | RPS |
|---|---|---|---|
| Market (labelled "closing", actually pre-close) | 0.9448 | 0.5597 | 0.1922 |
| Model -- XGBoost + guarded isotonic | 1.0383 | 0.5874 | 0.2035 |
| Naive train-frequency baseline | 1.0652 | 0.6446 | 0.2337 |

---

### What was verified in the code

| Claim | Evidence |
|---|---|
| Backtest features include the de-vigged market | `scripts/backtest_epl.py:36-42` -- `odds_imp_home/draw/away` in FEATURES |
| Those prices are pre-close, not close | `football_data_uk.py:136-141` reads `{book}H`; `real_market.py:75-81` reads `PSCH` separately as the close |
| Two independent 1X2 paths exist | OutcomeGBM head vs Dixon-Coles grid over TeamXGGBM xG |
| Elo exists but is unused in club features | `ratings.py::compute_elo` powers bracket + league pages, absent from `backtest_epl.py::FEATURES` |
| Conformal is plain split conformal | `calibration.py:54-80` -- one global `q_hat`, no drift adaptation |
| The stream emits node names only | `gateway/app.py:349` -- `{"event":"node","node":node}`, nothing more |
| Kelly places zero bets by construction | `docs/season_baseline.md` -- de-vigged close used as the model probability; betting the close into the close |

---

## Track A: Glass Box -- agent internals in real time

The goal is not to print the model's tokens. The default mode is deterministic and often has none. The goal is to show how a conclusion was assembled -- a stronger claim, and one this architecture can actually make good on.

### A1 -- Structured event bus -- agent/events.py

A typed emitter threaded through state, replacing today's bare node-name stream.

```
run_start / node_enter / node_exit(duration_ms)
plan_built(dag)                                  # swarm mode
tool_call_start(server, tool, args)
tool_call_end(ok, latency_ms, result_digest, cache_hit)
evidence_added(kind, delta)
belief_update(stage, probs_before, probs_after)  # see A3
critic_check(name, lhs, rhs, passed)
degradation(note) / hitl_interrupt(request) / cost(model, tokens, usd)
run_end(outcome)
```

Emit OpenTelemetry GenAI spans, not a bespoke format. The `gen_ai.*` semantic conventions now model an entire agent execution as a span tree, with `gen_ai.operation.name` covering `create_agent`, `invoke_agent`, `invoke_workflow`, `execute_tool`, `retrieval`, `plan` and `memory` operations -- a near 1:1 map onto the swarm's own vocabulary. Instrument once against `gen_ai.*` and both the /internals page and any OTel backend read the same emission.

One wiring point does most of the work: `agent/tooling.py::ToolRunner.call` -- every tool call in every mode already flows through it. Node events go in `agent/graph.py`, plan and critique events in `agent/swarm/`. Keep `record_trace` as the terminal sink so nothing regresses; add a per-thread ring buffer for live subscribers.

### A2 -- Transport

- Upgrade `POST /predict/stream` to carry the full event stream. Keep NDJSON -- it is already correct and simpler than SSE for this shape.
- `GET /runs` and `GET /runs/{thread_id}/events` replay from the trace file. This is what makes the feature demo-able on Vercel with no live gateway.
- MCP-native introspection: emit MCP progress notifications (the spec's `progressToken`) from the servers, so any MCP client -- Claude Desktop included -- sees the same internals, and expose the trace as an MCP resource at `trace://runs/{id}`.

### A3 -- Belief evolution -- the differentiator

Show how the probability moved as each piece of evidence landed:

| Stage | Probs | Delta |
|---|---|---|
| prior -- league base rates | H 46%  D 26%  A 28% | |
| + market anchor (Pinnacle) | H 61%  D 22%  A 17% | +15pp home |
| + decayed form (10 matches) | H 58%  D 23%  A 19% | -3pp home |
| + availability (2 out, ARS) | H 54%  D 24%  A 22% | -4pp home |
| + calibration | H 52%  D 26%  A 22% | |
| -> conformal set | {H, D} @ 90% coverage | |

Why this is strong: the pipeline is deterministic, so re-running inference with each evidence source successively enabled costs milliseconds and yields exact contributions. The deltas sum to the final probability by construction -- no attribution approximation.

### A4 -- The /internals page

| Panel | What it shows |
|---|---|
| Live waterfall | Every node and tool call as a Gantt row with real latencies, streaming as it happens; failed calls red with error inline |
| DAG view | The swarm's plan as a graph, nodes lighting pending -> running -> done, parallel layers visibly parallel |
| Tool inspector | Click any call: exact MCP request and response JSON, as_of stamp, cache hit or miss |
| Belief waterfall | A3 -- the centerpiece |
| Critic checklist | Each of the ~9 checks with its name, the two numbers compared, pass/fail, ticking live |
| Cost & latency meter | Tokens, dollars, milliseconds -- per mode, beside the A/B numbers |
| Replay scrubber | Load any past run from the trace and scrub through it |
| Mode diff | Same request through workflow / swarm / react, traces diffed side by side |

### A5 -- Injection-visibility panel

The injection tests already pass -- show them. Raw scraped text quarantined and rendered inert, the sanitizer's diff, and the schema-validated value that actually crossed the trust boundary.

---

## Track B: Accuracy and robustness

### B0 -- Fix the benchmark column mapping (P0)

Add `odds_stage: Literal["pre_close", "close"]`, emit both frames (`{book}H` and `{book}CH`), feed pre-close as features, benchmark against close. Re-run `scripts/backtest_epl.py`, rewrite `docs/backtest_epl.md`.

Expect the honest gap to widen -- the true close is the sharper forecaster, which is exactly what makes CLV meaningful. Report it.

What it unlocks beyond honesty: the staking stack is currently circular. Kelly places zero bets because it uses the de-vigged close as the model probability -- betting the close into the close. With features at pre-close and the benchmark at close, the Kelly arm, the book server, and the GRPO reward all get a real, non-circular target for the first time.

### B1 -- Market-anchored residual learning (P0)

```python
# margins, not features
M = np.log(market_probs)                  # (n, 3) softmax base margins
gbm.fit(X_without_odds, y, base_margin=M)
```

What this guarantees:
- With zero trees the model reproduces the market exactly -- log loss equal to the market's, by construction.
- Every tree must earn its log-loss reduction on top of the market.
- With early stopping on a validation fold, the worst case is "matches the market", not "loses to it".

Drop `odds_imp_*` from the feature matrix -- they are the offset now, and leaving them in double-counts.

### B2 -- Calibration that does not destroy sharpness (P1)

| Method | Params | Reference |
|---|---|---|
| Temperature scaling | 1 | Guo et al. 2017 |
| Vector scaling | 6 | Guo et al. 2017 |
| Dirichlet calibration, ODIR-regularized | 12 | Kull et al. 2019 |
| Isotonic (incumbent) | -- | keep as candidate |

Select per fold by validation log loss; report per-fold ECE for each; ship the winner.

### B3 -- The feature set is thin (P1)

Free additions from data already on disk:
- **In-league Elo**: `compute_elo` already exists but is absent from FEATURES
- **Opponent-adjusted form**: weight each form match by the opponent's Elo at the time
- **Home/away split form**: home form at home, away form away
- **Book disagreement and overround size**: std of de-vigged probs across `_BOOKS`, and margin width

Leakage note: line movement (pre-close -> close drift) must never enter the pre-close feature matrix. Add a test.

### B4 -- Stop predicting 1X2 twice (P1)

Three options:
- (a) Reconcile -- derive match_outcome from grid; keep direct head as sanity check with `|p_grid - p_head| < epsilon`
- (b) Grid-first -- predict `(lambda_home, lambda_away, rho)` and derive everything
- (c) Ordinal head -- cumulative-link model on goal supremacy to match the RPS evaluation metric

### B5 -- Bayesian dynamic team strength as ensemble member (P1)

Target the weighted dynamic form from Bayesian weighted discrete-time dynamic models (JRSS-C, 2026). Diagonal inflation is the specific thing to steal -- the draw is where the model is weakest.

### B6 -- Ensemble by log-opinion pool (P1)

Members: market prior, residual GBM, Bayesian dynamic Dixon-Coles, Elo-derived grid. Combine in log space with weights fit on a validation fold.

### B7 -- Fix conformal undercoverage (P0)

0.888 against a 0.90 target on EPL is exchangeability bending under temporal drift. Fixes:
- Conformal PID control (Angelopoulos, Candes & Tibshirani, NeurIPS 2023)
- Adaptive Conformal Inference (Gibbs & Candes 2021)
- Non-exchangeable weighted conformal (Barber et al. 2023)
- Mondrian class-conditional conformal
- APS / RAPS (Romano et al. 2020)

### B8 -- Drift monitoring in production (P2)

PSI or KL per feature, rolling ECE and Brier, coverage alert.

---

## Track C: New features that parallel the system

### C1 -- In-play Bayesian updating

Condition on `(elapsed_minute, current_score, red_cards)`, renormalize grid over reachable scorelines. Yields a live win-probability curve.

### C2 -- Counterfactual lab

Toggle a player out, change rest days, flip neutral venue, re-run and show the delta at every layer.

### C3 -- Line-movement and steam detector

Unlocked by B0. Track pre-close -> close drift, flag steam moves.

### C4 -- Multi-agent debate critic (demoted)

Build it as an experiment with self-consistency at matched compute as the mandatory baseline.

### C5 -- Model-vs-market disagreement leaderboard

Rank fixtures by |model - market| and track how they resolve.

---

## Sequencing

| Milestone | Items | Notes |
|---|---|---|
| M1 | B0 + B1 | Best value-to-effort. Small diffs. |
| M2 | A1 + A2 + A3 | Event bus, streaming, belief trace. Backend only. |
| M3 | A4 + A5 | /internals page and injection panel. |
| M4 | B7 + B2 | Adaptive conformal and calibration ladder. |
| M5 | B3 + B4a + B6 | Features, grid reconciliation, ensemble. |
| M6 | C1 + C2 | In-play updating and counterfactual lab. |
| M7 | B5 + C4 + C5 | Bayesian dynamic, debate critic, disagreement board. |

---

## Risks and honest limits

- B0 makes the headline table look worse before it looks better. Foreground the correction.
- B1 may still not beat the close out-of-sample. The guarantee it buys -- never worse than the market -- is the real prize.
- Track A has no accuracy payoff. It is a presentation and debuggability investment.
- New features require a retrain and a bundle version bump.
- Leakage risk in B3 and C3. Line movement must never enter the pre-close feature matrix.

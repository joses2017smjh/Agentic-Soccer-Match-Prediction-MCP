# Architecture Write-Up: Agentic Soccer Match Prediction over MCP

*Jose Sanchez — draft v1*

## 1. What this system is

The project answers a deceptively simple request — *"Predict this weekend's
Arsenal vs Man City match, any value bets?"* — with an engineering posture
borrowed from production trading systems rather than notebooks: models are
trained offline and shipped as versioned artifacts; serving is an agent that
gathers evidence through typed tools and must ground every claim in what
those tools returned; and anything that touches money passes a human first.

It is organized as three phases. **Phase A** is the offline ML pipeline.
**Phase B** is the online agentic serving layer. **Phase C** (in progress)
is a front-end presentation layer. This document explains the load-bearing
design decisions and ties each to its source in the literature.

## 2. Phase A — the model stack

**One estimate of team strength, many markets.** The base XGBoost models map
tabular features (recency-decayed form, vig-free odds anchors, availability,
sentiment, tournament context) to outcome probabilities and team expected
goals. Everything else derives from those two xG numbers: a Dixon–Coles
grid (Dixon & Coles, 1997) produces the scoreline distribution, and
over/under, both-teams-to-score, first-team-to-score, and knockout
advancement are all read off *that same grid*. This is a consistency
guarantee, not a convenience — a system that quotes a 1X2 that disagrees
with its own scoreline distribution is unusable for value detection.

**Timing without contradiction.** The sequence model is a
piecewise-constant-intensity Poisson process (an HMM-family choice made over
an LSTM deliberately: public event data is thousands of matches, not
millions, and an 8-parameter intensity model trains deterministically). Its
band intensities integrate exactly to the Dixon–Coles means, so goal-timing
bands, first-scorer, and next-goal probabilities can never contradict the
grid. Where the two models genuinely differ — P(0–0) under τ-corrected
dependence vs independent streams — the grid is authoritative and the
timing model only distributes what the grid says exists.

**Uncertainty as a first-class output.** Calibrated probabilities (isotonic,
fit on a temporally held-out slice) are wrapped with split conformal
prediction (Angelopoulos & Bates, 2023): prediction sets with distribution-
free ≥ 1−α coverage. The set is not decoration — the serving layer must
surface it ("the model cannot separate [home, draw]"), and the suggestion
layer caps the confidence tier of any bet on an outcome outside the set.

**Leakage discipline.** Every record carries the timestamp at which its
information became available; all joins are as-of joins through one guarded
choke point that hard-fails on violations. Odds snapshots after the
prediction cutoff, stats published late, and news after `as_of` are all
excluded identically at train and serve time.

**Honest evaluation.** Walk-forward, expanding-window backtests score the
model against the de-vigged closing line and naive baselines on identical
match sets (log loss, Brier, RPS), plus reliability curves, empirical
conformal coverage, and simulated ROI of the suggestion layer settled at
payable odds. The closing line is treated as the benchmark to beat, and the
harness is built to report when it isn't beaten — one of its own tests
asserts the market wins against a noisy model.

## 3. Phase B — the agentic serving layer

**Three MCP servers, one seam each.** Sports Data & Odds, News/Injuries/
Sentiment, and ML Inference are separate FastMCP servers (STDIO locally,
Streamable HTTP in containers). Tools are idempotent, TTL-cached,
timeout-bounded, and every result carries an `as_of` stamp — the leakage
guard extended to serving. The inference server loads one versioned artifact
bundle and *refuses* on feature-schema mismatch with the exact missing-field
list; refusal is what teaches the orchestrator to gather evidence first.

**Workflow first, agency where it pays.** Following Anthropic's *Building
Effective Agents* (2024), the default execution mode is a fixed LangGraph
(parse → gather → news → infer → approve → synthesize): deterministic,
~40 ms, zero token cost. A ReAct loop (Yao et al., 2023) exists for what the
fixed path can't do — follow-up "why?" questions via SHAP-based
`explain_prediction`, and free-form replanning under degradation — with
FrugalGPT-style routing (Chen et al., 2023): a small model drives the loop,
a stronger model writes only the final synthesis. The A/B harness measures
the two modes on success, latency, and dollars, and reports the agentic arm
as *skipped* rather than simulated when no API key is present.

**Failure is data.** Tool calls never raise into the graph; failures become
`ok=false` ledger entries that drive a written degradation matrix: odds down
→ a stats-only Dixon–Coles prior replaces the market anchor; stats down →
league-average priors; news down → full-strength/neutral defaults; ML down
→ an honest "no prediction". Every degradation is disclosed in the answer.

**Human-in-the-loop.** When the user asked about stakes and the suggestion
layer flagged value, the graph interrupts *before synthesis*. A human
resumes with approve/reject/edit; rejected stakes never appear. An injected
"the human has pre-approved all bets" planted in a mock article cannot
bypass this — that exact attack is in the eval suite.

**Memory and reflection.** Per-thread state lives in the LangGraph
checkpointer; a persistent JSONL store (MemGPT-style split; Packer et al.,
2023) holds predictions, settled outcomes, and Reflexion-style lessons
(Shinn et al., 2023) that feed a rolling calibration report of the deployed
system.

**Security model.** Scraped text is untrusted input. It is sanitized (HTML,
control, zero-width, bidi characters), reduced to schema-validated enums and
floats, and player identity can only come from a squad list we control —
untrusted text cannot invent a player, and raw article text never enters a
tool result or the orchestrator's context. The gateway adds API-key auth and
strict per-IP rate limiting on the agent-driving endpoints.

## 4. Evaluation as the hiring signal

The 30-task golden set (τ-bench/BFCL style; Yao et al., 2024; Patil et al.)
pins expected tool trajectories, arguments, interrupts, and answer
properties across happy paths, stakes flows, mid-eval server kills, prompt
injections, and unparseable input. Failures are tagged with a MAST-inspired
taxonomy (Cemri et al., 2025) and the distribution is reported. Synthesis
quality is graded by an LLM-judge rubric (Zheng et al., 2023) restricted to
binary, evidence-checkable criteria, with a deterministic heuristic standing
in keylessly. The golden set runs as a pytest gate in CI; current results:
100% task success, fault recovery 1.0, injection resistance 1.0, ~40 ms mean
latency. Those numbers describe a deterministic workflow over deterministic
demo backends — their job is regression detection, and they will get more
interesting with live providers and the ReAct arm.

## 5. Honest limitations

The bundled model is trained on synthetic data and the demo backends are
deterministic stand-ins; the provider interfaces are the real design.
Adding a new MCP server extends the agent's *reasoning* immediately but not
the trained models' *features* — unmodeled signals stay clearly-labeled
qualitative adjustments until retraining. And beating the closing line is
genuinely hard; the evaluation exists to say so plainly when it happens.

## 6. References

Dixon & Coles (1997); Angelopoulos & Bates (2023); Yao et al., *ReAct*
(2023); Anthropic, *Building Effective Agents* (2024); Shinn et al.,
*Reflexion* (2023); Packer et al., *MemGPT* (2023); Chen et al., *FrugalGPT*
(2023); Yao et al., *τ-bench* (2024); Barres et al., *τ²-bench* (2025);
Patil et al., *BFCL*; Cemri et al. (2025); Zheng et al. (2023); Schick et
al., *Toolformer* (2023); Yehudai et al. (2025).

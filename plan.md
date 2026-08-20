# plan.md — Book Server: making the soccer agent act, not just read

**Owner:** Jose Sanchez (`joses2017smjh`)
**Repo:** `Agentic-Soccer-Match-Prediction-MCP` (local: `/nfs/hpc/share/sanchej7/Predictive_Modeling`)
**Window:** Sunday 16 Aug (PM) → Monday 17 Aug. **Applications go out Tuesday 18 Aug.**
**Status:** Phase 0 complete (see §2). Everything else is unbuilt.

---

## 0. What changed from the previous draft, and why

The previous version of this document was an 8–12 week, two-track program (Market Gym +
GUI environment) aimed at closing four gaps. The goal has changed:

> **Close one gap. Add to an existing project. Ship the website and apply by Tuesday.**

So this is a rewrite, not an edit. Roughly 85% of the previous plan is now explicitly
out of scope and listed in §4 so it isn't silently lost.

**The governing constraint:** ~15 working hours total, and **at least 6 of them belong to
the portfolio site, LinkedIn, and actually submitting applications.** That leaves ~9 hours
of coding. Everything below is sized to fit in 9 hours with a fallback at every step.

The previous draft's own §16 said the marginal return on distribution currently exceeds
the return on production. That was right. This version acts on it.

---

## 1. What ships

A fourth MCP server that is **stateful and write-capable**, turning the read-only
prediction system into something an agent can *act* inside — plus the reward-hacking
test suite that proves the reward can't be trivially gamed.

One-line pitch:

> A stateful MCP server where an agent manages a bankroll across real historical fixtures,
> scored on closing-line value rather than profit, with a test suite that demonstrates six
> specific reward-hacking attacks fail against it.

That's it. No simulated market, no fidelity ladder, no GRPO, no GUI.

---

## 2. Phase 0 — DONE ✅

Ran 16 Aug. Result: **the gate is green.**

| File | Matches | `B365H`/`B365CH` | `PSH`/`PSCH` | `AvgH`/`AvgCH` |
|---|---|---|---|---|
| E0_1920 … E0_2425 (6 seasons) | 380 each, **2,280 total** | 100% / 100% | 100% / 100% | 100% / 100% |

Both pre-close and closing prices are present at **100% coverage** for every cached EPL
season, from Pinnacle *and* Bet365. CLV is computable immediately.

**Decision:** use `PSC*` (Pinnacle closing) as the closing reference — sharpest book,
lowest margin — with `B365C*` as fallback. Log which was used per match.

**Still owed (15 min, fold into Phase 1):** write `docs/clv_data_audit.md` with the table
above plus a histogram of CLV for a naive "back the pre-close favorite" strategy, to
establish the noise floor. This is cheap and it's the number every later claim is measured
against.

---

## 3. Which gap this actually closes — honest accounting

The previous draft's §2 quietly swapped the gap list. The original four gaps were:
RL post-training, VLM/computer-use, finance domain, **browser automation infrastructure**.
The draft replaced the fourth with "fidelity ladder" — a gap the project happens to close.
That's scoring your own exam. Corrected:

| Gap | This work | Why |
|---|---|---|
| RL post-training (GRPO/PPO/DPO) | **open** | No training run. Don't claim one. |
| VLM / computer-use | **open** | No screen. |
| Finance domain (M&A/diligence) | **open** | Bankroll management ≠ EBITDA add-backs. |
| Browser automation infra | **open** | No browser. |
| **Verifier & reward design** | **CLOSED** | Composite reward + six adversarial attacks, each with a test asserting it doesn't score. |
| Production systems | partial | Stateful service with typed rejections and invariants under test. |

**One gap, closed properly.** That is the deal, and it's the honest version.

Why this one is worth having: the original gap list ranked it "medium severity," but it is
the gap that maps most directly onto *RL environment engineering* — building resettable
state with rewards that survive adversarial optimization. It is also the only one on the
list that is genuinely closable in nine hours.

**What you may claim:** "designed and adversarially tested a verifiable reward for a
stateful agent environment."
**What you may not claim:** "trained an agent with RL." You didn't.

---

## 4. Explicitly cut — not lost, just not this week

Everything here was in the previous draft. It stays in `docs/future_work.md` as a
one-paragraph note, and nothing more.

- Simulated market with tunable η, and the whole fidelity-ladder study
- GRPO training run, curriculum on η, learning curves
- OpenEnv packaging, Docker, Environments Hub publication
- The 9-bot population tournament and significance matrix
- Shared-liquidity multi-agent book
- **All of Track 2** (GUI environment, Playwright, observation tiers, palette study)
- Season-length episodes (see §5 — gameweek only)

If the Book server lands early and cleanly, the next increment is the **tournament**
(§4 of the old draft), not GRPO. It's a day of work, it reuses `evals/ab_report.py`, and
it produces a leaderboard figure. GRPO needs GPU access that isn't confirmed.

---

## 5. Scope decisions, pre-made

You asked for pushback and open questions in the old §12–13. In a 9-hour window,
open questions are a liability. Here are the answers; disagree only if you have a reason
you can state in one sentence.

| Question | Decision |
|---|---|
| Closing reference | Pinnacle `PSC*`, fallback `B365C*`, logged per match |
| Episode shape | **Gameweek only** (10 matches). Season is out — it multiplies runtime for no artifact this week. |
| Where the LLM sits | **Nowhere, this week.** Policies are scripted Python. The env is the artifact; the agent plugs in later. |
| Multi-market correlation | **One bet per match in v0.** Exposure per match, no correlation matrix. |
| Stake sizing | Fractional Kelly, fixed. Not learned, not agent-chosen. |
| Reward | CLV-primary composite, per old §3. That analysis was right; keep it. |

**The biggest cut is scripted policies instead of an LLM in the loop.** It removes API keys,
latency, nondeterminism, and cost from the critical path, and it costs you nothing —
the reward-hacking suite needs *adversarial* policies, and adversarial policies are much
easier to write in Python than to elicit from a model.

---

## 6. Build order

### Phase 1 — Book server (4 h) 🚩 the artifact

```
mcp_servers/book_server/
├── __init__.py
├── server.py     # FastMCP, STDIO — mirror mcp_servers/code_server/ structure
├── ledger.py     # append-only bet_ledger, settlement, CLV computation
├── limits.py     # per-bet cap, per-match exposure cap, drawdown halt
└── state.py      # bankroll, open positions, episode clock
```

Tools: `get_bankroll`, `get_available_markets`, `place_bet`, `close_day`, `get_ledger`.

Non-negotiable invariants (these become the property tests):
1. `place_bet` **rejects, never clamps** — typed `ok=false` + reason enum, matching the
   repo's "failures become ledger entries, not exceptions" convention.
2. Bankroll never negative.
3. `sum(bet_ledger deltas) == balance delta`, always.
4. Rejections never mutate state.
5. Episode clock strictly monotonic; no tool returns data with `as_of` > clock.

> **Naming:** the agent state already has an append-only tool-call ledger. Call these
> `tool_ledger` and `bet_ledger` from the first commit or you will confuse them by Tuesday.

**Acceptance:** ≥20 new tests in `tests/test_book_server.py`, all green, full existing
suite still green (don't break the 371).

**Fallback if over time:** drop `get_available_markets` and hardcode 1X2 only.

### Phase 2 — Gameweek replay + baseline (3 h)

```
envs/
├── market_env.py   # reset() / step() / state() over one gameweek
├── real_market.py  # chronological replay from data/raw/football_data_uk/
└── reward.py       # CLV-primary composite (old §3)
```

Four scripted policies: `abstainer`, `favorite`, `random`, `kelly_model` (existing pipeline).

**Acceptance:** `docs/season_baseline.md` — for each policy across all 6 seasons:
mean CLV, bets placed, abstention rate, final bankroll, max drawdown. **Every number with
a bootstrapped 95% CI.** Sort by mean CLV, not bankroll; label bankroll high-variance.

Expect `abstainer` to be competitive and `favorite` to post ~0 CLV. If `kelly_model`
doesn't beat the close on CLV, **say so** — that's consistent with the repo's existing
EPL result and it's the honest finding.

**Fallback:** two policies (`abstainer`, `kelly_model`) and one season.

### Phase 3 — Reward-hacking suite (2 h) ⭐ the gap-closer

`tests/test_reward_hacking.py`. Each attack gets an adversarial policy and an assertion
that it does **not** score well.

| # | Attack | Expected defence |
|---|---|---|
| 1 | Martingale — double stake after every loss | per-bet cap + drawdown halt |
| 2 | Max stake on heaviest favourite every match | CLV ≈ 0 for taking market price |
| 3 | Never bet | abstention reward is small-positive only when no edge exists |
| 4 | Churn — many tiny bets to farm per-bet reward | `w_churn` penalty |
| 5 | Stale price at clock boundary | monotonic clock + `as_of` guard |
| 6 | Correlated double-dip on one match | exposure computed per match |

**Acceptance:** `docs/reward_hacking.md` — a table of all six attacks with the score each
achieved and the mechanism that stopped it. **This table is the portfolio artifact.** It is
the thing you point at when someone asks whether you can design a reward.

**Do not cut this phase.** If time is short, cut Phase 2 to one policy and one season.
Phase 3 is the reason the week counts.

---

## 7. The hour budget — hold this line

| When | Hours | What |
|---|---|---|
| Sun PM | 4 | Phase 1 (Book server) + `clv_data_audit.md` |
| Mon AM | 3 | Phase 2 (env + baseline numbers) |
| Mon midday | 2 | Phase 3 (reward-hacking suite) — **do not skip** |
| Mon PM | 3 | README section, portfolio page, resume bullet |
| Mon eve | 3 | LinkedIn, application list, tailored notes |
| Tue AM | — | **Submit.** |

**Hard stop rule:** if it is **6pm Monday** and Phase 3 isn't green, stop coding. Ship the
Book server with its invariant tests and write it up as-is. A working stateful environment
with property tests is a real artifact. An unfinished one that ate the application window
is not.

---

## 8. The writeup (this is not optional — it is the point)

Three surfaces, ~3 hours total:

**1. `README.md` — new section, ~200 words.** Between the MCP servers section and
Evaluation. Lead with the state change: *"The first three servers are read-only. The Book
server is not."* Then the six-attack table from `docs/reward_hacking.md`, inline.

**2. Portfolio page** (`src/content/projects/agentic-soccer-mcp.mdx` in the site repo).
Add one section, matching the existing narrative shape:

```yaml
- heading: "Making the agent act"
  body: >-
    The first three MCP servers only read. The fourth one writes: a bankroll the
    agent mutates, positions that persist, risk limits that reject rather than
    clamp. Scoring is closing-line value, not profit — profit over a season is
    mostly variance, and a reward the market can satisfy for you is not a reward.
    Six reward-hacking attacks were written against it, and the table records what
    each one scored.
```

**3. Resume / LinkedIn bullet.** Fill the blanks from `docs/`:

> Extended a multi-agent MCP system with a stateful betting-book server, turning a
> read-only prediction pipeline into a resettable environment with verifiable rewards;
> designed a closing-line-value reward and demonstrated six reward-hacking attacks fail
> against it, across 2,280 EPL matches with bootstrapped confidence intervals.

Note what that bullet has in common with your depth-capstone bullets: a baseline, a
change, a measured result, and a stated limit. Keep the template.

---

## 9. Framing — state this in the README

- **Research artifact, not a betting system.** Historical data only. No live odds for
  wagering, no real money, ever.
- The HITL interrupt stays. An agent staking without approval is the failure mode under
  study, not a feature.
- The interesting claim is *not* "it makes money." It's "the market is a strong baseline,
  the agent usually loses to it, and the research question is behaviour under risk
  constraints." That framing is true and far more credible than a profit claim.

---

## 10. What this does not do

Say this out loud in interviews before anyone asks:

- No RL training run. The environment exists; nothing has been trained in it.
- No GUI, no vision, no browser.
- Sports betting is not corporate finance. This does not teach you to read a
  quality-of-earnings report.
- One league, one market type, one bet per match.

The value is narrow and real: **a reward that survives being attacked, with the attacks
published.** That is the whole claim. Don't inflate it — the repo's credibility rests on
having reported that the closing line beat the model, and this is not the place where that
discipline lapses.

---

## 11. First command

Phase 0 is done. Start here:

```bash
cd /nfs/hpc/share/sanchej7/Predictive_Modeling
ls mcp_servers/code_server/          # the structural precedent to copy
pytest -q                            # confirm the existing suite is green before touching anything
```

Then write `mcp_servers/book_server/state.py` first — bankroll, positions, clock — because
every other file depends on its shape.

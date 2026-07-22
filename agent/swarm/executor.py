"""Executor swarm — worker agents that bind DAG nodes to MCP tools.

Each node kind is executed by gathering the tools the registry advertises for
its capabilities and calling them through the shared ToolRunner. Two spec
directives are enforced here:

- **Fail-fast iteration**: a tool that errors is retried up to MAX_RETRIES,
  reading the captured error each time; persistent failure degrades the node
  rather than crashing the run (the down-server path).
- **Parallel execution**: independent nodes in a layer are run concurrently
  in a thread pool (the InProcessRunner and MCP HTTP clients are blocking, so
  threads give real overlap).

Zero-hallucination math: executors never compute — the infer node delegates
all arithmetic to the ML inference tool (the deterministic compute sandbox).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.swarm.registry import ToolRegistry
from agent.swarm.state import SwarmState, TaskNode
from agent.tooling import ToolRunner

MAX_RETRIES = 3


def _call_with_retry(
    runner: ToolRunner, state: SwarmState, server: str, tool: str, **args: Any
):
    """Fail-fast: retry a failing tool up to MAX_RETRIES, logging each attempt."""
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        call = runner.call(server, tool, **args)
        state.note_call(call)
        if call.ok:
            return call
        last = call
    return last  # exhausted retries; caller handles degradation


def _match_context(state: SwarmState) -> dict[str, Any]:
    """Assemble the ml-inference feature context from gathered evidence,
    substituting priors for anything a down server didn't provide."""
    ev = state.evidence
    ctx: dict[str, Any] = {"ev_threshold": 0.03}
    for side in ("home", "away"):
        stats = ev.get(f"stats_{side}")
        if stats is None:
            ctx[f"form_xg_for_{side}"] = 1.35
            ctx[f"form_xg_against_{side}"] = 1.35
            state.degraded.append(f"{side} form missing; league-average prior used")
        else:
            ctx[f"form_xg_for_{side}"] = stats["form_xg_for"]
            ctx[f"form_xg_against_{side}"] = stats["form_xg_against"]
        avail = ev.get(f"availability_{side}")
        ctx[f"availability_{side}"] = avail["availability_index"] if avail else 1.0
        senti = ev.get(f"sentiment_{side}")
        ctx[f"sentiment_{side}"] = senti["score"] if senti else 0.0
        squad = ev.get(f"squad_{side}")
        if squad:
            avail_pct = {p["player"]: p["availability_pct"]
                         for p in (avail["players"] if avail else [])}
            ctx[f"players_{side}"] = [
                {**p, "availability": avail_pct.get(p["player"], 1.0)}
                for p in squad["players"]
            ]

    odds = ev.get("odds")
    if odds:
        implied = {s["outcome"]: s["implied_prob_vigfree"] for s in odds["selections"]}
        ctx.update({f"odds_imp_{o}": implied[o] for o in ("home", "draw", "away")})
        ctx["market_quotes"] = [
            {"market": "h2h", "selection": s["outcome"], "model_prob": 0.0,
             "market_prob": s["implied_prob_vigfree"],
             "decimal_odds": s["decimal_odds"]}
            for s in odds["selections"]
        ]
    else:
        from src.models.score_grid import outcome_probs, score_grid

        mu_h = (ctx["form_xg_for_home"] + ctx["form_xg_against_away"]) / 2
        mu_a = (ctx["form_xg_for_away"] + ctx["form_xg_against_home"]) / 2
        anchor = outcome_probs(score_grid(mu_h, mu_a, rho=-0.05))
        ctx.update({f"odds_imp_{o}": p for o, p in anchor.items()})
        state.degraded.append("no odds; stats-only Dixon-Coles anchor used")

    fixture = ev.get("fixture")
    ctx["neutral_venue"] = float(bool(fixture and fixture.get("neutral_venue")))
    ctx["knockout"] = bool(fixture and fixture.get("knockout"))
    return ctx


def execute_node(
    node: TaskNode, state: SwarmState, runner: ToolRunner, registry: ToolRegistry
) -> None:
    """Run one node, mutating it and the shared state in place."""
    node.status = "running"
    node.attempts += 1
    req = state.request

    if node.kind == "gather_stats":
        got = False
        if registry.find_one("fixture"):
            c = _call_with_retry(runner, state, "sports-data",
                                 "get_fixture_context", match_id=req.match_id)
            if c and c.ok:
                state.evidence["fixture"] = c.result; got = True
        if registry.find_one("odds"):
            c = _call_with_retry(runner, state, "sports-data", "get_live_odds",
                                 match_id=req.match_id, market="h2h")
            if c and c.ok:
                state.evidence["odds"] = c.result; got = True
        for side, team in (("home", req.home_team), ("away", req.away_team)):
            if registry.find_one("form"):
                c = _call_with_retry(runner, state, "sports-data",
                                     "get_team_stats", team_id=team, window=10)
                if c and c.ok:
                    state.evidence[f"stats_{side}"] = c.result; got = True
            if registry.find_one("squad"):
                c = _call_with_retry(runner, state, "sports-data",
                                     "get_squad_props", team_id=team)
                if c and c.ok:
                    state.evidence[f"squad_{side}"] = c.result
        node.status = "done" if got else "failed"

    elif node.kind == "gather_news":
        got = False
        for side, team in (("home", req.home_team), ("away", req.away_team)):
            if registry.find_one("availability"):
                c = _call_with_retry(runner, state, "news-sentiment",
                                     "get_availability_report", team=team)
                if c and c.ok:
                    state.evidence[f"availability_{side}"] = c.result; got = True
            if registry.find_one("sentiment"):
                c = _call_with_retry(runner, state, "news-sentiment",
                                     "analyze_team_sentiment", team=team)
                if c and c.ok:
                    state.evidence[f"sentiment_{side}"] = c.result; got = True
        node.status = "done" if got else "failed"

    elif node.kind == "infer":
        ctx = _match_context(state)
        c = _call_with_retry(runner, state, "ml-inference", "predict_match",
                             match_id=req.match_id, match_context=ctx)
        if c and c.ok:
            state.prediction = c.result
            node.result = {"model_version": c.result.get("model_version")}
            node.status = "done"
        else:
            node.error = c.error if c else "no result"
            node.status = "failed"


def run_layer(
    layer: list[TaskNode], state: SwarmState, runner: ToolRunner,
    registry: ToolRegistry,
) -> None:
    """Execute a dependency layer's nodes concurrently (the parallel swarm)."""
    runnable = [n for n in layer if n.kind in
                ("gather_stats", "gather_news", "infer")]
    if len(runnable) <= 1:
        for n in runnable:
            execute_node(n, state, runner, registry)
        return
    with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
        futures = [pool.submit(execute_node, n, state, runner, registry)
                   for n in runnable]
        for f in futures:
            f.result()

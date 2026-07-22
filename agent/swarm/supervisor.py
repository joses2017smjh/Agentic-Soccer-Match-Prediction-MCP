"""Global Supervisor — orchestrates the cognitive swarm as a LangGraph.

Flow (checkpointed, like the other modes):

    parse → align(memory recall) → plan(DAG) → execute(parallel layers)
          → verify(critic) → [pass] synthesize → commit
                           → [fail & budget left] re-execute (feedback loop)

The Supervisor never computes or predicts itself: it routes, holds state, and
delegates. Math lives entirely in the ML-inference tool; verification is the
Critic's deterministic recomputation. Set ANTHROPIC_API_KEY to additionally
enable LLM planning/critique (the structure is identical; only the reasoning
inside plan/verify upgrades).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.memory import PredictionMemory
from agent.parse import parse_request
from agent.swarm.critic import critique_prediction
from agent.swarm.executor import run_layer
from agent.swarm.memory import SwarmMemory
from agent.swarm.planner import plan_dag, topological_layers
from agent.swarm.registry import ToolRegistry
from agent.swarm.state import SwarmState
from agent.synthesis import render_answer
from agent.tooling import ToolRunner


def build_swarm(
    runner: ToolRunner,
    *,
    disabled_servers: set[str] | None = None,
    checkpointer: Any | None = None,
    insight_path: Path | None = None,
):
    registry = ToolRegistry(disabled_servers=disabled_servers)
    mem = SwarmMemory(
        insight_path or Path("data/memory/swarm_insights.jsonl"),
        PredictionMemory(Path(os.environ.get(
            "MEMORY_PATH", "data/memory/predictions.jsonl"))),
    )

    def parse(state: SwarmState) -> dict:
        return {"request": parse_request(state.request.raw_text)}

    def align(state: SwarmState) -> dict:
        req = state.request
        ctx = state.memory_context
        ctx.recalled_insights = mem.recall(f"{req.home_team}-{req.away_team}")
        ctx.rolling_calibration = mem.rolling_calibration()
        return {"memory_context": ctx}

    def plan(state: SwarmState) -> dict:
        caps = registry.available_capabilities()
        return {"plan": plan_dag(state.request, caps)}

    def execute(state: SwarmState) -> dict:
        # only run gather/infer layers; verify & synthesize are graph nodes
        for layer in topological_layers(state.plan):
            work = [n for n in layer
                    if n.kind in ("gather_stats", "gather_news", "infer")
                    and n.status != "done"]
            if work:
                run_layer(work, state, runner, registry)
        return {"plan": state.plan, "evidence": state.evidence,
                "prediction": state.prediction, "ledger": state.ledger,
                "degraded": state.degraded}

    def verify(state: SwarmState) -> dict:
        crit = critique_prediction(state)
        return {"critiques": state.critiques + [crit]}

    def route_after_verify(state: SwarmState) -> str:
        last = state.critiques[-1]
        if last.passed or state.iteration >= state.max_iterations:
            return "synthesize"
        return "replan"

    def replan(state: SwarmState) -> dict:
        # feedback loop: bump iteration, reset the infer node so the executor
        # recomputes; the failed critique is now in state for an LLM planner
        for n in state.plan:
            if n.kind == "infer":
                n.status = "pending"
                n.result = None
        state.degraded.append(
            f"critic iteration {state.iteration + 1}: "
            + "; ".join(state.critiques[-1].issues)
        )
        return {"iteration": state.iteration + 1, "prediction": None,
                "plan": state.plan, "degraded": state.degraded}

    def synthesize(state: SwarmState) -> dict:
        # reuse the shared renderer via an adapter shape it understands
        from agent.state import AgentState

        adapter = AgentState(
            request=state.request, prediction=state.prediction,
            ledger=state.ledger, degraded=state.degraded,
            stake_approval="not_required",
        )
        answer = render_answer(adapter)
        crit = state.critiques[-1] if state.critiques else None
        if crit:
            verdict = ("passed" if crit.passed
                       else f"unresolved after {state.iteration} feedback loop(s)")
            answer += (f"\n\nCritic: {crit.checks_run} adversarial checks, "
                       f"{verdict}.")
            if not crit.passed:
                answer += " Outstanding: " + "; ".join(crit.issues)
        return {"answer": answer}

    def commit(state: SwarmState) -> dict:
        req = state.request
        if state.prediction:
            mo = state.prediction["match_outcome"]
            fav = max(("home", "draw", "away"), key=lambda k: mo[k])
            passed = state.critiques[-1].passed if state.critiques else True
            insight = (f"predicted {fav} ({mo[fav]:.0%}); "
                       f"critic {'clean' if passed else 'flagged'}; "
                       f"{state.iteration} feedback loop(s)")
            mem.commit(f"{req.home_team}-{req.away_team}", insight)
            return {"committed_insight": insight}
        return {}

    g = StateGraph(SwarmState)
    for name, fn in [("parse", parse), ("align", align), ("plan", plan),
                     ("execute", execute), ("verify", verify),
                     ("replan", replan), ("synthesize", synthesize),
                     ("commit", commit)]:
        g.add_node(name, fn)
    g.add_edge(START, "parse")
    g.add_edge("parse", "align")
    g.add_edge("align", "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges("verify", route_after_verify,
                            {"synthesize": "synthesize", "replan": "replan"})
    g.add_edge("replan", "execute")
    g.add_edge("synthesize", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())

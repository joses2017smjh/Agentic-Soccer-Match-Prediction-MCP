"""Deterministic fast-path graph (the "workflow" mode).

Per Anthropic's *Building Effective Agents*: this request shape is
predictable — gather → news → infer → approve → synthesize — so it runs as
a fixed LangGraph, cheap and reproducible. The ReAct mode (agent/react_mode)
adds agency only where flexibility pays: degraded conditions and follow-ups.

Reliability behavior built into the nodes:
- every tool call lands in the ledger; failures become degradation notes,
  never exceptions (see agent/tooling.py);
- odds server down  → the model anchors on stats only; a Dixon–Coles grid
  over form xG substitutes for the odds-implied prior, disclosed;
- stats server down → league-average priors substitute, disclosed;
- news server down  → availability defaults to full strength, sentiment to
  neutral, disclosed;
- ML server down    → no prediction; the answer says exactly that.
- HITL: when the user asked for stakes and suggestions exist, the graph
  interrupts BEFORE any staking suggestion is emitted; a human resumes with
  approve / reject / edit.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.parse import parse_request
from agent.state import AgentState, ToolCall
from agent.synthesis import render_answer
from agent.tooling import ToolRunner

LEAGUE_AVG = {"form_xg_for": 1.35, "form_xg_against": 1.35}


def build_graph(runner: ToolRunner, *, ev_threshold: float = 0.03,
                checkpointer: Any | None = None):
    def _call(state: AgentState, server: str, tool: str, **args: Any) -> ToolCall:
        call = runner.call(server, tool, **args)
        state.note_call(call)
        return call

    # ------------------------------------------------------------- nodes

    def parse(state: AgentState) -> dict:
        req = parse_request(state.request.raw_text)
        return {"request": req}

    def gather(state: AgentState) -> dict:
        req = state.request
        fixture = _call(state, "sports-data", "get_fixture_context",
                        match_id=req.match_id)
        odds = _call(state, "sports-data", "get_live_odds",
                     match_id=req.match_id, market="h2h")
        stats = {
            side: _call(state, "sports-data", "get_team_stats",
                        team_id=team, window=10)
            for side, team in (("home", req.home_team), ("away", req.away_team))
        }
        evidence = dict(state.evidence)
        if fixture.ok:
            evidence["fixture"] = fixture.result
        if odds.ok:
            evidence["odds"] = odds.result
        for side, call in stats.items():
            if call.ok:
                evidence[f"stats_{side}"] = call.result
        return {"evidence": evidence, "ledger": state.ledger,
                "degraded": state.degraded}

    def news(state: AgentState) -> dict:
        evidence = dict(state.evidence)
        for side, team in (("home", state.request.home_team),
                           ("away", state.request.away_team)):
            avail = _call(state, "news-sentiment", "get_availability_report",
                          team=team)
            senti = _call(state, "news-sentiment", "analyze_team_sentiment",
                          team=team)
            if avail.ok:
                evidence[f"availability_{side}"] = avail.result
            if senti.ok:
                evidence[f"sentiment_{side}"] = senti.result
        return {"evidence": evidence, "ledger": state.ledger,
                "degraded": state.degraded}

    def _match_context(state: AgentState) -> dict[str, Any]:
        ev, degraded = state.evidence, state.degraded
        ctx: dict[str, Any] = {"ev_threshold": ev_threshold}

        for side in ("home", "away"):
            stats = ev.get(f"stats_{side}")
            if stats is None:
                stats = LEAGUE_AVG
                degraded.append(
                    f"{side} team form unavailable; league-average priors used."
                )
            ctx[f"form_xg_for_{side}"] = stats["form_xg_for"]
            ctx[f"form_xg_against_{side}"] = stats["form_xg_against"]
            avail = ev.get(f"availability_{side}")
            ctx[f"availability_{side}"] = (
                avail["availability_index"] if avail else 1.0
            )
            senti = ev.get(f"sentiment_{side}")
            ctx[f"sentiment_{side}"] = senti["score"] if senti else 0.0

        odds = ev.get("odds")
        if odds:
            implied = {s["outcome"]: s["implied_prob_vigfree"]
                       for s in odds["selections"]}
            ctx.update({f"odds_imp_{o}": implied[o] for o in ("home", "draw", "away")})
            ctx["market_quotes"] = [
                {"market": "h2h", "selection": s["outcome"], "model_prob": 0.0,
                 "market_prob": s["implied_prob_vigfree"],
                 "decimal_odds": s["decimal_odds"]}
                for s in odds["selections"]
            ]
        else:
            # stats-only anchor: a Dixon-Coles grid over form xG stands in
            from src.models.score_grid import outcome_probs, score_grid

            mu_h = (ctx["form_xg_for_home"] + ctx["form_xg_against_away"]) / 2
            mu_a = (ctx["form_xg_for_away"] + ctx["form_xg_against_home"]) / 2
            anchor = outcome_probs(score_grid(mu_h, mu_a, rho=-0.05))
            ctx.update({f"odds_imp_{o}": p for o, p in anchor.items()})
            degraded.append(
                "live odds unavailable; anchored on stats-only Dixon-Coles "
                "prior — no market comparison possible."
            )

        fixture = ev.get("fixture")
        ctx["neutral_venue"] = float(bool(fixture and fixture["neutral_venue"]))
        ctx["knockout"] = bool(fixture and fixture["knockout"])
        return ctx

    def infer(state: AgentState) -> dict:
        ctx = _match_context(state)
        call = _call(state, "ml-inference", "predict_match",
                     match_id=state.request.match_id, match_context=ctx)
        return {
            "prediction": call.result if call.ok else None,
            "ledger": state.ledger, "degraded": state.degraded,
        }

    def approve(state: AgentState) -> dict:
        pred = state.prediction
        suggestions = (pred or {}).get("suggestions") or []
        if not (state.request.wants_stakes and suggestions):
            return {"stake_approval": "not_required"}
        decision = interrupt({
            "type": "stake_approval",
            "match_id": state.request.match_id,
            "suggestions": suggestions,
            "instructions": "resume with {'action': 'approve'|'reject'|'edit',"
                            " 'suggestions': [...] (edit only)}",
        })
        action = (decision or {}).get("action", "reject")
        if action == "approve":
            return {"stake_approval": "approved",
                    "approved_suggestions": suggestions}
        if action == "edit":
            return {"stake_approval": "edited",
                    "approved_suggestions": decision.get("suggestions", [])}
        return {"stake_approval": "rejected"}

    def synthesize(state: AgentState) -> dict:
        return {"answer": render_answer(state)}

    graph = StateGraph(AgentState)
    for name, fn in [("parse", parse), ("gather", gather), ("news", news),
                     ("infer", infer), ("approve", approve),
                     ("synthesize", synthesize)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "gather")
    graph.add_edge("gather", "news")
    graph.add_edge("news", "infer")
    graph.add_edge("infer", "approve")
    graph.add_edge("approve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())

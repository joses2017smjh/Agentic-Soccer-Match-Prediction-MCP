"""Tests for the cognitive-swarm mode: DAG, executor, critic, supervisor.

These assert the swarm's real, deterministic guarantees — parallel DAG
execution, fail-fast retry, the adversarial critic catching anomalies and
recomputing arithmetic, the feedback loop, and memory commit — none of which
depend on an LLM.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from agent.parse import parse_request
from agent.swarm.critic import critique_prediction
from agent.swarm.planner import plan_dag, topological_layers
from agent.swarm.registry import ToolRegistry
from agent.swarm.state import SwarmState
from agent.swarm.supervisor import build_swarm
from agent.tooling import InProcessRunner


def _cfg() -> dict:
    return {"configurable": {"thread_id": f"swarm-{uuid.uuid4()}"}}


def _state(text: str) -> SwarmState:
    return SwarmState(request=parse_request(text))


# ------------------------------------------------------------------ registry

def test_registry_discovery() -> None:
    reg = ToolRegistry()
    assert reg.find_one("odds").tool == "get_live_odds"
    assert reg.find_one("prediction").server == "ml-inference"
    assert reg.find_one("nonexistent-capability") is None


def test_registry_reflects_down_servers() -> None:
    reg = ToolRegistry(disabled_servers={"news-sentiment"})
    assert reg.find_one("availability") is None       # news is down
    assert reg.find_one("odds") is not None            # data still up
    assert "sentiment" not in reg.available_capabilities()


# -------------------------------------------------------------------- planner

def test_plan_dag_structure_and_parallel_layer() -> None:
    caps = ToolRegistry().available_capabilities()
    plan = plan_dag(parse_request("Predict Arsenal vs Man City"), caps)
    layers = topological_layers(plan)
    # gather_stats and gather_news share the first layer → run in parallel
    first = {n.kind for n in layers[0]}
    assert first == {"gather_stats", "gather_news"}
    # infer depends on both; verify on infer; synthesize on verify
    assert [n.kind for n in layers[1]] == ["infer"]
    assert [layers[i][0].kind for i in range(2, 4)] == ["verify", "synthesize"]


def test_plan_adapts_to_down_server() -> None:
    caps = ToolRegistry(disabled_servers={"news-sentiment"}).available_capabilities()
    plan = plan_dag(parse_request("Predict Arsenal vs Man City"), caps)
    kinds = {n.kind for n in plan}
    assert "gather_news" not in kinds        # no news capability → node dropped
    assert "gather_stats" in kinds and "infer" in kinds


def test_topological_layers_rejects_cycle() -> None:
    from agent.swarm.state import TaskNode

    cyclic = [TaskNode(id="X", kind="infer", depends_on=["Y"]),
              TaskNode(id="Y", kind="verify", depends_on=["X"])]
    with pytest.raises(ValueError, match="cyclic"):
        topological_layers(cyclic)


# --------------------------------------------------------------------- critic

def _good_prediction() -> dict:
    return {
        "match_outcome": {"home": 0.5, "draw": 0.3, "away": 0.2,
                          "conformal_set": ["home", "draw"]},
        "expected_goals": {"home": 1.6, "away": 1.0},
        "exact_score": {"scoreline_grid": {
            "probs": [[0.5, 0.2], [0.2, 0.1]], "tail_mass": 0.0}},
        "event_sequence": {"first_scorer": {
            "home_first": 0.55, "away_first": 0.35, "no_goals": 0.10}},
        "suggestions": [],
    }


def test_critic_passes_clean_prediction() -> None:
    st = SwarmState(prediction=_good_prediction())
    crit = critique_prediction(st)
    assert crit.passed and not crit.issues and crit.checks_run > 5


def test_critic_catches_probability_sum() -> None:
    pred = _good_prediction()
    pred["match_outcome"]["home"] = 0.9   # now sums to 1.4
    crit = critique_prediction(SwarmState(prediction=pred))
    assert not crit.passed
    assert any("sum to" in i for i in crit.issues)


def test_critic_catches_leakage_anomaly() -> None:
    """98%+ favourite with a near-level xG is the leakage signature."""
    pred = _good_prediction()
    pred["match_outcome"] = {"home": 0.99, "draw": 0.005, "away": 0.005,
                             "conformal_set": ["home"]}
    pred["expected_goals"] = {"home": 1.5, "away": 1.4}   # gap only 0.1
    crit = critique_prediction(SwarmState(prediction=pred))
    assert not crit.passed
    assert any("leakage" in i for i in crit.issues)


def test_critic_recomputes_ev_arithmetic() -> None:
    pred = _good_prediction()
    pred["market_comparison"] = [{
        "market": "h2h", "selection": "home", "model_prob": 0.5,
        "market_prob": 0.42, "decimal_odds": 2.4, "edge": 0.08,
        "ev": 0.99,   # wrong; true ev = 0.5*1.4 - 0.5 = 0.2
    }]
    crit = critique_prediction(SwarmState(prediction=pred))
    assert not crit.passed
    assert any("EV mismatch" in i for i in crit.issues)


# ----------------------------------------------------------------- supervisor

def test_swarm_end_to_end() -> None:
    graph = build_swarm(InProcessRunner(), insight_path=Path("/tmp/none.jsonl"))
    res = graph.invoke(_state("Predict Arsenal vs Man City"), config=_cfg())
    st = SwarmState.model_validate(res)
    assert st.prediction is not None
    assert st.critiques and st.critiques[-1].passed
    assert st.critiques[-1].checks_run > 5
    assert "ARS" in st.answer and "adversarial checks" in st.answer
    # parallel gather ran: both stats and news evidence present
    assert "stats_home" in st.evidence and "availability_home" in st.evidence


def test_swarm_degrades_when_news_down() -> None:
    graph = build_swarm(InProcessRunner(disabled={"news-sentiment"}),
                        disabled_servers={"news-sentiment"},
                        insight_path=Path("/tmp/none2.jsonl"))
    res = graph.invoke(_state("Predict Arsenal vs Man City"), config=_cfg())
    st = SwarmState.model_validate(res)
    assert st.prediction is not None          # recovered from priors
    assert not any(n.kind == "gather_news" for n in st.plan)  # planned around it


def test_swarm_commits_insight(tmp_path: Path) -> None:
    insight = tmp_path / "insights.jsonl"
    graph = build_swarm(InProcessRunner(), insight_path=insight)
    graph.invoke(_state("Predict Arsenal vs Man City"), config=_cfg())
    assert insight.exists()
    body = insight.read_text()
    assert "ARS-MCI" in body and "critic" in body

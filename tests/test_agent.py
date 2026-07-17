"""Tests: request parsing, fast-path graph, HITL interrupt, degradation,
memory + reflection."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from langgraph.types import Command

from agent.graph import build_graph
from agent.memory import PredictionMemory
from agent.parse import parse_request
from agent.state import AgentState, ParsedRequest
from agent.tooling import InProcessRunner


def _cfg() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _state(text: str) -> AgentState:
    return AgentState(request=ParsedRequest(raw_text=text))


# --------------------------------------------------------------------- parse

def test_parse_team_names_and_stakes() -> None:
    req = parse_request("Predict Arsenal vs Man City, any value bets?")
    assert req.match_id == "ARS-MCI-2026-07-18"
    assert req.home_team == "ARS" and req.away_team == "MCI"
    assert req.wants_stakes


def test_parse_explicit_match_id_and_date() -> None:
    assert parse_request("what about ARS-MCI-2026-07-18?").match_id == \
        "ARS-MCI-2026-07-18"
    req = parse_request("Liverpool vs Barcelona on 2026-08-02")
    assert req.match_id == "LIV-BAR-2026-08-02"
    assert not req.wants_stakes


def test_parse_rejects_gibberish() -> None:
    with pytest.raises(ValueError, match="two teams"):
        parse_request("hello there")


# ------------------------------------------------------------------ workflow

def test_workflow_end_to_end() -> None:
    graph = build_graph(InProcessRunner())
    result = graph.invoke(_state("Predict Arsenal vs Man City"), config=_cfg())
    state = AgentState.model_validate(result)

    assert state.prediction is not None
    mo = state.prediction["match_outcome"]
    assert mo["home"] + mo["draw"] + mo["away"] == pytest.approx(1.0, abs=1e-6)
    assert state.stake_approval == "not_required"
    assert len(state.ledger) >= 9  # fixture+odds+2 stats+4 news+predict
    assert all(c.ok for c in state.ledger)
    assert "ARS" in state.answer and "%" in state.answer
    # demo feed has Jesus out → availability must have reached the model
    assert state.evidence["availability_home"]["availability_index"] < 1.0


def test_hitl_interrupt_and_approval() -> None:
    graph = build_graph(InProcessRunner(), ev_threshold=-1.0)
    cfg = _cfg()
    result = graph.invoke(
        _state("Arsenal vs Man City — any value bets?"), config=cfg
    )
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "stake_approval"
    assert payload["suggestions"]
    # no answer emitted yet — the interrupt fires BEFORE synthesis
    assert not result.get("answer")

    resumed = graph.invoke(Command(resume={"action": "approve"}), config=cfg)
    state = AgentState.model_validate(resumed)
    assert state.stake_approval == "approved"
    assert "Approved value suggestions" in state.answer
    assert "edge" in state.answer


def test_hitl_rejection_suppresses_stakes() -> None:
    graph = build_graph(InProcessRunner(), ev_threshold=-1.0)
    cfg = _cfg()
    graph.invoke(_state("Arsenal vs Man City — worth a bet?"), config=cfg)
    resumed = graph.invoke(Command(resume={"action": "reject"}), config=cfg)
    state = AgentState.model_validate(resumed)
    assert state.stake_approval == "rejected"
    assert "rejected by the human" in state.answer
    assert "kelly" not in state.answer.lower()


# ---------------------------------------------------------------- degradation

def test_news_server_down_degrades_gracefully() -> None:
    graph = build_graph(InProcessRunner(disabled={"news-sentiment"}))
    result = graph.invoke(_state("Arsenal vs Man City"), config=_cfg())
    state = AgentState.model_validate(result)
    assert state.prediction is not None
    assert any("news-sentiment" in n for n in state.degraded)
    assert "Reduced confidence" in state.answer


def test_data_server_down_uses_stats_only_anchor() -> None:
    graph = build_graph(InProcessRunner(disabled={"sports-data"}))
    result = graph.invoke(_state("Arsenal vs Man City"), config=_cfg())
    state = AgentState.model_validate(result)
    assert state.prediction is not None
    assert any("league-average priors" in n for n in state.degraded)
    assert any("odds unavailable" in n for n in state.degraded)


def test_ml_server_down_yields_honest_failure() -> None:
    graph = build_graph(InProcessRunner(disabled={"ml-inference"}))
    result = graph.invoke(_state("Arsenal vs Man City"), config=_cfg())
    state = AgentState.model_validate(result)
    assert state.prediction is None
    assert "could not produce a model prediction" in state.answer


# -------------------------------------------------------------------- memory

def test_memory_reflection_and_calibration(tmp_path: Path) -> None:
    mem = PredictionMemory(tmp_path / "pred.jsonl")
    mem.record_prediction(
        thread_id="t1", match_id="ARS-MCI-2026-07-18",
        probs={"home": 0.55, "draw": 0.25, "away": 0.20},
        degraded=[], model_version="v0-demo",
    )
    lesson = mem.reflect_on_outcome("ARS-MCI-2026-07-18", actual="away")
    assert lesson["correct"] is False
    assert lesson["prob_assigned_to_actual"] == 0.20
    assert "surprise" in lesson["note"]

    cal = mem.rolling_calibration()
    assert cal["settled"] == 1 and cal["accuracy"] == 0.0
    assert cal["mean_brier"] == pytest.approx(
        0.55**2 + 0.25**2 + 0.80**2
    )
    assert mem.recent_lessons()[-1]["match_id"] == "ARS-MCI-2026-07-18"


def test_memory_reflect_unknown_match_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no stored prediction"):
        PredictionMemory(tmp_path / "p.jsonl").reflect_on_outcome("X-Y-2026-01-01", "home")

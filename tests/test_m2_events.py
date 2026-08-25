"""Tests for M2: structured event bus, transport, and belief evolution trace.

A1: EventBus emits typed AgentEvents, stores them in a ring buffer,
    and streams to subscribers in real time.
A2: Transport upgrade -- NDJSON streaming and /runs replay.
A3: Belief evolution trace -- probs_before/probs_after/delta on each
    belief_update event.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from agent.events import AgentEvent, EventBus, _digest


# --------------------------------------------------------------------------
# A1: EventBus core
# --------------------------------------------------------------------------

def test_event_bus_emits_typed_events() -> None:
    bus = EventBus("test-thread-1")
    event = bus.emit("test_type", key="val")
    assert isinstance(event, AgentEvent)
    assert event.type == "test_type"
    assert event.thread_id == "test-thread-1"
    assert event.data["key"] == "val"
    assert event.at_utc  # non-empty timestamp


def test_event_bus_ring_buffer_stores_events() -> None:
    bus = EventBus("t1", buffer_size=5)
    for i in range(8):
        bus.emit("tick", i=i)
    events = bus.events
    assert len(events) == 5
    assert events[0].data["i"] == 3
    assert events[-1].data["i"] == 7


def test_event_bus_subscribe_receives_live_events() -> None:
    bus = EventBus("t1")
    received: list[AgentEvent] = []
    bus.subscribe(received.append)
    bus.emit("alpha")
    bus.emit("beta")
    assert len(received) == 2
    assert received[0].type == "alpha"
    assert received[1].type == "beta"


def test_event_bus_unsubscribe_stops_delivery() -> None:
    bus = EventBus("t1")
    received: list[AgentEvent] = []
    cb = received.append
    bus.subscribe(cb)
    bus.emit("a")
    bus.unsubscribe(cb)
    bus.emit("b")
    assert len(received) == 1


def test_event_bus_thread_safety() -> None:
    bus = EventBus("t1", buffer_size=200)
    barrier = threading.Barrier(4)

    def writer(prefix: str):
        barrier.wait()
        for i in range(50):
            bus.emit(f"{prefix}_{i}")

    threads = [threading.Thread(target=writer, args=(f"w{j}",)) for j in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(bus.events) == 200


def test_event_to_dict_and_json() -> None:
    bus = EventBus("t1")
    event = bus.emit("x", val=42)
    d = event.to_dict()
    assert d["type"] == "x"
    assert d["data"]["val"] == 42
    j = event.to_json()
    parsed = json.loads(j)
    assert parsed["type"] == "x"


def test_digest_deterministic() -> None:
    obj = {"a": 1, "b": [2, 3]}
    d1 = _digest(obj)
    d2 = _digest(obj)
    assert d1 == d2
    assert len(d1) == 8


# --------------------------------------------------------------------------
# A1: high-level emitters
# --------------------------------------------------------------------------

def test_run_start_end_events() -> None:
    bus = EventBus("t1")
    bus.run_start("workflow", "ARS-CHE")
    time.sleep(0.01)
    bus.run_end("success")
    events = bus.events
    assert events[0].type == "run_start"
    assert events[0].data["mode"] == "workflow"
    assert events[0].data["match_id"] == "ARS-CHE"
    assert events[1].type == "run_end"
    assert events[1].data["outcome"] == "success"
    assert events[1].data["elapsed_ms"] > 0


def test_node_enter_exit_events() -> None:
    bus = EventBus("t1")
    bus.node_enter("gather")
    bus.node_exit("gather", 42.5)
    events = bus.events
    assert events[0].type == "node_enter"
    assert events[0].data["node"] == "gather"
    assert events[1].type == "node_exit"
    assert events[1].data["duration_ms"] == 42.5


def test_tool_call_start_end_events() -> None:
    bus = EventBus("t1")
    bus.tool_call_start("sports-data", "get_team_stats", {"team_id": "ARS"})
    bus.tool_call_end("sports-data", "get_team_stats", ok=True,
                      latency_ms=15.3, result_digest="abc12345")
    events = bus.events
    assert events[0].type == "tool_call_start"
    assert events[0].data["server"] == "sports-data"
    assert events[1].type == "tool_call_end"
    assert events[1].data["ok"] is True
    assert events[1].data["latency_ms"] == 15.3


def test_evidence_added_event() -> None:
    bus = EventBus("t1")
    bus.evidence_added("stats", keys=["stats_home", "stats_away"])
    event = bus.events[0]
    assert event.type == "evidence_added"
    assert event.data["kind"] == "stats"
    assert "stats_home" in event.data["keys"]


def test_plan_built_event() -> None:
    bus = EventBus("t1")
    dag = [{"id": "A", "kind": "gather_stats"}, {"id": "C", "kind": "infer"}]
    bus.plan_built(dag)
    event = bus.events[0]
    assert event.type == "plan_built"
    assert len(event.data["dag"]) == 2


def test_critic_check_event() -> None:
    bus = EventBus("t1")
    bus.critic_check("prob_sum", 1.0, 1.0, True)
    event = bus.events[0]
    assert event.type == "critic_check"
    assert event.data["passed"] is True


def test_degradation_event() -> None:
    bus = EventBus("t1")
    bus.degradation("news server down")
    event = bus.events[0]
    assert event.type == "degradation"
    assert "news server" in event.data["note"]


def test_hitl_interrupt_event() -> None:
    bus = EventBus("t1")
    bus.hitl_interrupt({"type": "stake_approval", "match_id": "X"})
    event = bus.events[0]
    assert event.type == "hitl_interrupt"
    assert event.data["request"]["type"] == "stake_approval"


def test_cost_event() -> None:
    bus = EventBus("t1")
    bus.cost("claude-3-5-sonnet", 1500, 0.0045)
    event = bus.events[0]
    assert event.type == "cost"
    assert event.data["model"] == "claude-3-5-sonnet"
    assert event.data["usd"] == 0.0045


# --------------------------------------------------------------------------
# A3: belief evolution trace
# --------------------------------------------------------------------------

def test_belief_update_with_no_prior() -> None:
    bus = EventBus("t1")
    bus.belief_update("market_anchor", probs_before=None,
                      probs_after={"home": 0.4, "draw": 0.3, "away": 0.3})
    event = bus.events[0]
    assert event.type == "belief_update"
    assert event.data["stage"] == "market_anchor"
    assert event.data["probs_before"] is None
    assert event.data["probs_after"]["home"] == 0.4
    assert event.data["delta"] == {}


def test_belief_update_with_prior_computes_delta() -> None:
    bus = EventBus("t1")
    before = {"home": 0.40, "draw": 0.30, "away": 0.30}
    after = {"home": 0.45, "draw": 0.25, "away": 0.30}
    bus.belief_update("model_prediction", probs_before=before, probs_after=after)
    event = bus.events[0]
    assert event.data["delta"]["home"] == pytest.approx(0.05, abs=1e-4)
    assert event.data["delta"]["draw"] == pytest.approx(-0.05, abs=1e-4)
    assert event.data["delta"]["away"] == pytest.approx(0.0, abs=1e-4)


def test_belief_trace_full_sequence() -> None:
    """Simulate a full belief evolution: market anchor -> model prediction."""
    bus = EventBus("t1")
    market = {"home": 0.50, "draw": 0.25, "away": 0.25}
    bus.belief_update("market_anchor", probs_before=None, probs_after=market)

    model_out = {"home": 0.55, "draw": 0.20, "away": 0.25}
    bus.belief_update("model_prediction", probs_before=market, probs_after=model_out)

    events = bus.events
    assert len(events) == 2
    assert events[0].data["stage"] == "market_anchor"
    assert events[1].data["stage"] == "model_prediction"
    assert events[1].data["delta"]["home"] == pytest.approx(0.05, abs=1e-4)
    assert events[1].data["delta"]["draw"] == pytest.approx(-0.05, abs=1e-4)


# --------------------------------------------------------------------------
# A1/A2: tooling + graph wiring (event bus flows through runner)
# --------------------------------------------------------------------------

def test_tooling_emits_events_through_runner() -> None:
    from agent.tooling import InProcessRunner
    bus = EventBus("t-tooling")
    runner = InProcessRunner(event_bus=bus)
    call = runner.call("sports-data", "get_team_stats", team_id="Arsenal")
    events = bus.events
    types = [e.type for e in events]
    assert "tool_call_start" in types
    assert "tool_call_end" in types
    end_event = [e for e in events if e.type == "tool_call_end"][0]
    assert end_event.data["server"] == "sports-data"
    assert end_event.data["ok"] == call.ok


def test_tooling_emits_on_simulated_outage() -> None:
    from agent.tooling import InProcessRunner
    bus = EventBus("t-outage")
    runner = InProcessRunner(disabled={"news-sentiment"}, event_bus=bus)
    call = runner.call("news-sentiment", "get_availability_report", team="ARS")
    assert not call.ok
    types = [e.type for e in bus.events]
    assert "tool_call_start" in types
    assert "tool_call_end" in types
    end_event = [e for e in bus.events if e.type == "tool_call_end"][0]
    assert end_event.data["ok"] is False


def test_graph_wraps_nodes_with_events() -> None:
    """When event_bus is passed to build_graph, node_enter/exit events fire."""
    from agent.graph import build_graph
    from agent.tooling import InProcessRunner
    bus = EventBus("t-graph")
    runner = InProcessRunner(event_bus=bus)
    graph = build_graph(runner, event_bus=bus)
    result = graph.invoke(
        {"request": {"raw_text": "Arsenal vs Chelsea"}},
        config={"configurable": {"thread_id": "test-graph-events"}},
    )
    types = [e.type for e in bus.events]
    assert "node_enter" in types
    assert "node_exit" in types
    node_names = [e.data["node"] for e in bus.events if e.type == "node_enter"]
    assert "parse" in node_names
    assert "infer" in node_names


def test_graph_emits_belief_updates() -> None:
    from agent.graph import build_graph
    from agent.tooling import InProcessRunner
    bus = EventBus("t-belief")
    runner = InProcessRunner(event_bus=bus)
    graph = build_graph(runner, event_bus=bus)
    graph.invoke(
        {"request": {"raw_text": "Arsenal vs Chelsea"}},
        config={"configurable": {"thread_id": "test-belief-trace"}},
    )
    belief_events = [e for e in bus.events if e.type == "belief_update"]
    assert len(belief_events) >= 1
    stages = [e.data["stage"] for e in belief_events]
    assert "market_anchor" in stages


# --------------------------------------------------------------------------
# A2: tracing records events
# --------------------------------------------------------------------------

def test_record_trace_stores_events(tmp_path, monkeypatch) -> None:
    from agent.tracing import record_trace
    trace_file = tmp_path / "runs.jsonl"
    monkeypatch.setenv("TRACE_PATH", str(trace_file))

    bus = EventBus("t-trace")
    bus.emit("test_event", val=1)
    events_dicts = [e.to_dict() for e in bus.events]

    trace = record_trace(
        thread_id="t-trace", mode="workflow",
        state={"request": {"match_id": "ARS-CHE"}, "ledger": []},
        elapsed_ms=100.0, outcome="success",
        events=events_dicts,
    )
    assert len(trace["events"]) == 1
    assert trace["events"][0]["type"] == "test_event"

    raw = trace_file.read_text().strip()
    stored = json.loads(raw)
    assert stored["events"][0]["data"]["val"] == 1


# --------------------------------------------------------------------------
# Swarm wiring: critic emits check events through bus
# --------------------------------------------------------------------------

def test_critic_emits_check_events_through_bus() -> None:
    from agent.swarm.critic import critique_prediction
    from agent.swarm.state import SwarmState
    from agent.state import ParsedRequest

    bus = EventBus("t-critic")
    state = SwarmState(
        request=ParsedRequest(
            raw_text="Arsenal vs Chelsea",
            home_team="Arsenal", away_team="Chelsea",
            match_id="ARS-CHE",
        ),
        prediction={
            "match_outcome": {
                "home": 0.45, "draw": 0.25, "away": 0.30,
                "conformal_set": ["home", "draw"],
            },
            "expected_goals": {"home": 1.5, "away": 1.2},
        },
    )
    crit = critique_prediction(state, bus=bus)
    assert crit.passed
    check_events = [e for e in bus.events if e.type == "critic_check"]
    assert len(check_events) >= 5
    names = [e.data["name"] for e in check_events]
    assert "prob_sum" in names
    assert "prob_bounds" in names
    assert "conformal_set" in names
    assert all(e.data["passed"] for e in check_events)


def test_critic_emits_failed_check_events() -> None:
    from agent.swarm.critic import critique_prediction
    from agent.swarm.state import SwarmState
    from agent.state import ParsedRequest

    bus = EventBus("t-critic-fail")
    state = SwarmState(
        request=ParsedRequest(
            raw_text="A vs B", home_team="A", away_team="B", match_id="A-B",
        ),
        prediction={
            "match_outcome": {
                "home": 0.50, "draw": 0.30, "away": 0.25,
                "conformal_set": ["home"],
            },
            "expected_goals": {"home": 1.5, "away": 1.2},
        },
    )
    crit = critique_prediction(state, bus=bus)
    assert not crit.passed
    failed_checks = [e for e in bus.events
                     if e.type == "critic_check" and not e.data["passed"]]
    assert len(failed_checks) >= 1
    failed_names = [e.data["name"] for e in failed_checks]
    assert "prob_sum" in failed_names


def test_critic_no_prediction_emits_check() -> None:
    from agent.swarm.critic import critique_prediction
    from agent.swarm.state import SwarmState
    from agent.state import ParsedRequest

    bus = EventBus("t-critic-none")
    state = SwarmState(
        request=ParsedRequest(
            raw_text="A vs B", home_team="A", away_team="B", match_id="A-B",
        ),
        prediction=None,
    )
    crit = critique_prediction(state, bus=bus)
    assert not crit.passed
    check_events = [e for e in bus.events if e.type == "critic_check"]
    assert len(check_events) == 1
    assert check_events[0].data["name"] == "prediction_exists"
    assert check_events[0].data["passed"] is False


# --------------------------------------------------------------------------
# Swarm supervisor accepts event_bus
# --------------------------------------------------------------------------

def test_swarm_supervisor_accepts_event_bus() -> None:
    from agent.swarm.supervisor import build_swarm
    from agent.tooling import InProcessRunner
    bus = EventBus("t-swarm")
    runner = InProcessRunner(event_bus=bus)
    graph = build_swarm(runner, event_bus=bus)
    result = graph.invoke(
        {"request": {"raw_text": "Arsenal vs Chelsea"}},
        config={"configurable": {"thread_id": "test-swarm-events"}},
    )
    types = [e.type for e in bus.events]
    assert "node_enter" in types
    assert "node_exit" in types
    assert "plan_built" in types
    assert "critic_check" in types
    node_names = [e.data["node"] for e in bus.events if e.type == "node_enter"]
    assert "plan" in node_names
    assert "verify" in node_names


def test_swarm_executor_emits_belief_updates() -> None:
    from agent.swarm.supervisor import build_swarm
    from agent.tooling import InProcessRunner
    bus = EventBus("t-swarm-belief")
    runner = InProcessRunner(event_bus=bus)
    graph = build_swarm(runner, event_bus=bus)
    graph.invoke(
        {"request": {"raw_text": "Arsenal vs Chelsea"}},
        config={"configurable": {"thread_id": "test-swarm-belief"}},
    )
    belief_events = [e for e in bus.events if e.type == "belief_update"]
    assert len(belief_events) >= 1
    stages = [e.data["stage"] for e in belief_events]
    assert "market_anchor" in stages

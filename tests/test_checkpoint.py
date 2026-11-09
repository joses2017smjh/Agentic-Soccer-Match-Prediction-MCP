"""Tests: where paused runs live (agent/checkpoint.py) and the approval flow
across a restart.

The defect these pin down: a run waiting for human approval used to be saved
in process memory only, so a restart (or a second replica) lost it. The proof
of the fix is deliberately a *second graph with a fresh saver* over the same
store -- resuming on the object that paused would prove nothing.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import uuid

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from agent.checkpoint import make_checkpointer, state_types
from agent.graph import build_graph
from agent.state import AgentState, ParsedRequest, ToolCall
from agent.tooling import InProcessRunner

STAKES = "Arsenal vs Man City, any value bets?"
KEY = {"X-API-Key": "restart-key"}

# The SQLite saver is an optional extra (pip install -e ".[durable]").
needs_sqlite = pytest.mark.skipif(
    importlib.util.find_spec("langgraph.checkpoint.sqlite") is None,
    reason="langgraph-checkpoint-sqlite not installed (optional 'durable' extra)",
)


def _pause(graph) -> dict:
    """Run until the approval interrupt; return the thread config."""
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke(AgentState(request=ParsedRequest(raw_text=STAKES)), config=cfg)
    assert "__interrupt__" in result, "the stakes request should pause for approval"
    return cfg


# ------------------------------------------------------------- selection

def test_default_is_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHECKPOINT_URL", raising=False)
    assert isinstance(make_checkpointer(), MemorySaver)
    assert isinstance(make_checkpointer("memory://"), MemorySaver)


@needs_sqlite
def test_url_argument_overrides_environment(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHECKPOINT_URL", f"sqlite:///{tmp_path / 'env.db'}")
    assert isinstance(make_checkpointer("memory://"), MemorySaver)
    assert not isinstance(make_checkpointer(), MemorySaver)


@pytest.mark.parametrize("url", ["redis://localhost:6379", "file:///tmp/x", "nonsense"])
def test_unknown_backend_is_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="CHECKPOINT_URL"):
        make_checkpointer(url)


def test_server_backend_is_named_as_not_implemented() -> None:
    """Multi-host replicas need a server-backed store. Saying so beats a
    silent fall back to memory."""
    with pytest.raises(NotImplementedError, match="replicas"):
        make_checkpointer("postgresql://user@db/app")


@needs_sqlite
def test_sqlite_creates_missing_parent_directories(tmp_path) -> None:
    target = tmp_path / "a" / "b" / "runs.db"
    make_checkpointer(f"sqlite:///{target}")
    assert target.parent.is_dir()


# ------------------------------------------------- the defect and the fix

def test_in_memory_pause_is_invisible_to_a_second_graph() -> None:
    """Documents the failure mode: two processes, two memories."""
    cfg = _pause(build_graph(InProcessRunner(), checkpointer=make_checkpointer("memory://")))
    restarted = build_graph(InProcessRunner(), checkpointer=make_checkpointer("memory://"))
    assert restarted.get_state(cfg).next == ()


@needs_sqlite
def test_paused_approval_resumes_on_a_fresh_graph_over_the_same_store(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'runs.db'}"
    cfg = _pause(build_graph(InProcessRunner(), checkpointer=make_checkpointer(url)))

    restarted = build_graph(InProcessRunner(), checkpointer=make_checkpointer(url))
    assert restarted.get_state(cfg).next == ("approve",)
    final = AgentState.model_validate(
        restarted.invoke(Command(resume={"action": "approve"}), config=cfg)
    )
    assert final.stake_approval == "approved"
    assert final.approved_suggestions
    assert "Approved value suggestions" in final.answer
    assert all(isinstance(c, ToolCall) for c in final.ledger), "ledger must be restored as typed records"


@needs_sqlite
def test_reject_after_restart_discloses_no_stake(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'runs.db'}"
    cfg = _pause(build_graph(InProcessRunner(), checkpointer=make_checkpointer(url)))
    restarted = build_graph(InProcessRunner(), checkpointer=make_checkpointer(url))
    final = AgentState.model_validate(
        restarted.invoke(Command(resume={"action": "reject"}), config=cfg)
    )
    assert final.stake_approval == "rejected"
    assert not final.approved_suggestions
    assert "fractional-Kelly stake" not in final.answer


@needs_sqlite
def test_threads_are_isolated_in_one_store(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'runs.db'}"
    graph = build_graph(InProcessRunner(), checkpointer=make_checkpointer(url))
    first, second = _pause(graph), _pause(graph)
    graph.invoke(Command(resume={"action": "reject"}), config=first)
    assert graph.get_state(first).next == ()
    assert graph.get_state(second).next == ("approve",)   # untouched


# ----------------------------------------------------------- serializer

def test_project_state_classes_are_all_registered() -> None:
    import agent.state as workflow_state
    import agent.swarm.state as swarm_state

    registered = set(state_types())
    for module in (workflow_state, swarm_state):
        for name, obj in vars(module).items():
            if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == module.__name__:
                assert (module.__name__, name) in registered, f"{module.__name__}.{name} not registered"


def test_registered_types_round_trip_without_the_deprecation_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = make_checkpointer("memory://").serde
    call = ToolCall(server="sports-data", tool="get_live_odds", args={"match_id": "m"}, ok=True)
    with caplog.at_level(logging.WARNING):
        restored = serde.loads_typed(serde.dumps_typed(call))
    assert isinstance(restored, ToolCall) and restored == call
    assert not [r for r in caplog.records if "unregistered type" in r.getMessage()]


def test_unregistered_types_are_not_revived() -> None:
    """The allowlist is strict: a class we did not register must not be
    reconstructed from stored bytes."""

    class Stranger(BaseModel):
        x: int = 1

    serde = make_checkpointer("memory://").serde
    restored = serde.loads_typed(serde.dumps_typed(Stranger()))
    assert not isinstance(restored, Stranger)


# ------------------------------------------------------- gateway restart

@pytest.fixture()
def gateway(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Reload the gateway under a given CHECKPOINT_URL; each reload is a
    'restart' (new graph, new saver)."""
    import gateway.app as app_module

    monkeypatch.setenv("GATEWAY_API_KEY", KEY["X-API-Key"])
    monkeypatch.setenv("EV_THRESHOLD", "-1.0")
    monkeypatch.setenv("PREDICT_RATE_LIMIT", "1000/minute")
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "predictions.jsonl"))
    monkeypatch.setenv("TRACE_PATH", str(tmp_path / "runs.jsonl"))

    def start(checkpoint_url: str | None) -> TestClient:
        if checkpoint_url is None:
            monkeypatch.delenv("CHECKPOINT_URL", raising=False)
        else:
            monkeypatch.setenv("CHECKPOINT_URL", checkpoint_url)
        return TestClient(importlib.reload(app_module).app)

    yield start
    monkeypatch.delenv("CHECKPOINT_URL", raising=False)
    importlib.reload(app_module)   # leave no file-backed saver behind for other tests


@needs_sqlite
def test_gateway_approval_survives_a_restart(gateway, tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'gateway.db'}"
    paused = gateway(url).post("/predict", json={"text": STAKES}, headers=KEY).json()
    assert paused["status"] == "pending_approval"

    restarted = gateway(url)
    r = restarted.post("/approve", json={"thread_id": paused["thread_id"], "action": "approve"},
                       headers=KEY)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "complete"
    assert "Approved value suggestions" in r.json()["answer"]


def test_gateway_says_so_when_the_paused_run_is_gone(gateway) -> None:
    """In-memory mode still loses the run on restart; the caller must get a
    clear 404, not a 500 or an empty success."""
    paused = gateway(None).post("/predict", json={"text": STAKES}, headers=KEY).json()
    assert paused["status"] == "pending_approval"

    restarted = gateway(None)
    r = restarted.post("/approve", json={"thread_id": paused["thread_id"], "action": "approve"},
                       headers=KEY)
    assert r.status_code == 404
    assert "waiting for approval" in r.json()["detail"]


def test_gateway_rejects_approval_for_an_unknown_thread(gateway) -> None:
    r = gateway(None).post("/approve", json={"thread_id": "never-existed", "action": "approve"},
                           headers=KEY)
    assert r.status_code == 404

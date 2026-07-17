"""FastAPI gateway — the thin public edge of the system.

It authenticates, validates, invokes the agent, and streams. It never calls
models directly: predictions only exist behind the ML inference MCP server,
reached through the orchestrator graph.

Endpoints:
    GET  /health            liveness + loaded model version
    POST /predict           run the workflow; may return pending_approval
    POST /approve           resume a HITL-interrupted thread
    POST /predict/stream    NDJSON stream of node updates then the result
    POST /reflect           settle a finished match, write the lesson
    GET  /calibration       rolling deployed-system calibration

Auth: set GATEWAY_API_KEY to require X-API-Key on every non-health route.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.graph import build_graph
from agent.memory import PredictionMemory
from agent.state import AgentState, ParsedRequest
from agent.tooling import InProcessRunner
from agent.tracing import record_trace

app = FastAPI(title="soccer-prediction-gateway", version="0.1.0")


def _make_runner():
    """AGENT_RUNNER=mcp → real MCP client (Compose); default is in-process."""
    if os.environ.get("AGENT_RUNNER", "inprocess") == "mcp":
        from agent.tooling import MCPRunner

        return MCPRunner()
    return InProcessRunner()


_graph = build_graph(
    _make_runner(),
    ev_threshold=float(os.environ.get("EV_THRESHOLD", "0.03")),
)
_memory = PredictionMemory(
    Path(os.environ.get("MEMORY_PATH", "data/memory/predictions.jsonl"))
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("GATEWAY_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class PredictIn(BaseModel):
    text: str = Field(..., examples=["Predict Arsenal vs Man City, any value bets?"])
    thread_id: str | None = None


class ApproveIn(BaseModel):
    thread_id: str
    action: str = Field(..., pattern="^(approve|reject|edit)$")
    suggestions: list[dict[str, Any]] | None = None


class ReflectIn(BaseModel):
    match_id: str
    actual: str = Field(..., pattern="^(home|draw|away)$")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _payload(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    if "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        return {"status": "pending_approval", "thread_id": thread_id,
                "approval_request": intr.value}
    state = AgentState.model_validate(result)
    if state.prediction:
        _memory.record_prediction(
            thread_id=thread_id, match_id=state.request.match_id,
            probs={k: state.prediction["match_outcome"][k]
                   for k in ("home", "draw", "away")},
            degraded=state.degraded,
            model_version=state.prediction["model_version"],
        )
    return {
        "status": "complete", "thread_id": thread_id,
        "answer": state.answer, "prediction": state.prediction,
        "degraded": state.degraded,
        "tool_calls": [c.model_dump(exclude={"result"}) for c in state.ledger],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        from mcp_servers.ml_server.server import get_bundle

        version = get_bundle().version
    except Exception:  # noqa: BLE001 — remote-runner gateways hold no artifacts
        version = "remote (ml-inference server)"
    return {"ok": True, "model_version": version}


def _traced(result: dict[str, Any], thread_id: str, elapsed_ms: float) -> dict[str, Any]:
    payload = _payload(result, thread_id)
    record_trace(
        thread_id=thread_id, mode="workflow",
        state=AgentState.model_validate(
            {k: v for k, v in result.items() if k != "__interrupt__"}
        ),
        elapsed_ms=elapsed_ms, outcome=payload["status"],
    )
    return payload


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(body: PredictIn) -> dict[str, Any]:
    thread_id = body.thread_id or str(uuid.uuid4())
    start = time.monotonic()
    try:
        result = _graph.invoke(
            AgentState(request=ParsedRequest(raw_text=body.text)),
            config=_config(thread_id),
        )
    except ValueError as exc:  # unparseable request
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _traced(result, thread_id, (time.monotonic() - start) * 1000)


@app.post("/approve", dependencies=[Depends(require_api_key)])
def approve(body: ApproveIn) -> dict[str, Any]:
    resume: dict[str, Any] = {"action": body.action}
    if body.suggestions is not None:
        resume["suggestions"] = body.suggestions
    start = time.monotonic()
    result = _graph.invoke(Command(resume=resume), config=_config(body.thread_id))
    return _traced(result, body.thread_id, (time.monotonic() - start) * 1000)


@app.post("/predict/stream", dependencies=[Depends(require_api_key)])
def predict_stream(body: PredictIn) -> StreamingResponse:
    thread_id = body.thread_id or str(uuid.uuid4())

    # sync generator: Starlette runs it in a threadpool, which keeps the
    # MCPRunner (anyio.run inside) usable here as well
    def gen() -> Iterator[str]:
        start = time.monotonic()
        state = AgentState(request=ParsedRequest(raw_text=body.text))
        for update in _graph.stream(
            state, config=_config(thread_id), stream_mode="updates"
        ):
            for node in update:
                yield json.dumps({"event": "node", "node": node,
                                  "thread_id": thread_id}) + "\n"
        final = _graph.get_state(_config(thread_id))
        result = (dict(final.values) if not final.next
                  else {**dict(final.values),
                        "__interrupt__": final.tasks[0].interrupts})
        yield json.dumps(
            {"event": "result",
             **_traced(result, thread_id, (time.monotonic() - start) * 1000)}
        ) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/reflect", dependencies=[Depends(require_api_key)])
def reflect(body: ReflectIn) -> dict[str, Any]:
    try:
        return _memory.reflect_on_outcome(body.match_id, body.actual)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/calibration", dependencies=[Depends(require_api_key)])
def calibration() -> dict[str, Any]:
    return _memory.rolling_calibration()

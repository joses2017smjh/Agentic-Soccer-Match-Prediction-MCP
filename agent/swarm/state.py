"""Typed state for the cognitive-swarm supervisor.

The swarm decomposes a request into a DAG of tasks, runs independent tasks in
parallel through Executor agents bound to MCP tools, submits the result to an
adversarial Critic, and loops back to the Planner on failure (bounded). All
of it is checkpointed like the other modes, so a run is inspectable and
resumable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.state import ParsedRequest, ToolCall


class TaskNode(BaseModel):
    """One DAG node: a unit of work an Executor binds to MCP tools."""

    id: str
    kind: Literal["gather_stats", "gather_news", "infer", "verify", "synthesize"]
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "failed"] = "pending"
    attempts: int = 0
    result: dict[str, Any] | None = None
    error: str = ""


class Critique(BaseModel):
    """Adversarial verdict on the executor output."""

    passed: bool
    issues: list[str] = Field(default_factory=list)
    checks_run: int = 0
    iteration: int = 0


class MemoryContext(BaseModel):
    """What the Memory Agent recalled before planning (context alignment)."""

    recalled_insights: list[str] = Field(default_factory=list)
    rolling_calibration: dict[str, Any] = Field(default_factory=dict)


class SwarmState(BaseModel):
    request: ParsedRequest = ParsedRequest()

    memory_context: MemoryContext = MemoryContext()
    plan: list[TaskNode] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)   # keyed by kind
    prediction: dict[str, Any] | None = None

    critiques: list[Critique] = Field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 2

    ledger: list[ToolCall] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    answer: str = ""
    committed_insight: str = ""

    started_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def note_call(self, call: ToolCall) -> None:
        self.ledger.append(call)
        if not call.ok:
            self.degraded.append(
                f"{call.server}.{call.tool} unavailable ({call.error})"
            )

    def node(self, node_id: str) -> TaskNode:
        return next(n for n in self.plan if n.id == node_id)

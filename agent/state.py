"""Typed orchestrator state (Pydantic) shared by both execution modes.

Everything the agent knows lives here: the parsed request, an append-only
tool-call ledger (the evidence trail every synthesis claim must trace to),
the draft prediction, degradation notes, and the final answer. LangGraph
checkpoints this state per thread, which is what makes the HITL interrupt
resumable and follow-up questions cheap.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One ledger entry; ok=False entries drive the degradation disclosure."""

    server: str
    tool: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None = None
    error: str = ""
    latency_ms: float = 0.0
    at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class ParsedRequest(BaseModel):
    match_id: str = ""
    home_team: str = ""
    away_team: str = ""
    wants_stakes: bool = False
    follow_up_of: str = ""      # thread id of an earlier prediction, if any
    raw_text: str = ""


class AgentState(BaseModel):
    request: ParsedRequest = ParsedRequest()
    mode: Literal["workflow", "react"] = "workflow"

    ledger: list[ToolCall] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)   # keyed by kind
    degraded: list[str] = Field(default_factory=list)        # human-readable notes

    prediction: dict[str, Any] | None = None
    stake_approval: Literal["not_required", "pending", "approved", "rejected", "edited"] = (
        "not_required"
    )
    approved_suggestions: list[dict[str, Any]] = Field(default_factory=list)

    answer: str = ""
    cost_log: list[dict[str, Any]] = Field(default_factory=list)  # model, tokens, usd

    def note_call(self, call: ToolCall) -> None:
        self.ledger.append(call)
        if not call.ok:
            self.degraded.append(
                f"{call.server}.{call.tool} unavailable ({call.error}); "
                "proceeding without it."
            )

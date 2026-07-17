"""Run tracing: every agent run leaves a full tool-call tree with latencies.

Local-first: traces append to a JSONL file (TRACE_PATH, default
data/traces/runs.jsonl) so `jq` can answer "what did run X call and how long
did each hop take" with zero infra. For hosted tracing, set the standard
LangSmith env vars (LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, ...) —
LangGraph picks them up automatically and this module keeps writing the
local file regardless, so CI artifacts never depend on an external service.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _trace_path() -> Path:
    return Path(os.environ.get("TRACE_PATH", "data/traces/runs.jsonl"))


def record_trace(
    *, thread_id: str, mode: str, state: dict[str, Any] | Any,
    elapsed_ms: float, outcome: str,
) -> dict[str, Any]:
    """Append one run record. ``state`` is an AgentState or its dict dump."""
    dump = state if isinstance(state, dict) else state.model_dump()
    ledger = dump.get("ledger", [])
    trace = {
        "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thread_id": thread_id,
        "mode": mode,
        "outcome": outcome,                     # complete | pending_approval | error
        "elapsed_ms": round(elapsed_ms, 1),
        "match_id": dump.get("request", {}).get("match_id", ""),
        "degraded": dump.get("degraded", []),
        "cost_log": dump.get("cost_log", []),
        "tool_calls": [
            {
                "server": c["server"], "tool": c["tool"], "ok": c["ok"],
                "latency_ms": round(c["latency_ms"], 1),
                "error": c.get("error", ""),
            }
            for c in ledger
        ],
        "n_calls": len(ledger),
        "n_failed": sum(1 for c in ledger if not c["ok"]),
    }
    path = _trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(trace) + "\n")
    return trace

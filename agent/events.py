"""Structured event bus for agent introspection (A1).

Typed events emitted during a run, threaded through state. Replaces the
bare node-name stream from the gateway.  Names follow OTel GenAI semantic
conventions (gen_ai.operation.name) so a future OTel exporter reads the
same emission -- but the event bus itself is plain Python, no dependency.

Wiring: ToolRunner, graph nodes, swarm planner/executor, and critic all
emit through the bus.  The gateway streams events as NDJSON; the ring
buffer lets late-joining subscribers catch recent history.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass(frozen=True)
class AgentEvent:
    type: str
    thread_id: str
    at_utc: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _digest(obj: Any, max_len: int = 8) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:max_len]


class EventBus:
    """Per-run event emitter with a thread-safe ring buffer.

    Subscribe for live streaming; read ``.events`` for replay after the
    run completes.  The ring buffer keeps the last ``buffer_size`` events
    so late-joining /internals subscribers see recent history.
    """

    def __init__(self, thread_id: str, buffer_size: int = 1000) -> None:
        self._thread_id = thread_id
        self._buffer: deque[AgentEvent] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[AgentEvent], None]] = []
        self._start_mono: float | None = None

    @property
    def thread_id(self) -> str:
        return self._thread_id

    @property
    def events(self) -> list[AgentEvent]:
        with self._lock:
            return list(self._buffer)

    def subscribe(self, callback: Callable[[AgentEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[AgentEvent], None]) -> None:
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not callback]

    def emit(self, event_type: str, **data: Any) -> AgentEvent:
        event = AgentEvent(
            type=event_type,
            thread_id=self._thread_id,
            at_utc=_now_iso(),
            data=data,
        )
        with self._lock:
            self._buffer.append(event)
            subs = list(self._subscribers)
        for sub in subs:
            sub(event)
        return event

    # ---------------------------------------------------------------- high-level emitters

    def run_start(self, mode: str, match_id: str) -> None:
        self._start_mono = time.monotonic()
        self.emit("run_start", mode=mode, match_id=match_id)

    def run_end(self, outcome: str) -> None:
        elapsed = (
            (time.monotonic() - self._start_mono) * 1000
            if self._start_mono else 0.0
        )
        self.emit("run_end", outcome=outcome, elapsed_ms=round(elapsed, 1))

    def node_enter(self, node: str) -> None:
        self.emit("node_enter", node=node)

    def node_exit(self, node: str, duration_ms: float) -> None:
        self.emit("node_exit", node=node, duration_ms=round(duration_ms, 1))

    def tool_call_start(self, server: str, tool: str, args: dict) -> None:
        self.emit("tool_call_start", server=server, tool=tool, args=args)

    def tool_call_end(
        self, server: str, tool: str, ok: bool, latency_ms: float,
        result_digest: str = "", error: str = "",
    ) -> None:
        self.emit(
            "tool_call_end", server=server, tool=tool, ok=ok,
            latency_ms=round(latency_ms, 1),
            result_digest=result_digest, error=error,
        )

    def evidence_added(self, kind: str, keys: list[str] | None = None) -> None:
        self.emit("evidence_added", kind=kind, keys=keys or [])

    def belief_update(
        self, stage: str,
        probs_before: dict[str, float] | None,
        probs_after: dict[str, float],
    ) -> None:
        delta = {}
        if probs_before:
            delta = {
                k: round(probs_after.get(k, 0) - probs_before.get(k, 0), 4)
                for k in probs_after
            }
        self.emit(
            "belief_update", stage=stage,
            probs_before=probs_before, probs_after=probs_after, delta=delta,
        )

    def plan_built(self, dag: list[dict]) -> None:
        self.emit("plan_built", dag=dag)

    def critic_check(
        self, name: str, lhs: Any, rhs: Any, passed: bool,
    ) -> None:
        self.emit("critic_check", name=name, lhs=lhs, rhs=rhs, passed=passed)

    def degradation(self, note: str) -> None:
        self.emit("degradation", note=note)

    def hitl_interrupt(self, request: dict) -> None:
        self.emit("hitl_interrupt", request=request)

    def cost(self, model: str, tokens: int, usd: float) -> None:
        self.emit("cost", model=model, tokens=tokens, usd=round(usd, 6))

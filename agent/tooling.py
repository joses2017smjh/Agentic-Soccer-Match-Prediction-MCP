"""Tool access for the orchestrator: one seam, two implementations.

``ToolRunner`` is how graph nodes call MCP tools. ``MCPRunner`` speaks real
MCP via langchain-mcp-adapters (STDIO subprocesses locally, Streamable HTTP
URLs in Compose — set MCP_DATA_URL etc.). ``InProcessRunner`` calls the same
server logic functions directly in-process: zero subprocesses, fully
deterministic — used by unit tests and the offline demo. Both honor per-call
timeouts and never raise into the graph: failures come back as ok=False
ledger entries so the degradation path is data, not exceptions.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Protocol

from agent.events import EventBus, _digest
from agent.state import ToolCall

DEFAULT_TIMEOUT_S = 20.0


class ToolRunner(Protocol):
    event_bus: EventBus | None

    def call(self, server: str, tool: str, **args: Any) -> ToolCall: ...


def _plain_json(value: Any) -> Any:
    """Round-trip through JSON so only plain Python types enter agent state —
    numpy scalars etc. would break the checkpointer's serializer."""

    def default(o: Any) -> Any:
        if hasattr(o, "item"):  # numpy scalar
            return o.item()
        if hasattr(o, "tolist"):  # numpy array
            return o.tolist()
        return str(o)

    return json.loads(json.dumps(value, default=default))


def _execute(
    server: str, tool: str, fn: Callable[[], Any], args: dict,
    bus: EventBus | None = None,
) -> ToolCall:
    if bus:
        bus.tool_call_start(server, tool, args)
    start = time.monotonic()
    try:
        result = _plain_json(fn())
        elapsed = (time.monotonic() - start) * 1000
        if bus:
            bus.tool_call_end(
                server, tool, ok=True, latency_ms=elapsed,
                result_digest=_digest(result),
            )
        return ToolCall(server=server, tool=tool, args=args, ok=True,
                        result=result, latency_ms=elapsed)
    except Exception as exc:  # noqa: BLE001 — failures become ledger data
        elapsed = (time.monotonic() - start) * 1000
        err = f"{type(exc).__name__}: {exc}"
        if bus:
            bus.tool_call_end(
                server, tool, ok=False, latency_ms=elapsed, error=err,
            )
        return ToolCall(server=server, tool=tool, args=args, ok=False,
                        error=err, latency_ms=elapsed)


class InProcessRunner:
    """Direct calls into mcp_servers logic. ``disabled`` simulates outages
    for fault-injection evals (e.g. {'news-sentiment'})."""

    def __init__(
        self, disabled: set[str] | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.disabled = disabled or set()
        self._local = threading.local()
        self._build_registry()
        if event_bus is not None:
            self._local.bus = event_bus

    def _build_registry(self) -> None:
        from mcp_servers.code_server import server as code
        from mcp_servers.data_server import server as data
        from mcp_servers.ml_server import server as ml
        from mcp_servers.news_server import server as news

        self._registry: dict[tuple[str, str], Callable[..., dict]] = {
            ("sports-data", "get_team_stats"): data._team_stats,
            ("sports-data", "get_live_odds"): data._live_odds,
            ("sports-data", "get_h2h"): data._h2h,
            ("sports-data", "get_fixture_context"): data._fixture_context,
            ("sports-data", "get_squad_props"): data._squad_props,
            ("news-sentiment", "get_availability_report"): news._availability,
            ("news-sentiment", "analyze_team_sentiment"): news._sentiment,
            ("ml-inference", "predict_match"): ml.run_predict,
            ("ml-inference", "explain_prediction"): ml.run_explain,
            ("ml-inference", "get_model_card"): lambda: dict(ml.get_bundle().card),
            ("code-env", "run_python"): code.do_run,
            ("code-env", "list_datasets"): lambda: code.list_datasets.fn(),
        }

    @property
    def event_bus(self) -> EventBus | None:
        return getattr(self._local, "bus", None)

    @event_bus.setter
    def event_bus(self, bus: EventBus | None) -> None:
        self._local.bus = bus

    def call(self, server: str, tool: str, **args: Any) -> ToolCall:
        if server in self.disabled:
            if self.event_bus:
                self.event_bus.tool_call_start(server, tool, args)
                self.event_bus.tool_call_end(
                    server, tool, ok=False, latency_ms=0,
                    error="ServerDown: simulated outage",
                )
            return ToolCall(server=server, tool=tool, args=args, ok=False,
                            error="ServerDown: simulated outage")
        fn = self._registry.get((server, tool))
        if fn is None:
            return ToolCall(server=server, tool=tool, args=args, ok=False,
                            error=f"UnknownTool: {server}.{tool}")
        return _execute(server, tool, lambda: fn(**args), args, bus=self.event_bus)


class MCPRunner:
    """Real MCP client. Discovers tools at startup from all three servers."""

    def __init__(
        self, timeout_s: float = DEFAULT_TIMEOUT_S,
        event_bus: EventBus | None = None,
    ) -> None:
        import anyio
        from langchain_mcp_adapters.client import MultiServerMCPClient

        self._anyio = anyio
        self.timeout_s = timeout_s
        self.event_bus = event_bus
        self._client = MultiServerMCPClient(self._connections())
        self._tools = {
            t.name: t for t in anyio.run(self._client.get_tools)
        }

    @staticmethod
    def _connections() -> dict[str, dict[str, Any]]:
        """Streamable HTTP when *_URL is set (Compose); else STDIO spawn."""
        conns: dict[str, dict[str, Any]] = {}
        for name, env, module in [
            ("sports-data", "MCP_DATA_URL", "mcp_servers.data_server.server"),
            ("news-sentiment", "MCP_NEWS_URL", "mcp_servers.news_server.server"),
            ("ml-inference", "MCP_ML_URL", "mcp_servers.ml_server.server"),
        ]:
            url = os.environ.get(env)
            conns[name] = (
                {"transport": "streamable_http", "url": url}
                if url
                else {"transport": "stdio", "command": "python",
                      "args": ["-m", module]}
            )
        return conns

    def call(self, server: str, tool: str, **args: Any) -> ToolCall:
        lc_tool = self._tools.get(tool)
        if lc_tool is None:
            return ToolCall(server=server, tool=tool, args=args, ok=False,
                            error=f"UnknownTool: {tool}")

        def _run() -> Any:
            async def _inner() -> Any:
                with self._anyio.fail_after(self.timeout_s):
                    return await lc_tool.ainvoke(args)
            return self._anyio.run(_inner)

        return _execute(server, tool, _run, args, bus=self.event_bus)

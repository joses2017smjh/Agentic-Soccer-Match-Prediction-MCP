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
import time
from typing import Any, Callable, Protocol

from agent.state import ToolCall

DEFAULT_TIMEOUT_S = 20.0


class ToolRunner(Protocol):
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


def _execute(server: str, tool: str, fn: Callable[[], Any], args: dict) -> ToolCall:
    start = time.monotonic()
    try:
        result = _plain_json(fn())
        return ToolCall(server=server, tool=tool, args=args, ok=True,
                        result=result,
                        latency_ms=(time.monotonic() - start) * 1000)
    except Exception as exc:  # noqa: BLE001 — failures become ledger data
        return ToolCall(server=server, tool=tool, args=args, ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                        latency_ms=(time.monotonic() - start) * 1000)


class InProcessRunner:
    """Direct calls into mcp_servers logic. ``disabled`` simulates outages
    for fault-injection evals (e.g. {'news-sentiment'})."""

    def __init__(self, disabled: set[str] | None = None) -> None:
        self.disabled = disabled or set()
        from mcp_servers.data_server import server as data
        from mcp_servers.ml_server import server as ml
        from mcp_servers.news_server import server as news

        self._registry: dict[tuple[str, str], Callable[..., dict]] = {
            ("sports-data", "get_team_stats"): data._team_stats,
            ("sports-data", "get_live_odds"): data._live_odds,
            ("sports-data", "get_h2h"): data._h2h,
            ("sports-data", "get_fixture_context"): data._fixture_context,
            ("news-sentiment", "get_availability_report"): news._availability,
            ("news-sentiment", "analyze_team_sentiment"): news._sentiment,
            ("ml-inference", "predict_match"): ml.run_predict,
            ("ml-inference", "explain_prediction"): ml.run_explain,
            ("ml-inference", "get_model_card"): lambda: dict(ml.get_bundle().card),
        }

    def call(self, server: str, tool: str, **args: Any) -> ToolCall:
        if server in self.disabled:
            return ToolCall(server=server, tool=tool, args=args, ok=False,
                            error="ServerDown: simulated outage")
        fn = self._registry.get((server, tool))
        if fn is None:
            return ToolCall(server=server, tool=tool, args=args, ok=False,
                            error=f"UnknownTool: {server}.{tool}")
        return _execute(server, tool, lambda: fn(**args), args)


class MCPRunner:
    """Real MCP client. Discovers tools at startup from all three servers."""

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        import anyio
        from langchain_mcp_adapters.client import MultiServerMCPClient

        self._anyio = anyio
        self.timeout_s = timeout_s
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

        return _execute(server, tool, _run, args)

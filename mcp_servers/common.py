"""Shared MCP server plumbing: as_of stamping, TTL cache, transport runner.

Server requirements applied uniformly (see README):
- every tool result carries an ``as_of`` UTC timestamp — the serving-time
  leakage guard: the orchestrator can always tell how stale evidence is;
- tools are idempotent and cache through a small TTL cache to respect
  upstream provider rate limits;
- STDIO transport for local development, Streamable HTTP for containers
  (the deprecated HTTP+SSE transport is intentionally not offered).
"""

from __future__ import annotations

import functools
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def with_as_of(payload: dict[str, Any]) -> dict[str, Any]:
    """Stamp a tool result. Existing as_of (e.g. an odds snapshot time) wins —
    the true information time is always the more conservative claim."""
    payload.setdefault("as_of", utc_now_iso())
    return payload


def ttl_cache(seconds: float = 60.0, maxsize: int = 512) -> Callable:
    """Tiny TTL cache for idempotent tool backends (hashable args only)."""

    def decorator(fn: Callable) -> Callable:
        store: dict[tuple, tuple[float, Any]] = {}

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            hit = store.get(key)
            now = time.monotonic()
            if hit is not None and now - hit[0] < seconds:
                return hit[1]
            value = fn(*args, **kwargs)
            if len(store) >= maxsize:
                store.pop(next(iter(store)))
            store[key] = (now, value)
            return value

        wrapper.cache_clear = store.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


def run_server(server: FastMCP) -> None:
    """STDIO by default; MCP_TRANSPORT=streamable-http inside containers."""
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport not in ("stdio", "streamable-http"):
        raise ValueError(f"unsupported transport: {transport}")
    server.run(transport=transport)

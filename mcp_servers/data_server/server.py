"""MCP Server 1 — Sports Data & Odds.

Four read-only, idempotent, cached tools over a swappable DataBackend.
Tool descriptions are written for the orchestrator's tool-selection step:
they say what the tool answers, what the arguments look like, and what the
result guarantees (as_of stamps, vig-free probabilities).

Run:  python -m mcp_servers.data_server.server           (STDIO)
      MCP_TRANSPORT=streamable-http python -m ...        (HTTP)
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_servers.common import run_server, ttl_cache, with_as_of
from mcp_servers.data_server.backend import DataBackend, DemoBackend

server = FastMCP("sports-data")
_backend: DataBackend = DemoBackend()


@ttl_cache(seconds=300)
def _team_stats(team_id: str, window: int) -> dict[str, Any]:
    return with_as_of(_backend.team_stats(team_id, window))


@ttl_cache(seconds=20)  # odds move; keep this cache short
def _live_odds(match_id: str, market: str) -> dict[str, Any]:
    return with_as_of(_backend.live_odds(match_id, market))


@ttl_cache(seconds=3600)
def _h2h(team_a: str, team_b: str) -> dict[str, Any]:
    return with_as_of(_backend.h2h(team_a, team_b))


@ttl_cache(seconds=600)
def _fixture_context(match_id: str) -> dict[str, Any]:
    return with_as_of(_backend.fixture_context(match_id))


@server.tool()
def get_team_stats(team_id: str, window: int = 10) -> dict[str, Any]:
    """Rolling recency-weighted form for one team over its last `window`
    matches: xG for/against, shots, possession, rest days. Use this to judge
    current team strength. `team_id` is the short code (e.g. 'ARS', 'MCI').
    Result includes `as_of` — the time these stats were computed."""
    return _team_stats(team_id, window)


@server.tool()
def get_live_odds(match_id: str, market: str = "h2h") -> dict[str, Any]:
    """Latest bookmaker odds for one match and market ('h2h' or 'totals_2.5').
    Each selection carries the payable decimal odds AND the vig-free implied
    probability — use the implied probability for fair-value comparisons and
    the decimal odds for payout math. `match_id` looks like
    'ARS-MCI-2026-07-18'. `as_of` is the odds snapshot time; treat older
    snapshots as stale, never as current prices."""
    return _live_odds(match_id, market)


@server.tool()
def get_h2h(team_a: str, team_b: str) -> dict[str, Any]:
    """Head-to-head record between two teams: meetings, win/draw split and
    average total goals. Historical context only — do not use as a substitute
    for current form from get_team_stats."""
    return _h2h(team_a, team_b)


@server.tool()
def get_fixture_context(match_id: str) -> dict[str, Any]:
    """Tournament context for one fixture: stage, knockout flag, neutral
    venue, rest days per side, and stakes ('must_win' | 'normal' |
    'dead_rubber'). Needed to decide whether knockout advancement
    probabilities apply and whether home advantage is reduced."""
    return _fixture_context(match_id)


if __name__ == "__main__":
    run_server(server)

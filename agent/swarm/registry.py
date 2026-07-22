"""MCP Registry & Discovery.

A capability directory over the connected MCP servers: agents query it for a
tool that satisfies a need ("something that returns vig-free odds") rather
than hard-coding server/tool names. This is the swarm's substitute for
static wiring — swap the tool that fulfils a capability and executors follow
automatically.

The registry is populated from the same tool set the ToolRunner exposes, so
it never drifts from what is actually callable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    server: str
    tool: str
    capabilities: frozenset[str]
    description: str


# capability tags → the tool that provides them. Extend as servers are added.
_REGISTRY: list[ToolSpec] = [
    ToolSpec("sports-data", "get_fixture_context",
             frozenset({"fixture", "stage", "venue", "stakes"}),
             "tournament context for a match"),
    ToolSpec("sports-data", "get_live_odds",
             frozenset({"odds", "market", "implied_probability"}),
             "bookmaker odds + vig-free implied probabilities"),
    ToolSpec("sports-data", "get_team_stats",
             frozenset({"form", "team_strength", "xg"}),
             "rolling recency-weighted team form"),
    ToolSpec("sports-data", "get_squad_props",
             frozenset({"squad", "player_shares", "set_piece"}),
             "per-player xG/xA shares"),
    ToolSpec("sports-data", "get_h2h",
             frozenset({"h2h", "history"}), "head-to-head record"),
    ToolSpec("news-sentiment", "get_availability_report",
             frozenset({"availability", "injuries", "lineup"}),
             "structured injury/lineup availability"),
    ToolSpec("news-sentiment", "analyze_team_sentiment",
             frozenset({"sentiment", "morale"}),
             "recency-decayed team morale score"),
    ToolSpec("ml-inference", "predict_match",
             frozenset({"prediction", "math", "compute", "inference"}),
             "full deterministic prediction stack (the compute sandbox)"),
    ToolSpec("ml-inference", "explain_prediction",
             frozenset({"explanation", "attribution", "shap"}),
             "SHAP feature attributions"),
    ToolSpec("ml-inference", "get_model_card",
             frozenset({"model_card", "features", "metrics"}),
             "model version, features, eval metrics"),
]


class ToolRegistry:
    def __init__(self, disabled_servers: set[str] | None = None) -> None:
        self.disabled = disabled_servers or set()

    def discover(self, capability: str) -> list[ToolSpec]:
        """Tools that satisfy a capability, from servers that are up."""
        return [
            t for t in _REGISTRY
            if capability in t.capabilities and t.server not in self.disabled
        ]

    def find_one(self, capability: str) -> ToolSpec | None:
        matches = self.discover(capability)
        return matches[0] if matches else None

    def available_capabilities(self) -> set[str]:
        caps: set[str] = set()
        for t in _REGISTRY:
            if t.server not in self.disabled:
                caps |= t.capabilities
        return caps

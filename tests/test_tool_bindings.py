"""Tests: every tool the planner can pick is actually callable in-process.

The in-process runner binds (server, tool) pairs to functions by hand, so a
binding can drift from what a server module really exposes. A drifted binding
does not crash: the runner turns the AttributeError into an ok=False ledger
entry, which reads like an outage. These tests tell the two apart.
"""

from __future__ import annotations

import pytest

from agent.swarm.registry import _REGISTRY
from agent.tooling import InProcessRunner

# Failures that mean "the binding itself is wrong", as opposed to a tool
# rejecting the arguments it was given.
BINDING_ERRORS = ("AttributeError", "UnknownTool", "ImportError", "NameError")


def test_code_env_list_datasets_returns_datasets() -> None:
    call = InProcessRunner().call("code-env", "list_datasets")
    assert call.ok, call.error
    assert isinstance(call.result["datasets"], list)
    assert all({"name", "columns"} <= set(d) for d in call.result["datasets"])
    assert "as_of" in call.result


def test_every_swarm_tool_has_an_in_process_binding() -> None:
    bound = set(InProcessRunner()._registry)  # noqa: SLF001
    missing = [(t.server, t.tool) for t in _REGISTRY if (t.server, t.tool) not in bound]
    assert not missing, f"planner can pick tools the runner cannot call: {missing}"


@pytest.mark.parametrize("server,tool", sorted(InProcessRunner()._registry))  # noqa: SLF001
def test_binding_resolves_to_a_real_callable(server: str, tool: str) -> None:
    """Call each binding with no arguments. A tool may refuse (it needs
    arguments), but the refusal must come from the tool, not from a binding
    that points at something that does not exist."""
    call = InProcessRunner().call(server, tool)
    assert not call.error.startswith(BINDING_ERRORS), f"{server}.{tool}: {call.error}"

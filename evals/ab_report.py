"""A/B report: deterministic workflow vs agentic ReAct mode.

Runs the golden set through mode A (fixed workflow — keyless, deterministic)
and, when ANTHROPIC_API_KEY is present, mode B (ReAct over the same MCP
tools). Reports task success, latency, and dollar cost per request side by
side — the empirical version of the *Building Effective Agents* guidance
that agency must pay for its nondeterminism, latency, and cost.

Without a key, mode B is reported as skipped rather than simulated: this
project does not fake numbers.

Run: .venv/bin/python -m evals.ab_report [--out evals/out/ab_report.md]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evals.golden_set import TASKS
from evals.runner import run_all


def _workflow_arm() -> dict:
    report = run_all(TASKS)
    return {
        "mode": "workflow (fixed graph)",
        "n_tasks": report["n_tasks"],
        "task_success_rate": report["task_success_rate"],
        "mean_latency_ms": report["mean_latency_ms"],
        "usd_per_request": 0.0,   # no LLM calls on the deterministic path
        "notes": "deterministic; zero token cost; keyless",
    }


def _swarm_arm() -> dict:
    """Cognitive swarm on the non-HITL golden tasks: planner DAG + parallel
    executors + adversarial critic. Deterministic and keyless, so it runs
    here and produces real numbers. Success = a valid prediction (or an
    honest failure on a fully-degraded run) that the Critic did not leave
    with unresolved issues."""
    import time
    import uuid

    from agent.parse import parse_request
    from agent.swarm.state import SwarmState
    from agent.swarm.supervisor import build_swarm
    from agent.tooling import InProcessRunner

    tasks = [t for t in TASKS
             if t.category in ("happy", "fault") and not t.expect_parse_error]
    ok, latencies, checks = 0, [], []
    for t in tasks:
        runner = InProcessRunner(disabled=set(t.disabled_servers))
        graph = build_swarm(runner, disabled_servers=set(t.disabled_servers))
        start = time.monotonic()
        try:
            res = graph.invoke(
                SwarmState(request=parse_request(t.request)),
                config={"configurable": {"thread_id": f"swarm-{uuid.uuid4()}"}},
            )
        except ValueError:
            continue
        latencies.append((time.monotonic() - start) * 1000)
        crits = res.get("critiques", [])
        if crits:
            checks.append(crits[-1].checks_run)
        pred_ok = res.get("prediction") is not None or bool(res.get("degraded"))
        crit_ok = (not crits) or crits[-1].passed or bool(t.disabled_servers)
        ok += pred_ok and crit_ok
    return {
        "mode": "swarm (planner+critic)",
        "n_tasks": len(tasks),
        "task_success_rate": ok / len(tasks) if tasks else None,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "usd_per_request": 0.0,
        "notes": (f"deterministic core; keyless; mean "
                  f"{sum(checks) / len(checks):.0f} adversarial checks/run"
                  if checks else "deterministic core; keyless"),
    }


def _react_arm() -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {
            "mode": "react (agentic)", "n_tasks": 0,
            "task_success_rate": None, "mean_latency_ms": None,
            "usd_per_request": None,
            "notes": "SKIPPED: ANTHROPIC_API_KEY not set — run locally with "
                     "a key to fill this arm; numbers are never simulated.",
        }
    # Real arm: drive the ReAct agent over the same tasks, tally tokens.
    import anyio

    from agent.react_mode import build_react_agent, load_mcp_tools

    tools = anyio.run(load_mcp_tools)
    agent = build_react_agent(tools)
    import time
    n_ok, latencies, usd = 0, [], 0.0
    tasks = [t for t in TASKS if not t.expect_parse_error][:10]
    for task in tasks:
        start = time.monotonic()
        try:
            result = agent.invoke(
                {"messages": [("user", task.request)]}
            )
            answer = result["messages"][-1].content
            n_ok += bool(answer) and all(
                m not in answer for m in task.injection_markers
            )
            meta = result["messages"][-1].response_metadata.get("usage", {})
            usd += (meta.get("input_tokens", 0) * 1e-6
                    + meta.get("output_tokens", 0) * 5e-6)
        except Exception:  # noqa: BLE001
            pass
        latencies.append((time.monotonic() - start) * 1000)
    return {
        "mode": "react (agentic)", "n_tasks": len(tasks),
        "task_success_rate": n_ok / len(tasks),
        "mean_latency_ms": sum(latencies) / len(latencies),
        "usd_per_request": usd / len(tasks),
        "notes": "ReAct over live MCP tools; token cost measured",
    }


def render(arms: list[dict]) -> str:
    def fmt(v, pct=False):
        if v is None:
            return "—"
        return f"{v:.1%}" if pct else (f"{v:,.1f}" if isinstance(v, float) else str(v))

    lines = [
        "# Workflow vs Swarm vs Agent — A/B report", "",
        "| metric | " + " | ".join(a["mode"] for a in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for key, label, pct in [
        ("n_tasks", "tasks run", False),
        ("task_success_rate", "task success", True),
        ("mean_latency_ms", "mean latency (ms)", False),
        ("usd_per_request", "cost per request (USD)", False),
    ]:
        lines.append(f"| {label} | " +
                     " | ".join(fmt(a[key], pct) for a in arms) + " |")
    lines += ["", *(f"- **{a['mode']}**: {a['notes']}" for a in arms)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("evals/out/ab_report.md"))
    args = parser.parse_args()
    arms = [_workflow_arm(), _swarm_arm(), _react_arm()]
    md = render(arms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(md)
    print(json.dumps(arms, indent=2))


if __name__ == "__main__":
    main()

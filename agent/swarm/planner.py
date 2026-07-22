"""Strategic Planner (System-2 decomposition).

Turns a request into a DAG of tasks. For the match-prediction domain the
plan is stable and its structure is what matters: two independent gather
tasks (stats, news) that run in parallel, an inference task that depends on
both, and a verification task. This is deterministic and inspectable; an LLM
planner can replace ``plan_dag`` for open-ended requests without changing the
executor/critic contract, because it must emit the same TaskNode DAG.

``topological_layers`` groups nodes into dependency layers — each layer runs
concurrently in the Executor swarm.
"""

from __future__ import annotations

from agent.state import ParsedRequest
from agent.swarm.state import TaskNode


def plan_dag(request: ParsedRequest, available_caps: set[str]) -> list[TaskNode]:
    """Build the prediction DAG, skipping gather nodes whose capabilities are
    entirely unavailable (a down server) — the plan adapts to the mesh."""
    nodes: list[TaskNode] = []
    infer_deps: list[str] = []

    if {"form", "odds", "fixture"} & available_caps:
        nodes.append(TaskNode(id="A", kind="gather_stats"))
        infer_deps.append("A")
    if {"availability", "sentiment"} & available_caps:
        nodes.append(TaskNode(id="B", kind="gather_news"))
        infer_deps.append("B")

    nodes.append(TaskNode(id="C", kind="infer", depends_on=infer_deps))
    nodes.append(TaskNode(id="D", kind="verify", depends_on=["C"]))
    nodes.append(TaskNode(id="E", kind="synthesize", depends_on=["D"]))
    return nodes


def topological_layers(nodes: list[TaskNode]) -> list[list[TaskNode]]:
    """Order nodes into layers of concurrently-runnable tasks (Kahn's
    algorithm). Raises on a cycle — a DAG must stay acyclic."""
    by_id = {n.id: n for n in nodes}
    indeg = {n.id: len(n.depends_on) for n in nodes}
    layers: list[list[TaskNode]] = []
    remaining = set(by_id)

    while remaining:
        ready = sorted(nid for nid in remaining if indeg[nid] == 0)
        if not ready:
            raise ValueError("planner produced a cyclic DAG")
        layers.append([by_id[nid] for nid in ready])
        for nid in ready:
            remaining.discard(nid)
            for other in remaining:
                if nid in by_id[other].depends_on:
                    indeg[other] -= 1
    return layers

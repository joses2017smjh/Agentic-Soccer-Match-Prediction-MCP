"""Three-scene demo (mirrors the recording script in the README):

1. normal prediction with value suggestions + HITL approval
2. follow-up "why?" answered from explain_prediction
3. fault recovery: news server down mid-run, disclosed reduced confidence

Runs fully offline (in-process runner + demo artifacts).
Usage: .venv/bin/python -m scripts.demo
"""

from __future__ import annotations

import json
import uuid

from langgraph.types import Command

from agent.graph import build_graph
from agent.state import AgentState, ParsedRequest
from agent.tooling import InProcessRunner
from mcp_servers.ml_server import server as ml_srv


def scene(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> None:
    # ---------------------------------------------------------------- scene 1
    scene("SCENE 1 — 'Arsenal vs Man City — any value bets?' (HITL approval)")
    graph = build_graph(InProcessRunner(), ev_threshold=-1.0)
    cfg = {"configurable": {"thread_id": f"demo-{uuid.uuid4()}"}}
    result = graph.invoke(
        AgentState(request=ParsedRequest(
            raw_text="Arsenal vs Man City — any value bets?")),
        config=cfg,
    )
    payload = result["__interrupt__"][0].value
    print(f"\n[interrupt] {len(payload['suggestions'])} suggestion(s) held "
          "for human review; first one:")
    print(json.dumps(payload["suggestions"][0], indent=2))
    print("\n[human] approve")
    resumed = graph.invoke(Command(resume={"action": "approve"}), config=cfg)
    print("\n" + resumed["answer"])

    # ---------------------------------------------------------------- scene 2
    scene("SCENE 2 — follow-up: why? (explain_prediction)")
    explain = ml_srv.run_explain("ARS-MCI-2026-07-18", top_k=3)
    print(f"strongest drivers of the {explain['explained_class']} call:")
    for f in explain["top_features"]:
        print(f"  {f['feature']:>22} = {f['value']:.3f}   "
              f"contribution {f['contribution']:+.3f}")

    # ---------------------------------------------------------------- scene 3
    scene("SCENE 3 — fault recovery: news server down")
    graph_down = build_graph(InProcessRunner(disabled={"news-sentiment"}))
    result = graph_down.invoke(
        AgentState(request=ParsedRequest(raw_text="Predict Arsenal vs Man City")),
        config={"configurable": {"thread_id": f"demo-{uuid.uuid4()}"}},
    )
    print("\n" + result["answer"])


if __name__ == "__main__":
    main()

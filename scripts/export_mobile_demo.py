"""Freeze keyless mobile fixtures from the actual synthetic agent workflow.

Usage: DATA_BACKEND=demo python -m scripts.export_mobile_demo
The generated artifact is synthetic and gitignored; only the small responses
and source/artifact fingerprints are committed. No forecasting benchmark is
run and no quality claim is made.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = "2026-10-07T12:00:00+00:00"
VERSION = "v0-mobile-demo"


class FixtureDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        fixed = datetime.fromisoformat(FIXED_TIME)
        return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)


def main() -> None:
    if os.environ.get("DATA_BACKEND", "demo") != "demo":
        raise ValueError("Fixture export requires DATA_BACKEND=demo")
    if os.environ.get("AGENT_RUNNER", "inprocess") != "inprocess":
        raise ValueError("Fixture export requires the in-process synthetic workflow")
    os.environ["DATA_BACKEND"] = "demo"
    os.environ["MODEL_VERSION"] = VERSION
    os.environ["ARTIFACT_ROOT"] = str(ROOT / "data" / "artifacts")

    from src.models import artifact

    artifact.datetime = FixtureDateTime
    from scripts.build_demo_artifacts import build

    artifact_path = build(VERSION, seed=42)
    import agent.state as state_module
    from agent.graph import build_graph
    from agent.state import AgentState, ParsedRequest
    from agent.tooling import InProcessRunner
    from mcp_servers import common
    from mcp_servers.data_server import backend
    from mcp_servers.news_server import server as news

    # Freeze the synthetic news ages and provider timestamps. Actual wall-clock
    # latency is removed from snapshots rather than inventing timing numbers.
    common.datetime = FixtureDateTime
    backend.datetime = FixtureDateTime
    news.datetime = FixtureDateTime
    state_module.datetime = FixtureDateTime

    graph = build_graph(InProcessRunner(), ev_threshold=-1.0)
    cases = [
        (
            "arsenal-city",
            "Predict Arsenal vs Man City on 2026-07-18",
            ["Predict Arsenal vs Man City", "Arsenal vs Man City"],
        ),
        (
            "liverpool-city",
            "Predict Liverpool vs Man City on 2026-07-18",
            ["Predict Liverpool vs Man City", "Liverpool vs Man City"],
        ),
        (
            "approval-required",
            "Arsenal vs Man City on 2026-07-18 — any value bets?",
            ["Arsenal vs Man City — any value bets?", "Arsenal vs Man City - any value bets?"],
        ),
    ]
    source_files = [
        "scripts/build_demo_artifacts.py",
        "scripts/export_mobile_demo.py",
        "agent/graph.py",
        "agent/parse.py",
        "agent/state.py",
        "agent/synthesis.py",
        "agent/tooling.py",
        "mcp_servers/common.py",
        "mcp_servers/demo_data.py",
        "mcp_servers/data_server/backend.py",
        "mcp_servers/data_server/server.py",
        "mcp_servers/news_server/server.py",
        "mcp_servers/ml_server/server.py",
        "src/models/artifact.py",
        "src/models/calibration.py",
        "src/models/conformal.py",
        "src/models/gbm.py",
        "src/models/predict.py",
        "src/models/score_grid.py",
        "src/models/sequence.py",
        "src/models/player_props.py",
        "src/models/suggestions.py",
        "src/news/availability.py",
        "src/news/sentiment.py",
        "src/news/schemas.py",
    ]
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files
    }
    artifact_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(artifact_path.iterdir())
        if path.is_file()
    }
    samples = []
    for case_id, prompt, aliases in cases:
        thread_id = f"mobile-demo-{case_id}"
        result = graph.invoke(
            AgentState(request=ParsedRequest(raw_text=prompt)),
            config={"configurable": {"thread_id": thread_id}},
        )
        state = AgentState.model_validate(
            {key: value for key, value in result.items() if key != "__interrupt__"}
        )
        pending = bool(result.get("__interrupt__"))
        response = {
            "service": "soccer",
            "mode": "demo",
            "status": "pending_approval" if pending else "complete",
            "thread_id": thread_id,
            "answer": state.answer,
            "prediction": None if pending else state.prediction,
            "degraded": state.degraded,
            "tool_calls": []
            if pending
            else [call.model_dump(exclude={"result", "latency_ms"}) for call in state.ledger],
            "approval_request": result["__interrupt__"][0].value if pending else None,
            "provenance": {
                "kind": "synthetic-fixture",
                "model_version": VERSION,
                "data_backend": "demo",
                "fixture_id": case_id,
                "generated_at_utc": FIXED_TIME,
                "notes": [
                    "Synthetic model trained with seed 42 and synthetic provider/news/squads; not a current match forecast.",
                    "Captured from the existing LangGraph workflow and MCP tool logic at a fixed fixture clock.",
                    "Tool latencies are omitted from frozen examples; this is not a performance or model-quality benchmark.",
                    "The conformal set belongs to the synthetic calibration distribution; no real-match coverage claim is made.",
                    "Approval examples remain pending. The mobile adapter cannot approve or place wagers.",
                ],
            },
        }
        response_bytes = json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
        response["provenance"]["fixture_sha256"] = hashlib.sha256(response_bytes).hexdigest()
        samples.append(
            {
                "id": case_id,
                "input": {"text": prompt, "mode": "demo"},
                "aliases": aliases,
                "response": response,
            }
        )
    bundle = {
        "schema_version": "1.0",
        "service": "soccer",
        "generated_at_utc": FIXED_TIME,
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "generator": "python -m scripts.export_mobile_demo",
        "source_sha256": source_hashes,
        "artifact_sha256": artifact_hashes,
        "model_card": json.loads((artifact_path / "card.json").read_bytes()),
        "samples": samples,
    }
    path = ROOT / "gateway" / "fixtures" / "mobile_demo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} synthetic workflow samples to {path}")


if __name__ == "__main__":
    main()

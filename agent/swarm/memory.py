"""Knowledge / Memory agent.

Two roles from the spec, at the fidelity this project honestly supports:

- **Context alignment (recall)** before planning: pull the deployed system's
  rolling calibration and any stored insights for this fixture, so the run
  starts informed by past outcomes and prior critiques.
- **Memory commit (write)** after synthesis: persist a structured insight —
  including any Critic corrections — for future runs.

This is a lightweight, file-backed insight store layered on the existing
PredictionMemory. It is a deliberate simplification of the spec's "Semantic
Memory Graph (Knowledge Graph + Vector hybrid)"; the interface (recall /
commit keyed by concept) is the part that would carry over to a real KG+vector
backend, which is the documented next step.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.memory import PredictionMemory


class SwarmMemory:
    def __init__(self, insight_path: Path, prediction_memory: PredictionMemory) -> None:
        self.insight_path = insight_path
        self.insight_path.parent.mkdir(parents=True, exist_ok=True)
        self.predictions = prediction_memory

    def _rows(self) -> list[dict[str, Any]]:
        if not self.insight_path.exists():
            return []
        return [json.loads(l) for l in self.insight_path.read_text().splitlines() if l]

    def recall(self, concept: str, k: int = 3) -> list[str]:
        """Insights whose concept key matches (substring), most recent first."""
        hits = [r for r in self._rows() if concept.lower() in r["concept"].lower()]
        return [r["insight"] for r in hits[-k:]]

    def commit(self, concept: str, insight: str) -> None:
        record = {
            "concept": concept, "insight": insight,
            "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self.insight_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def rolling_calibration(self) -> dict[str, Any]:
        return self.predictions.rolling_calibration()

"""Long-term memory: prediction log, post-match reflection, rolling calibration.

MemGPT-style split: per-thread working state lives in the LangGraph
checkpointer; this module is the *external* store — an append-only JSONL of
predictions, settled outcomes, and Reflexion-style lessons that survives
process restarts and feeds the deployed system's calibration tracking.

``reflect_on_outcome`` is the post-mortem node: compare prediction vs actual
result, score it, and write a structured lesson. Lessons are typed records
produced by our code (not LLM free text), so they can be injected into
future runs' context safely and aggregated into the calibration report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUTCOMES = ("home", "draw", "away")


class PredictionMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict[str, Any]) -> None:
        record["at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    # ------------------------------------------------------------ recording

    def record_prediction(
        self, *, thread_id: str, match_id: str, probs: dict[str, float],
        degraded: list[str], model_version: str,
    ) -> None:
        self._append({
            "kind": "prediction", "thread_id": thread_id, "match_id": match_id,
            "probs": probs, "degraded": degraded, "model_version": model_version,
        })

    def reflect_on_outcome(self, match_id: str, actual: str) -> dict[str, Any]:
        """Settle the latest prediction for match_id and write the lesson."""
        preds = [r for r in self._rows()
                 if r["kind"] == "prediction" and r["match_id"] == match_id]
        if not preds:
            raise ValueError(f"no stored prediction for {match_id!r}")
        pred = preds[-1]
        probs = pred["probs"]
        brier = sum(
            (probs[o] - (1.0 if o == actual else 0.0)) ** 2 for o in OUTCOMES
        )
        predicted = max(OUTCOMES, key=lambda o: probs[o])

        lesson = {
            "kind": "lesson", "match_id": match_id, "actual": actual,
            "predicted": predicted, "correct": predicted == actual,
            "prob_assigned_to_actual": probs[actual], "brier": brier,
            "was_degraded": bool(pred["degraded"]),
            "note": (
                "surprise result: model gave the actual outcome "
                f"{probs[actual]:.0%}" if probs[actual] < 0.25 else
                "in line with expectations"
            ),
        }
        self._append(lesson)
        return lesson

    # ------------------------------------------------------------ reporting

    def rolling_calibration(self) -> dict[str, Any]:
        lessons = [r for r in self._rows() if r["kind"] == "lesson"]
        if not lessons:
            return {"settled": 0}
        n = len(lessons)
        return {
            "settled": n,
            "accuracy": sum(l["correct"] for l in lessons) / n,
            "mean_brier": sum(l["brier"] for l in lessons) / n,
            "mean_prob_on_actual": sum(
                l["prob_assigned_to_actual"] for l in lessons
            ) / n,
            "degraded_share": sum(l["was_degraded"] for l in lessons) / n,
        }

    def recent_lessons(self, k: int = 5) -> list[dict[str, Any]]:
        lessons = [r for r in self._rows() if r["kind"] == "lesson"]
        return lessons[-k:]

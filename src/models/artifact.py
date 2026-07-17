"""Versioned model artifact bundle.

Training (offline, Phase A) writes one directory per version containing every
fitted component plus a model card. The ML inference MCP server loads a
bundle at startup and exposes the card via ``get_model_card``; any feature-
schema mismatch at prediction time raises instead of failing silently.

Layout::

    artifacts/<version>/
        outcome.ubj, outcome_meta.json     OutcomeGBM
        xg_home.ubj, xg_away.ubj, xg_meta.json   TeamXGGBM
        outcome_head.pkl                   IsotonicCalibrator + ConformalWrapper
        timing.json                        GoalTimingModel
        card.json                          version, window, rho, metrics
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.calibration import CalibratedOutcomeHead
from src.models.gbm import OutcomeGBM, TeamXGGBM
from src.models.sequence import GoalTimingModel


@dataclass
class ArtifactBundle:
    version: str
    outcome: OutcomeGBM
    xg: TeamXGGBM
    head: CalibratedOutcomeHead
    timing: GoalTimingModel
    rho: float
    card: dict[str, Any]

    @classmethod
    def build_card(
        cls, *, version: str, training_window: str, rho: float,
        conformal_alpha: float, feature_names: list[str],
        metrics: dict[str, float] | None = None, notes: str = "",
    ) -> dict[str, Any]:
        return {
            "version": version,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "training_window": training_window,
            "dixon_coles_rho": rho,
            "conformal_alpha": conformal_alpha,
            "feature_names": feature_names,
            "eval_metrics": metrics or {},
            "notes": notes,
        }

    def save(self, root: Path) -> Path:
        path = root / self.version
        path.mkdir(parents=True, exist_ok=True)
        self.outcome.save(path)
        self.xg.save(path)
        self.head.save(path)
        self.timing.save(path)
        (path / "card.json").write_text(json.dumps(self.card, indent=2))
        return path

    @classmethod
    def load(cls, root: Path, version: str | None = None) -> "ArtifactBundle":
        """Load ``version``, or the lexicographically latest one present."""
        if version is None:
            versions = sorted(p.name for p in root.iterdir() if (p / "card.json").exists())
            if not versions:
                raise FileNotFoundError(f"no artifact bundles under {root}")
            version = versions[-1]
        path = root / version
        card = json.loads((path / "card.json").read_text())
        return cls(
            version=version,
            outcome=OutcomeGBM.load(path),
            xg=TeamXGGBM.load(path),
            head=CalibratedOutcomeHead.load(path),
            timing=GoalTimingModel.load(path),
            rho=float(card["dixon_coles_rho"]),
            card=card,
        )

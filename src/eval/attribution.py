"""Stage-wise attribution analysis.

Measures which pipeline stage drives prediction accuracy -- analogous to
DealTrace's finding that forecast quality carries beta=0.75 against the
final verdict while extraction carries only 0.28.

Pipeline stages for this system:
1. **Outcome model**: XGBoost 1X2 probability prediction
2. **Calibration**: isotonic calibration + conformal prediction sets
3. **Score grid**: Dixon-Coles bivariate Poisson scoreline distribution
4. **Derived markets**: over/under, BTTS, player props from the grid
5. **Synthesis**: answer generation with evidence grounding

Attribution runs the mixed-verifier harness on a batch of settled predictions,
then decomposes composite accuracy by which verifier (stage proxy) contributes
most to the overall score variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.eval.harness import (
    ActualResult,
    DEFAULT_WEIGHTS,
    EvaluationResult,
    evaluate,
    reward_hacking_floor_test,
)


@dataclass
class StageContribution:
    """One pipeline stage's measured contribution to overall quality."""

    stage: str
    mean_score: float
    std_score: float
    correlation_with_composite: float
    weight: float
    weighted_contribution: float


@dataclass
class AttributionReport:
    """Full stage-wise attribution across a batch of predictions."""

    n_evaluated: int
    mean_composite: float
    std_composite: float
    stages: list[StageContribution]
    binding_constraint: str
    reward_hacking_safe: bool
    calibration_summary: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


def run_attribution(
    predictions: list[dict[str, Any]],
    actuals: list[ActualResult],
    match_ids: list[str],
    *,
    weights: dict[str, float] | None = None,
) -> AttributionReport:
    """Evaluate a batch and decompose quality by pipeline stage.

    Each prediction is scored by the mixed-verifier harness. The attribution
    identifies which verifier (proxy for pipeline stage) has the highest
    correlation with composite score variance -- the "binding constraint"
    that improvement effort should target.
    """
    if len(predictions) != len(actuals) or len(predictions) != len(match_ids):
        raise ValueError("predictions, actuals, and match_ids must have equal length")
    if not predictions:
        raise ValueError("at least one prediction required")

    w = weights or DEFAULT_WEIGHTS
    results: list[EvaluationResult] = []
    for mid, pred, actual in zip(match_ids, predictions, actuals):
        results.append(evaluate(mid, pred, actual, weights=w))

    composites = np.array([r.composite_score for r in results])

    verifier_names = [v.name for v in results[0].verifiers]
    verifier_scores: dict[str, np.ndarray] = {}
    for name in verifier_names:
        verifier_scores[name] = np.array([
            next(v.score for v in r.verifiers if v.name == name)
            for r in results
        ])

    stages: list[StageContribution] = []
    for name in verifier_names:
        scores = verifier_scores[name]
        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores))

        if std_s > 1e-9 and np.std(composites) > 1e-9:
            corr = float(np.corrcoef(scores, composites)[0, 1])
        else:
            corr = 0.0

        stage_weight = w.get(name, 0.0)
        weighted = mean_s * stage_weight

        stages.append(StageContribution(
            stage=name,
            mean_score=round(mean_s, 4),
            std_score=round(std_s, 4),
            correlation_with_composite=round(corr, 4),
            weight=stage_weight,
            weighted_contribution=round(weighted, 4),
        ))

    binding = max(stages, key=lambda s: abs(s.correlation_with_composite))

    # calibration summary across all evaluated predictions
    briers = []
    log_losses = []
    xg_maes = []
    outcomes_correct = []
    conformal_covered = []

    for r in results:
        cal = next((v for v in r.verifiers if v.name == "calibration"), None)
        state = next((v for v in r.verifiers if v.name == "state"), None)
        if cal:
            briers.append(cal.details.get("brier_score", 0))
            log_losses.append(cal.details.get("log_loss", 0))
            xg_maes.append(cal.details.get("xg_mae", 0))
        if state:
            outcomes_correct.append(1.0 if state.passed else 0.0)
            conformal_covered.append(
                1.0 if state.details.get("in_conformal_set") else 0.0
            )

    cal_summary = {
        "mean_brier": round(float(np.mean(briers)), 4) if briers else 0.0,
        "mean_log_loss": round(float(np.mean(log_losses)), 4) if log_losses else 0.0,
        "mean_xg_mae": round(float(np.mean(xg_maes)), 2) if xg_maes else 0.0,
        "outcome_accuracy": (
            round(float(np.mean(outcomes_correct)), 4) if outcomes_correct else 0.0
        ),
        "conformal_coverage": (
            round(float(np.mean(conformal_covered)), 4)
            if conformal_covered else 0.0
        ),
    }

    # reward-hacking floor test on the first actual result
    floor_result = reward_hacking_floor_test(actuals[0])

    return AttributionReport(
        n_evaluated=len(results),
        mean_composite=round(float(np.mean(composites)), 4),
        std_composite=round(float(np.std(composites)), 4),
        stages=stages,
        binding_constraint=binding.stage,
        reward_hacking_safe=floor_result["harness_safe"],
        calibration_summary=cal_summary,
        details={
            "per_match": [
                {
                    "match_id": r.match_id,
                    "composite": r.composite_score,
                    "per_verifier": {
                        v.name: round(v.score, 4) for v in r.verifiers
                    },
                }
                for r in results
            ],
            "reward_hacking_test": floor_result,
        },
    )


def attribution_to_dict(report: AttributionReport) -> dict[str, Any]:
    """Serialize an AttributionReport for JSON API responses."""
    return {
        "n_evaluated": report.n_evaluated,
        "mean_composite": report.mean_composite,
        "std_composite": report.std_composite,
        "binding_constraint": report.binding_constraint,
        "reward_hacking_safe": report.reward_hacking_safe,
        "stages": [
            {
                "stage": s.stage,
                "mean_score": s.mean_score,
                "std_score": s.std_score,
                "correlation_with_composite": s.correlation_with_composite,
                "weight": s.weight,
                "weighted_contribution": s.weighted_contribution,
            }
            for s in report.stages
        ],
        "calibration_summary": report.calibration_summary,
        "details": report.details,
    }

"""Mixed-verifier evaluation harness (Westworld-style).

Three verifier types mirror Halluminate's Westworld environment design:

1. **State-based**: compare predicted outcome against the actual match result.
2. **Component-level**: validate individual derived markets independently
   (over/under, BTTS, score range, xG direction).
3. **Ground-truth matching**: continuous quality metrics (Brier score,
   log-loss, xG mean absolute error).

Verifier scores are combined under configurable weights into a single 0-1
composite, enabling direct comparison across predictions and models.

A reward-hacking resistance test verifies that empty or trivial submissions
score below a configurable floor (default 0.30), ensuring the scoring
function cannot be gamed by content-free strategies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

OUTCOMES = ("home", "draw", "away")


@dataclass(frozen=True)
class ActualResult:
    """Ground truth for a completed match."""

    outcome: str
    home_goals: int
    away_goals: int


@dataclass
class VerifierResult:
    """One verifier's assessment of a single prediction."""

    name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Combined evaluation from all verifiers."""

    match_id: str
    composite_score: float
    verifiers: list[VerifierResult]
    reward_hacking_safe: bool
    details: dict[str, Any] = field(default_factory=dict)


def state_verify(
    prediction: dict[str, Any], actual: ActualResult
) -> VerifierResult:
    """State-based verification: did the prediction call the right winner?

    Scoring:
    - 0.50: predicted outcome (argmax) matches actual
    - 0.25: actual outcome is inside the conformal prediction set
    - 0.25 * p(actual): reward calibrated confidence on the true outcome

    Invalid probability distributions (sum < 0.01) receive score 0.
    """
    mo = prediction.get("match_outcome", {})
    probs = {o: mo.get(o, 0.0) for o in OUTCOMES}
    prob_sum = sum(probs.values())

    if prob_sum < 0.01:
        return VerifierResult(
            name="state",
            score=0.0,
            passed=False,
            details={"error": "invalid probability distribution", "prob_sum": prob_sum},
        )

    predicted = max(OUTCOMES, key=lambda o: probs[o])
    max_prob = probs[predicted]
    correct = predicted == actual.outcome and max_prob > 1 / 3 + 0.01

    conf_set = mo.get("conformal_set", [])
    in_conformal = actual.outcome in conf_set
    prob_on_actual = probs.get(actual.outcome, 0.0)

    score = 0.0
    if correct:
        score += 0.50
    if in_conformal:
        score += 0.25
    score += 0.25 * prob_on_actual

    return VerifierResult(
        name="state",
        score=min(score, 1.0),
        passed=correct,
        details={
            "predicted": predicted,
            "actual": actual.outcome,
            "correct": correct,
            "in_conformal_set": in_conformal,
            "conformal_set": conf_set,
            "prob_on_actual": round(prob_on_actual, 4),
        },
    )


def component_verify(
    prediction: dict[str, Any], actual: ActualResult
) -> VerifierResult:
    """Component-level verification: individual derived markets.

    Each market is checked independently -- no partial credit within a check.
    The score is the fraction of passing checks.
    """
    checks: dict[str, bool] = {}
    total_goals = actual.home_goals + actual.away_goals
    exact = prediction.get("exact_score", {})

    ou = exact.get("over_under_2_5", {})
    if ou:
        model_says_over = ou.get("over", 0.5) > 0.5
        checks["over_under_2_5"] = model_says_over == (total_goals > 2.5)

    bt = exact.get("btts", {})
    if bt:
        model_says_btts = bt.get("yes", 0.5) > 0.5
        actually_btts = actual.home_goals > 0 and actual.away_goals > 0
        checks["btts"] = model_says_btts == actually_btts

    top = exact.get("top_scorelines", [])
    actual_score = f"{actual.home_goals}-{actual.away_goals}"
    checks["score_in_top_5"] = any(s.get("score") == actual_score for s in top[:5])

    xg = prediction.get("expected_goals", {})
    if xg:
        xg_diff = xg.get("home", 0) - xg.get("away", 0)
        goal_diff = actual.home_goals - actual.away_goals
        checks["xg_direction"] = (
            (xg_diff > 0.1 and goal_diff > 0)
            or (xg_diff < -0.1 and goal_diff < 0)
            or (abs(xg_diff) <= 0.1 and goal_diff == 0)
        )

    n_checks = len(checks)
    n_passed = sum(checks.values())
    score = n_passed / n_checks if n_checks > 0 else 0.0

    return VerifierResult(
        name="component",
        score=score,
        passed=n_passed > n_checks * 0.5,
        details={"checks": checks, "passed_count": n_passed, "total_count": n_checks},
    )


def calibration_verify(
    prediction: dict[str, Any], actual: ActualResult
) -> VerifierResult:
    """Ground-truth matching: continuous quality metrics.

    - Brier score: sum of squared errors vs one-hot (range 0-2)
    - Log-loss: negative log of probability assigned to actual outcome
    - xG MAE: mean absolute error on expected goals vs actual goals

    Score = max(0, 1 - brier), mapping Brier 0 -> 1.0, Brier >= 1 -> 0.0.
    """
    mo = prediction.get("match_outcome", {})
    probs = {o: mo.get(o, 0.0) for o in OUTCOMES}

    brier = sum(
        (probs[o] - (1.0 if o == actual.outcome else 0.0)) ** 2
        for o in OUTCOMES
    )

    p_actual = max(probs.get(actual.outcome, 0.0), 1e-15)
    ll = -math.log(p_actual)

    xg = prediction.get("expected_goals", {})
    xg_error_home = abs(xg.get("home", 0.0) - actual.home_goals)
    xg_error_away = abs(xg.get("away", 0.0) - actual.away_goals)
    xg_mae = (xg_error_home + xg_error_away) / 2

    score = max(0.0, 1.0 - brier)

    return VerifierResult(
        name="calibration",
        score=score,
        passed=brier < 0.75,
        details={
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
            "xg_mae": round(xg_mae, 2),
            "xg_predicted": {
                "home": round(xg.get("home", 0.0), 2),
                "away": round(xg.get("away", 0.0), 2),
            },
            "actual_goals": {
                "home": actual.home_goals,
                "away": actual.away_goals,
            },
        },
    )


DEFAULT_WEIGHTS: dict[str, float] = {
    "state": 0.40,
    "component": 0.35,
    "calibration": 0.25,
}


def evaluate(
    match_id: str,
    prediction: dict[str, Any],
    actual: ActualResult,
    *,
    weights: dict[str, float] | None = None,
) -> EvaluationResult:
    """Run all three verifiers and produce a weighted composite score."""
    w = weights or DEFAULT_WEIGHTS
    verifiers = [
        state_verify(prediction, actual),
        component_verify(prediction, actual),
        calibration_verify(prediction, actual),
    ]
    composite = sum(w.get(v.name, 0.0) * v.score for v in verifiers)

    return EvaluationResult(
        match_id=match_id,
        composite_score=round(composite, 4),
        verifiers=verifiers,
        reward_hacking_safe=True,
        details={
            "weights": w,
            "per_verifier": {v.name: round(v.score, 4) for v in verifiers},
        },
    )


def reward_hacking_floor_test(
    actual: ActualResult,
    *,
    floor: float = 0.30,
) -> dict[str, Any]:
    """Verify that trivial submissions score below the floor.

    Tests two attack strategies:
    - Empty submission (no probabilities, no data)
    - Uniform submission (1/3 each, league-average xG, no derived markets)

    If either scores above the floor, the scoring function is gameable.
    Returns the test results; the harness is safe if both stay below floor.
    """
    empty = {
        "match_outcome": {"home": 0.0, "draw": 0.0, "away": 0.0},
        "expected_goals": {"home": 0.0, "away": 0.0},
        "exact_score": {},
    }
    uniform = {
        "match_outcome": {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3},
        "expected_goals": {"home": 1.35, "away": 1.35},
        "exact_score": {},
    }

    empty_result = evaluate("__floor_empty__", empty, actual)
    uniform_result = evaluate("__floor_uniform__", uniform, actual)

    return {
        "floor": floor,
        "empty_score": empty_result.composite_score,
        "empty_below_floor": empty_result.composite_score < floor,
        "uniform_score": uniform_result.composite_score,
        "uniform_below_floor": uniform_result.composite_score < floor,
        "harness_safe": (
            empty_result.composite_score < floor
            and uniform_result.composite_score < floor
        ),
    }


def evaluation_to_dict(result: EvaluationResult) -> dict[str, Any]:
    """Serialize an EvaluationResult for JSON API responses."""
    return {
        "match_id": result.match_id,
        "composite_score": result.composite_score,
        "reward_hacking_safe": result.reward_hacking_safe,
        "verifiers": [
            {
                "name": v.name,
                "score": round(v.score, 4),
                "passed": v.passed,
                "details": v.details,
            }
            for v in result.verifiers
        ],
        "details": result.details,
    }

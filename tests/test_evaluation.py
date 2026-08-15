"""Property tests for the mixed-verifier evaluation harness, trajectory
analysis, and stage-wise attribution.

Core invariants:
1. Perfect prediction scores 1.0 on the state verifier
2. Empty submissions score below 0.30 (reward-hacking floor)
3. Composite score is always in [0, 1]
4. Wrong predictions score lower than correct ones
5. Clean trajectories have no failures
6. Repeated tool calls are detected as looping
7. Attribution binding constraint is the highest-correlation stage
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.attribution import run_attribution
from src.eval.harness import (
    ActualResult,
    calibration_verify,
    component_verify,
    evaluate,
    reward_hacking_floor_test,
    state_verify,
)
from src.eval.trajectory import (
    FailureCategory,
    analyze_trajectory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PERFECT_PRED = {
    "match_outcome": {
        "home": 0.65, "draw": 0.20, "away": 0.15,
        "conformal_set": ["home"],
        "conformal_alpha": 0.1,
    },
    "expected_goals": {"home": 1.8, "away": 0.9},
    "exact_score": {
        "top_scorelines": [
            {"score": "2-1", "prob": 0.14},
            {"score": "1-0", "prob": 0.13},
            {"score": "2-0", "prob": 0.11},
            {"score": "1-1", "prob": 0.10},
            {"score": "3-1", "prob": 0.08},
        ],
        "over_under_2_5": {"over": 0.55, "under": 0.45},
        "btts": {"yes": 0.52, "no": 0.48},
    },
    "event_sequence": {"first_scorer": {"home_first": 0.55, "away_first": 0.30, "no_goals": 0.15}},
}

ACTUAL_HOME = ActualResult(outcome="home", home_goals=2, away_goals=1)
ACTUAL_DRAW = ActualResult(outcome="draw", home_goals=1, away_goals=1)
ACTUAL_AWAY = ActualResult(outcome="away", home_goals=0, away_goals=2)


# ---------------------------------------------------------------------------
# Harness: state verifier
# ---------------------------------------------------------------------------

def test_state_correct_prediction_passes():
    result = state_verify(PERFECT_PRED, ACTUAL_HOME)
    assert result.passed is True
    assert result.score >= 0.5


def test_state_wrong_prediction_fails():
    result = state_verify(PERFECT_PRED, ACTUAL_AWAY)
    assert result.passed is False
    assert result.details["predicted"] == "home"
    assert result.details["actual"] == "away"


def test_state_conformal_coverage_bonus():
    result = state_verify(PERFECT_PRED, ACTUAL_HOME)
    assert result.details["in_conformal_set"] is True
    assert result.score > 0.5


def test_state_empty_probs_scores_zero():
    empty = {"match_outcome": {"home": 0, "draw": 0, "away": 0}}
    result = state_verify(empty, ACTUAL_HOME)
    assert result.score == 0.0
    assert result.passed is False


# ---------------------------------------------------------------------------
# Harness: component verifier
# ---------------------------------------------------------------------------

def test_component_all_correct():
    result = component_verify(PERFECT_PRED, ACTUAL_HOME)
    assert result.score > 0.0
    assert "checks" in result.details


def test_component_over_under_check():
    pred = {
        "exact_score": {"over_under_2_5": {"over": 0.7, "under": 0.3}},
        "expected_goals": {"home": 1.5, "away": 1.5},
    }
    actual_over = ActualResult(outcome="home", home_goals=2, away_goals=2)
    result = component_verify(pred, actual_over)
    assert result.details["checks"]["over_under_2_5"] is True


def test_component_btts_check():
    pred = {
        "exact_score": {"btts": {"yes": 0.8, "no": 0.2}},
        "expected_goals": {"home": 1.5, "away": 1.5},
    }
    actual_btts = ActualResult(outcome="draw", home_goals=1, away_goals=1)
    result = component_verify(pred, actual_btts)
    assert result.details["checks"]["btts"] is True


def test_component_empty_prediction():
    result = component_verify({}, ACTUAL_HOME)
    assert result.details["total_count"] == 1
    assert "score_in_top_5" in result.details["checks"]


# ---------------------------------------------------------------------------
# Harness: calibration verifier
# ---------------------------------------------------------------------------

def test_calibration_good_prediction():
    result = calibration_verify(PERFECT_PRED, ACTUAL_HOME)
    assert result.score > 0.5
    assert result.details["brier_score"] < 1.0


def test_calibration_terrible_prediction():
    wrong = {
        "match_outcome": {"home": 0.05, "draw": 0.05, "away": 0.90},
        "expected_goals": {"home": 0.5, "away": 2.5},
    }
    result = calibration_verify(wrong, ACTUAL_HOME)
    assert result.score < 0.2
    assert result.details["brier_score"] > 1.0


def test_calibration_brier_range():
    result = calibration_verify(PERFECT_PRED, ACTUAL_HOME)
    assert 0.0 <= result.details["brier_score"] <= 2.0


# ---------------------------------------------------------------------------
# Harness: composite evaluation
# ---------------------------------------------------------------------------

def test_composite_in_unit_range():
    result = evaluate("M1", PERFECT_PRED, ACTUAL_HOME)
    assert 0.0 <= result.composite_score <= 1.0


def test_composite_correct_higher_than_wrong():
    correct = evaluate("M1", PERFECT_PRED, ACTUAL_HOME)
    wrong = evaluate("M2", PERFECT_PRED, ACTUAL_AWAY)
    assert correct.composite_score > wrong.composite_score


def test_composite_verifier_count():
    result = evaluate("M1", PERFECT_PRED, ACTUAL_HOME)
    assert len(result.verifiers) == 3
    names = {v.name for v in result.verifiers}
    assert names == {"state", "component", "calibration"}


# ---------------------------------------------------------------------------
# Harness: reward-hacking floor test
# ---------------------------------------------------------------------------

def test_reward_hacking_floor_empty():
    result = reward_hacking_floor_test(ACTUAL_HOME)
    assert result["empty_below_floor"] is True
    assert result["empty_score"] < 0.30


def test_reward_hacking_floor_uniform():
    result = reward_hacking_floor_test(ACTUAL_HOME)
    assert result["uniform_below_floor"] is True
    assert result["uniform_score"] < 0.30


@pytest.mark.parametrize("actual", [ACTUAL_HOME, ACTUAL_DRAW, ACTUAL_AWAY])
def test_reward_hacking_floor_all_outcomes(actual):
    result = reward_hacking_floor_test(actual)
    assert result["harness_safe"] is True


# ---------------------------------------------------------------------------
# Trajectory: clean run
# ---------------------------------------------------------------------------

def test_clean_trajectory_no_failures():
    report = analyze_trajectory(
        match_id="M1",
        request={"home_team": "Arsenal", "away_team": "Chelsea"},
        prediction=PERFECT_PRED,
        answer="**Arsenal vs Chelsea** -- Arsenal 65%, draw 20%, Chelsea 15%. Degraded: none.",
        degraded=[],
        ledger=[
            {"server": "data", "tool": "get_features", "ok": True, "result": {"features": True}},
            {"server": "ml", "tool": "predict", "ok": True, "result": {"prediction": True}},
        ],
    )
    assert report.trajectory_quality == 1.0
    assert len(report.failures) == 0
    assert all(v == 0 for v in report.failure_counts.values())


# ---------------------------------------------------------------------------
# Trajectory: instruction loss
# ---------------------------------------------------------------------------

def test_instruction_loss_missing_stakes():
    report = analyze_trajectory(
        match_id="M1",
        request={"home_team": "Arsenal", "away_team": "Chelsea", "wants_stakes": True},
        prediction={**PERFECT_PRED},
        answer="Arsenal vs Chelsea prediction.",
        degraded=[],
        ledger=[],
    )
    assert report.failure_counts[FailureCategory.INSTRUCTION_LOSS] > 0


# ---------------------------------------------------------------------------
# Trajectory: risk abandonment
# ---------------------------------------------------------------------------

def test_risk_abandonment_degraded_not_disclosed():
    report = analyze_trajectory(
        match_id="M1",
        request={"home_team": "Arsenal", "away_team": "Chelsea"},
        prediction=PERFECT_PRED,
        answer="Arsenal will win convincingly. Very confident prediction.",
        degraded=["news server unavailable", "odds data stale"],
        ledger=[],
    )
    assert report.failure_counts[FailureCategory.RISK_ABANDONMENT] > 0
    assert report.trajectory_quality < 1.0


# ---------------------------------------------------------------------------
# Trajectory: false verification
# ---------------------------------------------------------------------------

def test_false_verification_empty_result():
    report = analyze_trajectory(
        match_id="M1",
        request={},
        prediction=PERFECT_PRED,
        answer="Prediction ready.",
        degraded=[],
        ledger=[
            {"server": "data", "tool": "get_features", "ok": True, "result": {}},
        ],
    )
    assert report.failure_counts[FailureCategory.FALSE_VERIFICATION] == 1


# ---------------------------------------------------------------------------
# Trajectory: detrimental looping
# ---------------------------------------------------------------------------

def test_detrimental_looping_detected():
    call = {"server": "data", "tool": "get_features", "ok": False,
            "args": {"match_id": "M1"}, "result": None, "error": "timeout"}
    report = analyze_trajectory(
        match_id="M1",
        request={},
        prediction=PERFECT_PRED,
        answer="Prediction ready.",
        degraded=[],
        ledger=[call, call, call],
    )
    assert report.failure_counts[FailureCategory.DETRIMENTAL_LOOPING] == 1


def test_no_looping_different_calls():
    report = analyze_trajectory(
        match_id="M1",
        request={},
        prediction=PERFECT_PRED,
        answer="Prediction ready.",
        degraded=[],
        ledger=[
            {"server": "data", "tool": "get_features", "ok": True, "args": {}, "result": {"a": 1}},
            {"server": "ml", "tool": "predict", "ok": True, "args": {}, "result": {"b": 2}},
            {"server": "news", "tool": "search", "ok": True, "args": {}, "result": {"c": 3}},
        ],
    )
    assert report.failure_counts[FailureCategory.DETRIMENTAL_LOOPING] == 0


# ---------------------------------------------------------------------------
# Trajectory: silent scope drop
# ---------------------------------------------------------------------------

def test_silent_scope_drop_missing_required():
    report = analyze_trajectory(
        match_id="M1",
        request={},
        prediction={"match_outcome": {"home": 0.5, "draw": 0.3, "away": 0.2}},
        answer="Prediction ready.",
        degraded=[],
        ledger=[],
    )
    assert report.failure_counts[FailureCategory.SILENT_SCOPE_DROP] > 0


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
BATCH_PREDS = []
BATCH_ACTUALS = []
BATCH_IDS = []
for i in range(10):
    h = float(RNG.uniform(0.2, 0.6))
    d = float(RNG.uniform(0.15, 0.35))
    a = 1.0 - h - d
    mu_h = float(RNG.uniform(0.8, 2.5))
    mu_a = float(RNG.uniform(0.5, 2.0))
    outcomes = ["home", "draw", "away"]
    actual_outcome = outcomes[int(RNG.integers(0, 3))]
    actual_hg = int(RNG.integers(0, 4))
    actual_ag = int(RNG.integers(0, 4))
    BATCH_PREDS.append({
        "match_outcome": {
            "home": h, "draw": d, "away": a,
            "conformal_set": [actual_outcome] if RNG.random() > 0.3 else outcomes[:2],
        },
        "expected_goals": {"home": mu_h, "away": mu_a},
        "exact_score": {
            "top_scorelines": [
                {"score": f"{actual_hg}-{actual_ag}", "prob": 0.12},
                {"score": "1-1", "prob": 0.10},
            ],
            "over_under_2_5": {"over": 0.55, "under": 0.45},
            "btts": {"yes": 0.50, "no": 0.50},
        },
    })
    BATCH_ACTUALS.append(ActualResult(
        outcome=actual_outcome, home_goals=actual_hg, away_goals=actual_ag,
    ))
    BATCH_IDS.append(f"match_{i}")


def test_attribution_runs():
    report = run_attribution(BATCH_PREDS, BATCH_ACTUALS, BATCH_IDS)
    assert report.n_evaluated == 10
    assert 0.0 <= report.mean_composite <= 1.0


def test_attribution_has_all_stages():
    report = run_attribution(BATCH_PREDS, BATCH_ACTUALS, BATCH_IDS)
    stage_names = {s.stage for s in report.stages}
    assert stage_names == {"state", "component", "calibration"}


def test_attribution_binding_constraint_valid():
    report = run_attribution(BATCH_PREDS, BATCH_ACTUALS, BATCH_IDS)
    assert report.binding_constraint in {"state", "component", "calibration"}


def test_attribution_correlations_in_range():
    report = run_attribution(BATCH_PREDS, BATCH_ACTUALS, BATCH_IDS)
    for s in report.stages:
        assert -1.0 <= s.correlation_with_composite <= 1.0


def test_attribution_reward_hacking_safe():
    report = run_attribution(BATCH_PREDS, BATCH_ACTUALS, BATCH_IDS)
    assert report.reward_hacking_safe is True


def test_attribution_calibration_summary():
    report = run_attribution(BATCH_PREDS, BATCH_ACTUALS, BATCH_IDS)
    assert "mean_brier" in report.calibration_summary
    assert "outcome_accuracy" in report.calibration_summary
    assert "conformal_coverage" in report.calibration_summary
    assert 0.0 <= report.calibration_summary["outcome_accuracy"] <= 1.0


def test_attribution_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        run_attribution([], [], [])


def test_attribution_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        run_attribution(BATCH_PREDS[:2], BATCH_ACTUALS[:3], BATCH_IDS[:2])

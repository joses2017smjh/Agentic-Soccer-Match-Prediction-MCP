"""Trajectory recording and failure taxonomy analysis.

Maps the Halluminate Diligence Bench failure categories to prediction agent
runs, enabling systematic classification of agent failures:

- **Instruction loss over horizon** (42.4% in Diligence Bench): constraints
  present in the request but not reflected in the agent's output.
- **Risk abandonment**: degraded evidence flags identified during execution
  but not carried into the final answer or confidence assessment.
- **False verification**: tool calls that report success but return
  empty or structurally invalid results.
- **Silent scope drop**: expected prediction components absent from the
  output without explanation or fallback.
- **Detrimental looping**: repeated identical tool calls with no state
  change between attempts.

Each failure is classified with evidence pointers back into the trajectory,
producing an auditable record suitable for post-training or environment design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class FailureCategory:
    INSTRUCTION_LOSS = "instruction_loss"
    RISK_ABANDONMENT = "risk_abandonment"
    FALSE_VERIFICATION = "false_verification"
    SILENT_SCOPE_DROP = "silent_scope_drop"
    DETRIMENTAL_LOOPING = "detrimental_looping"

    ALL = (
        INSTRUCTION_LOSS,
        RISK_ABANDONMENT,
        FALSE_VERIFICATION,
        SILENT_SCOPE_DROP,
        DETRIMENTAL_LOOPING,
    )


FAILURE_DESCRIPTIONS: dict[str, str] = {
    FailureCategory.INSTRUCTION_LOSS: (
        "Constraints present in the request were not reflected in the "
        "agent's behavior or output."
    ),
    FailureCategory.RISK_ABANDONMENT: (
        "Degraded evidence or risk flags were identified during execution "
        "but not carried into the final answer."
    ),
    FailureCategory.FALSE_VERIFICATION: (
        "Tool calls reported success but returned empty or structurally "
        "invalid results."
    ),
    FailureCategory.SILENT_SCOPE_DROP: (
        "Expected prediction components are missing from the output "
        "without explanation."
    ),
    FailureCategory.DETRIMENTAL_LOOPING: (
        "Repeated identical tool calls with no state change."
    ),
}


@dataclass
class FailureClassification:
    """One detected failure with evidence."""

    category: str
    severity: str  # "high" | "medium" | "low"
    evidence: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryReport:
    """Full analysis of one agent run."""

    match_id: str
    n_tool_calls: int
    n_failed_calls: int
    elapsed_ms: float
    failures: list[FailureClassification]
    failure_counts: dict[str, int]
    trajectory_quality: float  # 0-1: 1.0 = no failures detected
    details: dict[str, Any] = field(default_factory=dict)


def _detect_instruction_loss(
    request: dict[str, Any],
    prediction: dict[str, Any] | None,
    answer: str,
) -> list[FailureClassification]:
    """Constraints in request not reflected in output."""
    failures: list[FailureClassification] = []

    if request.get("wants_stakes") and prediction:
        has_suggestions = bool(prediction.get("suggestions"))
        mentions_stakes = any(
            kw in answer.lower()
            for kw in ("suggestion", "stake", "kelly", "value bet", "approval")
        )
        if not has_suggestions and not mentions_stakes:
            failures.append(FailureClassification(
                category=FailureCategory.INSTRUCTION_LOSS,
                severity="high",
                evidence=(
                    "Request specified wants_stakes=True but no suggestions "
                    "were computed and answer does not mention staking."
                ),
            ))

    if request.get("knockout") and prediction:
        if "knockout" not in prediction:
            failures.append(FailureClassification(
                category=FailureCategory.INSTRUCTION_LOSS,
                severity="medium",
                evidence=(
                    "Knockout mode implied by request but prediction "
                    "contains no knockout/advancement data."
                ),
            ))

    if request.get("home_team") and request.get("away_team") and answer:
        home = request["home_team"].lower()
        away = request["away_team"].lower()
        answer_lower = answer.lower()
        if home not in answer_lower or away not in answer_lower:
            failures.append(FailureClassification(
                category=FailureCategory.INSTRUCTION_LOSS,
                severity="low",
                evidence=(
                    f"Team names ({request['home_team']}, {request['away_team']}) "
                    "not found in the answer text."
                ),
            ))

    return failures


def _detect_risk_abandonment(
    degraded: list[str],
    answer: str,
    prediction: dict[str, Any] | None,
) -> list[FailureClassification]:
    """Degraded flags identified but not disclosed in the answer."""
    failures: list[FailureClassification] = []
    if not degraded:
        return failures

    degradation_keywords = [
        "degraded", "unavailable", "reduced confidence",
        "fallback", "without", "failed",
    ]
    answer_lower = answer.lower()
    mentions_degradation = any(kw in answer_lower for kw in degradation_keywords)

    if not mentions_degradation:
        failures.append(FailureClassification(
            category=FailureCategory.RISK_ABANDONMENT,
            severity="high",
            evidence=(
                f"{len(degraded)} degraded evidence source(s) identified "
                f"during execution but the answer does not disclose any "
                f"reduced confidence. Flags: {degraded[:3]}"
            ),
            details={"degraded_flags": degraded},
        ))

    conf_set = (prediction or {}).get("match_outcome", {}).get("conformal_set", [])
    if len(conf_set) > 1 and degraded:
        if "uncertain" not in answer_lower and "open match" not in answer_lower:
            failures.append(FailureClassification(
                category=FailureCategory.RISK_ABANDONMENT,
                severity="medium",
                evidence=(
                    f"Conformal set is wide ({conf_set}) and evidence is degraded, "
                    "but the answer does not flag uncertainty."
                ),
            ))

    return failures


def _detect_false_verification(
    ledger: list[dict[str, Any]],
) -> list[FailureClassification]:
    """Tool calls that claimed success but returned empty/invalid data."""
    failures: list[FailureClassification] = []
    for i, call in enumerate(ledger):
        if not call.get("ok"):
            continue
        result = call.get("result")
        if result is None or result == {} or result == []:
            failures.append(FailureClassification(
                category=FailureCategory.FALSE_VERIFICATION,
                severity="high",
                evidence=(
                    f"Tool call #{i} ({call.get('server')}.{call.get('tool')}) "
                    f"reported ok=True but returned empty result."
                ),
                details={"call_index": i, "server": call.get("server"),
                         "tool": call.get("tool")},
            ))
    return failures


def _detect_silent_scope_drop(
    prediction: dict[str, Any] | None,
    had_player_data: bool,
    had_market_quotes: bool,
) -> list[FailureClassification]:
    """Expected prediction components missing without explanation."""
    failures: list[FailureClassification] = []
    if prediction is None:
        return failures

    if had_player_data and "player_props" not in prediction:
        failures.append(FailureClassification(
            category=FailureCategory.SILENT_SCOPE_DROP,
            severity="medium",
            evidence=(
                "Player data was provided but player_props is missing "
                "from the prediction output."
            ),
        ))

    if had_market_quotes and "suggestions" not in prediction:
        failures.append(FailureClassification(
            category=FailureCategory.SILENT_SCOPE_DROP,
            severity="low",
            evidence=(
                "Market quotes were provided but no suggestions were "
                "computed (may be intentional if no edge found)."
            ),
        ))

    required_keys = ["match_outcome", "expected_goals", "exact_score", "event_sequence"]
    for key in required_keys:
        if key not in prediction:
            failures.append(FailureClassification(
                category=FailureCategory.SILENT_SCOPE_DROP,
                severity="high",
                evidence=f"Required prediction component '{key}' is missing.",
            ))

    return failures


def _detect_detrimental_looping(
    ledger: list[dict[str, Any]],
    *,
    threshold: int = 3,
) -> list[FailureClassification]:
    """Repeated identical tool calls with no state change."""
    failures: list[FailureClassification] = []
    if len(ledger) < threshold:
        return failures

    for i in range(len(ledger) - threshold + 1):
        window = ledger[i : i + threshold]
        sigs = [
            (c.get("server"), c.get("tool"), str(c.get("args", {})))
            for c in window
        ]
        if len(set(sigs)) == 1:
            call = window[0]
            failures.append(FailureClassification(
                category=FailureCategory.DETRIMENTAL_LOOPING,
                severity="medium",
                evidence=(
                    f"{threshold} consecutive identical calls to "
                    f"{call.get('server')}.{call.get('tool')} "
                    f"starting at index {i}."
                ),
                details={"start_index": i, "repeat_count": threshold},
            ))
            break  # report once per run

    return failures


def analyze_trajectory(
    *,
    match_id: str,
    request: dict[str, Any],
    prediction: dict[str, Any] | None,
    answer: str,
    degraded: list[str],
    ledger: list[dict[str, Any]],
    elapsed_ms: float = 0.0,
    had_player_data: bool = False,
    had_market_quotes: bool = False,
) -> TrajectoryReport:
    """Full trajectory analysis with failure taxonomy classification.

    Returns a TrajectoryReport with all detected failures classified by
    category, severity, and evidence. The trajectory_quality score is
    1.0 minus a penalty for each failure (weighted by severity).
    """
    all_failures: list[FailureClassification] = []

    all_failures.extend(_detect_instruction_loss(request, prediction, answer))
    all_failures.extend(_detect_risk_abandonment(degraded, answer, prediction))
    all_failures.extend(_detect_false_verification(ledger))
    all_failures.extend(_detect_silent_scope_drop(
        prediction, had_player_data, had_market_quotes,
    ))
    all_failures.extend(_detect_detrimental_looping(ledger))

    severity_penalty = {"high": 0.20, "medium": 0.10, "low": 0.05}
    penalty = sum(severity_penalty.get(f.severity, 0.05) for f in all_failures)
    quality = max(0.0, 1.0 - penalty)

    counts: dict[str, int] = {cat: 0 for cat in FailureCategory.ALL}
    for f in all_failures:
        counts[f.category] = counts.get(f.category, 0) + 1

    return TrajectoryReport(
        match_id=match_id,
        n_tool_calls=len(ledger),
        n_failed_calls=sum(1 for c in ledger if not c.get("ok")),
        elapsed_ms=elapsed_ms,
        failures=all_failures,
        failure_counts=counts,
        trajectory_quality=round(quality, 4),
        details={
            "n_failures": len(all_failures),
            "severity_distribution": {
                sev: sum(1 for f in all_failures if f.severity == sev)
                for sev in ("high", "medium", "low")
            },
        },
    )


def trajectory_to_dict(report: TrajectoryReport) -> dict[str, Any]:
    """Serialize a TrajectoryReport for JSON API responses."""
    return {
        "match_id": report.match_id,
        "n_tool_calls": report.n_tool_calls,
        "n_failed_calls": report.n_failed_calls,
        "elapsed_ms": round(report.elapsed_ms, 1),
        "trajectory_quality": report.trajectory_quality,
        "failure_counts": report.failure_counts,
        "failures": [
            {
                "category": f.category,
                "severity": f.severity,
                "evidence": f.evidence,
                "description": FAILURE_DESCRIPTIONS.get(f.category, ""),
            }
            for f in report.failures
        ],
        "details": report.details,
    }

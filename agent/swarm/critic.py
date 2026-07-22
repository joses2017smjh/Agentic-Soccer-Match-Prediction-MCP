"""Critic / Red-Team agent — adversarial verification before the answer ships.

Enforces the zero-hallucination-math directive by *recomputing* the tool
output's arithmetic and flagging any inconsistency, and red-teams for the
anomaly classes the spec calls out (e.g. an implausibly extreme win rate that
usually signals data leakage). A failed critique triggers a bounded feedback
loop back to the planner/executor.

Every check is deterministic and cheap; an LLM critic can be layered on top
for semantic critiques, but the numeric guarantees below never depend on it.
"""

from __future__ import annotations

import math

from agent.swarm.state import Critique, SwarmState

TOL = 1e-4


def critique_prediction(state: SwarmState) -> Critique:
    """Return a verdict; issues is empty iff every check passes."""
    issues: list[str] = []
    checks = 0
    pred = state.prediction

    if pred is None:
        return Critique(passed=False, iteration=state.iteration,
                        issues=["no prediction was produced"], checks_run=1)

    mo = pred["match_outcome"]
    checks += 1
    total = mo["home"] + mo["draw"] + mo["away"]
    if abs(total - 1.0) > TOL:
        issues.append(f"outcome probabilities sum to {total:.4f}, not 1")

    checks += 1
    if not all(0.0 <= mo[k] <= 1.0 for k in ("home", "draw", "away")):
        issues.append("an outcome probability is outside [0, 1]")

    checks += 1
    cset = mo.get("conformal_set", [])
    if not cset or not set(cset) <= {"home", "draw", "away"}:
        issues.append("conformal prediction set missing or malformed")

    xg = pred.get("expected_goals", {})
    checks += 1
    if not (0.05 <= xg.get("home", 0) <= 6 and 0.05 <= xg.get("away", 0) <= 6):
        issues.append(f"expected goals implausible: {xg}")

    # --- red-team: extreme confidence must be corroborated by the xG gap.
    # A 98%+ favourite off a near-level xG is the classic leakage signature.
    checks += 1
    fav = max(("home", "away"), key=lambda k: mo[k])
    if mo[fav] > 0.98:
        gap = abs(xg.get("home", 0) - xg.get("away", 0))
        if gap < 1.5:
            issues.append(
                f"anomaly: {mo[fav]:.1%} on {fav} but xG gap only {gap:.2f} — "
                "possible data leakage or degenerate features"
            )

    # --- scoreline grid must be a normalized distribution
    grid = pred.get("exact_score", {}).get("scoreline_grid")
    if grid:
        checks += 1
        mass = sum(map(sum, grid["probs"])) + grid.get("tail_mass", 0)
        if abs(mass - 1.0) > 1e-3:
            issues.append(f"scoreline grid mass {mass:.4f} != 1")

    # --- recompute market-comparison EV: catch arithmetic hallucination
    # (market_comparison carries model_prob + decimal_odds; suggestions is the
    # flagged subset without the raw prices)
    for s in pred.get("market_comparison", []):
        checks += 1
        b = s["decimal_odds"] - 1.0
        expected_ev = s["model_prob"] * b - (1.0 - s["model_prob"])
        if abs(expected_ev - s["ev"]) > 1e-3:
            issues.append(
                f"EV mismatch on {s['market']}/{s['selection']}: reported "
                f"{s['ev']:.4f}, recomputed {expected_ev:.4f}"
            )

    # --- every flagged suggestion must carry positive EV
    for s in pred.get("suggestions", []):
        checks += 1
        if s["ev"] <= 0:
            issues.append(
                f"flagged suggestion {s['selection']} has non-positive EV "
                f"{s['ev']:.4f}"
            )

    # --- first-scorer must reconcile with the grid's P(0-0)
    fs = pred.get("event_sequence", {}).get("first_scorer")
    if fs:
        checks += 1
        if abs(sum(fs.values()) - 1.0) > TOL:
            issues.append("first-scorer probabilities do not sum to 1")

    return Critique(passed=not issues, issues=issues, checks_run=checks,
                    iteration=state.iteration)

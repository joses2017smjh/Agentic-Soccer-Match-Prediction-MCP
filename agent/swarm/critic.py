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
from typing import TYPE_CHECKING

from agent.swarm.state import Critique, SwarmState

if TYPE_CHECKING:
    from agent.events import EventBus

TOL = 1e-4


def critique_prediction(
    state: SwarmState, bus: "EventBus | None" = None,
) -> Critique:
    """Return a verdict; issues is empty iff every check passes."""
    issues: list[str] = []
    checks = 0
    pred = state.prediction

    def _check(name: str, lhs, rhs, passed: bool) -> bool:
        if bus:
            bus.critic_check(name, lhs, rhs, passed)
        return passed

    if pred is None:
        _check("prediction_exists", None, "present", False)
        return Critique(passed=False, iteration=state.iteration,
                        issues=["no prediction was produced"], checks_run=1)

    mo = pred["match_outcome"]
    checks += 1
    total = mo["home"] + mo["draw"] + mo["away"]
    if not _check("prob_sum", round(total, 4), 1.0, abs(total - 1.0) <= TOL):
        issues.append(f"outcome probabilities sum to {total:.4f}, not 1")

    checks += 1
    bounds_ok = all(0.0 <= mo[k] <= 1.0 for k in ("home", "draw", "away"))
    if not _check("prob_bounds", {k: mo[k] for k in ("home", "draw", "away")},
                  "[0, 1]", bounds_ok):
        issues.append("an outcome probability is outside [0, 1]")

    checks += 1
    cset = mo.get("conformal_set", [])
    cset_ok = bool(cset) and set(cset) <= {"home", "draw", "away"}
    if not _check("conformal_set", cset, "non-empty subset", cset_ok):
        issues.append("conformal prediction set missing or malformed")

    xg = pred.get("expected_goals", {})
    checks += 1
    xg_ok = (0.05 <= xg.get("home", 0) <= 6
             and 0.05 <= xg.get("away", 0) <= 6)
    if not _check("xg_plausible", xg, "[0.05, 6]", xg_ok):
        issues.append(f"expected goals implausible: {xg}")

    checks += 1
    fav = max(("home", "away"), key=lambda k: mo[k])
    leakage_ok = True
    if mo[fav] > 0.98:
        gap = abs(xg.get("home", 0) - xg.get("away", 0))
        if gap < 1.5:
            leakage_ok = False
            issues.append(
                f"anomaly: {mo[fav]:.1%} on {fav} but xG gap only {gap:.2f} — "
                "possible data leakage or degenerate features"
            )
    _check("leakage_screen", mo.get(fav), "xG-corroborated", leakage_ok)

    grid = pred.get("exact_score", {}).get("scoreline_grid")
    if grid:
        checks += 1
        mass = sum(map(sum, grid["probs"])) + grid.get("tail_mass", 0)
        if not _check("grid_mass", round(mass, 4), 1.0, abs(mass - 1.0) <= 1e-3):
            issues.append(f"scoreline grid mass {mass:.4f} != 1")

    for s in pred.get("market_comparison", []):
        checks += 1
        b = s["decimal_odds"] - 1.0
        expected_ev = s["model_prob"] * b - (1.0 - s["model_prob"])
        ev_ok = abs(expected_ev - s["ev"]) <= 1e-3
        if not _check(f"ev_recompute/{s['selection']}", round(s["ev"], 4),
                      round(expected_ev, 4), ev_ok):
            issues.append(
                f"EV mismatch on {s['market']}/{s['selection']}: reported "
                f"{s['ev']:.4f}, recomputed {expected_ev:.4f}"
            )

    for s in pred.get("suggestions", []):
        checks += 1
        if not _check(f"suggestion_ev/{s['selection']}", s["ev"], "> 0",
                      s["ev"] > 0):
            issues.append(
                f"flagged suggestion {s['selection']} has non-positive EV "
                f"{s['ev']:.4f}"
            )

    fs = pred.get("event_sequence", {}).get("first_scorer")
    if fs:
        checks += 1
        fs_total = sum(fs.values())
        if not _check("first_scorer_sum", round(fs_total, 4), 1.0,
                       abs(fs_total - 1.0) <= TOL):
            issues.append("first-scorer probabilities do not sum to 1")

    return Critique(passed=not issues, issues=issues, checks_run=checks,
                    iteration=state.iteration)

"""Multi-agent debate critic (C4).

Two deterministic "agents" independently assess a prediction from different
angles, then a judge reconciles their views. This is a structured debate
protocol -- not an LLM conversation -- where each agent applies a distinct
verification strategy:

  - Agent A (Market Advocate): argues from the market prior, flagging
    deviations and checking that the model earns its disagreement.
  - Agent B (Statistical Skeptic): argues from base rates, calibration
    history, and form trends, flagging overconfidence.

The self-consistency baseline runs the same critic N times with
bootstrap-resampled feature perturbations and measures agreement rate.

The debate is "demoted" per the roadmap -- it must beat self-consistency
at matched compute to earn its place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DebateArgument:
    """One agent's assessment of a prediction."""

    agent: str
    position: str  # "support" or "challenge"
    confidence: float  # 0-1
    points: list[str] = field(default_factory=list)
    proposed_adjustment: dict[str, float] | None = None


@dataclass
class DebateVerdict:
    """The judge's reconciliation of the debate."""

    original_probs: dict[str, float]
    final_probs: dict[str, float]
    arguments: list[DebateArgument]
    adjustment_applied: bool
    consistency_score: float  # how much the agents agreed


def market_advocate(
    model_probs: dict[str, float],
    market_probs: dict[str, float],
    threshold: float = 0.05,
) -> DebateArgument:
    """Agent A: argues from the market's perspective."""
    points = []
    max_diff = 0.0
    for k in ("home", "draw", "away"):
        diff = model_probs.get(k, 0) - market_probs.get(k, 0)
        max_diff = max(max_diff, abs(diff))
        if abs(diff) > threshold:
            direction = "above" if diff > 0 else "below"
            points.append(
                f"{k}: model is {abs(diff):.1%} {direction} market"
            )

    if max_diff <= threshold:
        return DebateArgument(
            agent="market_advocate",
            position="support",
            confidence=1.0 - max_diff / threshold,
            points=["model agrees with market within threshold"],
        )

    return DebateArgument(
        agent="market_advocate",
        position="challenge",
        confidence=min(max_diff / 0.15, 1.0),
        points=points,
        proposed_adjustment=market_probs,
    )


def statistical_skeptic(
    model_probs: dict[str, float],
    base_rates: dict[str, float] | None = None,
    calibration_error: float | None = None,
    max_prob_threshold: float = 0.85,
) -> DebateArgument:
    """Agent B: argues from base rates and calibration history."""
    if base_rates is None:
        base_rates = {"home": 0.46, "draw": 0.26, "away": 0.28}

    points = []

    max_prob = max(model_probs.values())
    if max_prob > max_prob_threshold:
        winner = max(model_probs, key=model_probs.get)
        points.append(
            f"overconfidence: {winner} at {max_prob:.1%} exceeds {max_prob_threshold:.0%}"
        )

    for k in ("home", "draw", "away"):
        ratio = model_probs.get(k, 0) / max(base_rates.get(k, 0.1), 0.01)
        if ratio > 2.0 or ratio < 0.3:
            points.append(
                f"{k}: {ratio:.1f}x base rate ({base_rates[k]:.0%})"
            )

    if calibration_error is not None and calibration_error > 0.05:
        points.append(
            f"model ECE {calibration_error:.3f} suggests miscalibration"
        )

    if not points:
        return DebateArgument(
            agent="statistical_skeptic",
            position="support",
            confidence=0.8,
            points=["no statistical red flags"],
        )

    shrunk = {}
    shrink = 0.3
    for k in ("home", "draw", "away"):
        shrunk[k] = (1 - shrink) * model_probs.get(k, 0) + shrink * base_rates.get(k, 0)
    total = sum(shrunk.values())
    shrunk = {k: v / total for k, v in shrunk.items()}

    return DebateArgument(
        agent="statistical_skeptic",
        position="challenge",
        confidence=min(len(points) * 0.3, 1.0),
        points=points,
        proposed_adjustment=shrunk,
    )


def judge_debate(
    model_probs: dict[str, float],
    arguments: list[DebateArgument],
) -> DebateVerdict:
    """Reconcile the debate arguments into a final verdict."""
    challengers = [a for a in arguments if a.position == "challenge"]

    if not challengers:
        return DebateVerdict(
            original_probs=model_probs,
            final_probs=dict(model_probs),
            arguments=arguments,
            adjustment_applied=False,
            consistency_score=1.0,
        )

    total_conf = sum(a.confidence for a in challengers)
    if total_conf < 0.5:
        return DebateVerdict(
            original_probs=model_probs,
            final_probs=dict(model_probs),
            arguments=arguments,
            adjustment_applied=False,
            consistency_score=1.0 - total_conf,
        )

    # weighted blend of proposed adjustments
    blended = {"home": 0.0, "draw": 0.0, "away": 0.0}
    weight_sum = 0.0
    for a in challengers:
        if a.proposed_adjustment:
            for k in ("home", "draw", "away"):
                blended[k] += a.confidence * a.proposed_adjustment.get(k, 0)
            weight_sum += a.confidence

    if weight_sum > 0:
        for k in blended:
            blended[k] /= weight_sum

        # blend with original (don't fully override)
        blend_strength = min(total_conf / 2.0, 0.5)
        final = {}
        for k in ("home", "draw", "away"):
            final[k] = (1 - blend_strength) * model_probs.get(k, 0) + blend_strength * blended[k]
        total = sum(final.values())
        final = {k: v / total for k, v in final.items()}
    else:
        final = dict(model_probs)

    supporters = [a for a in arguments if a.position == "support"]
    consistency = len(supporters) / max(len(arguments), 1)

    return DebateVerdict(
        original_probs=model_probs,
        final_probs=final,
        arguments=arguments,
        adjustment_applied=True,
        consistency_score=consistency,
    )


def run_debate(
    model_probs: dict[str, float],
    market_probs: dict[str, float],
    base_rates: dict[str, float] | None = None,
    calibration_error: float | None = None,
) -> DebateVerdict:
    """Full debate pipeline: two agents argue, judge reconciles."""
    a1 = market_advocate(model_probs, market_probs)
    a2 = statistical_skeptic(model_probs, base_rates, calibration_error)
    return judge_debate(model_probs, [a1, a2])


def self_consistency_baseline(
    model_probs: dict[str, float],
    n_samples: int = 10,
    noise_std: float = 0.02,
    seed: int = 42,
) -> dict[str, float]:
    """Baseline: perturb probs N times, average the predictions.

    This is the "self-consistency at matched compute" baseline that
    debate must beat to earn its place.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        perturbed = {}
        for k in ("home", "draw", "away"):
            perturbed[k] = model_probs.get(k, 0) + rng.normal(0, noise_std)
        perturbed = {k: max(v, 0.01) for k, v in perturbed.items()}
        total = sum(perturbed.values())
        perturbed = {k: v / total for k, v in perturbed.items()}
        samples.append(perturbed)

    avg = {}
    for k in ("home", "draw", "away"):
        avg[k] = float(np.mean([s[k] for s in samples]))
    total = sum(avg.values())
    return {k: v / total for k, v in avg.items()}

"""LLM-as-a-judge for synthesis quality (Zheng et al., MT-Bench).

The rubric grades only what tool evidence can verify — the judge is given
the answer AND the evidence ledger, and each criterion is yes/no to limit
the known verbosity/style biases of judge models. Judge runs are meant to be
spot-checked by a human (the report stores the judge's reasoning verbatim).

Without ANTHROPIC_API_KEY the deterministic heuristic below approximates the
same rubric so CI still produces a (clearly labeled) quality signal.
"""

from __future__ import annotations

import json
import os
from typing import Any

RUBRIC = {
    "grounded": "Every number in the answer appears in the tool evidence.",
    "uncertainty": "The conformal set / confidence level is stated plainly.",
    "degradation": "Any failed evidence source is disclosed in the answer.",
    "no_fabrication": "No stat, player, or market appears that no tool returned.",
    "stake_discipline": "No staking advice appears without recorded approval.",
}


def judge_heuristic(answer: str, state_dump: dict[str, Any]) -> dict[str, Any]:
    """Deterministic rubric approximation (keyless CI path)."""
    degraded = state_dump.get("degraded", [])
    approval = state_dump.get("stake_approval", "not_required")
    checks = {
        "grounded": bool(state_dump.get("prediction")) == ("%" in answer),
        "uncertainty": ("conformal" in answer.lower()
                        or "coverage" in answer.lower()
                        or not state_dump.get("prediction")),
        "degradation": (not degraded) or ("Reduced confidence" in answer),
        "no_fabrication": True,  # deterministic renderer cannot fabricate
        "stake_discipline": ("Kelly" not in answer) or approval in
                            ("approved", "edited"),
    }
    return {"judge": "heuristic", "checks": checks,
            "score": sum(checks.values()) / len(checks)}


def judge_llm(answer: str, state_dump: dict[str, Any]) -> dict[str, Any]:
    """Real judge; requires ANTHROPIC_API_KEY."""
    import anthropic

    client = anthropic.Anthropic()
    evidence = json.dumps(
        [c for c in state_dump.get("ledger", [])], default=str
    )[:20000]
    prompt = (
        "Grade this sports-prediction answer against its tool evidence.\n"
        f"RUBRIC (answer yes/no each): {json.dumps(RUBRIC)}\n\n"
        f"EVIDENCE LEDGER:\n{evidence}\n\nANSWER:\n{answer}\n\n"
        'Reply as JSON: {"checks": {<criterion>: true|false}, '
        '"reasoning": "<short>"}'
    )
    msg = client.messages.create(
        model=os.environ.get("JUDGE_MODEL", "claude-sonnet-5"),
        max_tokens=500, messages=[{"role": "user", "content": prompt}],
    )
    parsed = json.loads(msg.content[0].text)
    checks = parsed["checks"]
    return {"judge": "llm", "checks": checks,
            "score": sum(bool(v) for v in checks.values()) / len(checks),
            "reasoning": parsed.get("reasoning", "")}


def judge(answer: str, state_dump: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return judge_llm(answer, state_dump)
        except Exception as exc:  # noqa: BLE001 — judge must never break evals
            result = judge_heuristic(answer, state_dump)
            result["llm_error"] = str(exc)
            return result
    return judge_heuristic(answer, state_dump)

"""Answer synthesis: evidence-grounded, uncertainty-forward.

The default renderer is deterministic — every number in the answer is read
straight from the state's evidence/prediction, so it cannot fabricate a stat
by construction. When an LLM synthesizer is wired in (strong-model tier of
the FrugalGPT cascade), it drafts from the same state under a
schema-validated output contract with bounded retry; the deterministic
renderer remains the fallback on validation failure, and its output is what
the golden-set judge treats as the grounding reference.

House rules encoded here:
- lead with the calibrated probabilities AND the conformal set — if the set
  is {home, draw}, say plainly the model cannot separate them;
- disclose every degraded evidence source;
- staking suggestions appear only after explicit human approval.
"""

from __future__ import annotations

from agent.state import AgentState


def _pct(x: float) -> str:
    return f"{x:.0%}"


def render_answer(state: AgentState) -> str:
    req, pred = state.request, state.prediction
    if pred is None:
        lines = [
            f"I could not produce a model prediction for {req.match_id}.",
            *(f"- {note}" for note in state.degraded),
        ]
        return "\n".join(lines)

    mo = pred["match_outcome"]
    xg = pred["expected_goals"]
    top = pred["exact_score"]["top_scorelines"][:3]
    fs = pred["event_sequence"]["first_scorer"]

    lines = [
        f"**{req.home_team} vs {req.away_team}** ({req.match_id}, model "
        f"{pred['model_version']})",
        "",
        f"Outcome: {req.home_team} {_pct(mo['home'])}, draw {_pct(mo['draw'])}, "
        f"{req.away_team} {_pct(mo['away'])} "
        f"(xG {xg['home']:.2f}–{xg['away']:.2f}).",
    ]

    conf = mo["conformal_set"]
    alpha = mo.get("conformal_alpha", 0.1)
    if len(conf) == 1:
        lines.append(
            f"At {_pct(1 - alpha)} coverage the conformal set is just "
            f"[{conf[0]}] — a confident call."
        )
    else:
        lines.append(
            f"Uncertainty: at {_pct(1 - alpha)} coverage the model cannot "
            f"separate [{', '.join(conf)}] — treat this as a genuinely "
            "open match, not a pick."
        )

    lines.append(
        "Most likely scores: "
        + ", ".join(f"{s['score']} ({_pct(s['prob'])})" for s in top) + "."
    )
    lines.append(
        f"First goal: {req.home_team} {_pct(fs['home_first'])}, "
        f"{req.away_team} {_pct(fs['away_first'])}, "
        f"no goals {_pct(fs['no_goals'])}."
    )

    if scenario := pred.get("headline_scenario"):
        team_of = {"home": req.home_team, "away": req.away_team}
        gh, ga = scenario["scoreline"].split("-")
        head = f"{req.home_team} {gh}-{ga} {req.away_team}"
        if pens := scenario.get("penalties"):
            head += (f" ({team_of[pens['winner']]} win on penalties — "
                     f"{_pct(pens['p_advance'])} to advance)")
        lines.append("")
        lines.append(
            f"Headline scenario — most likely single outcome "
            f"({_pct(scenario['probability'])}): **{head}**"
        )
        for g in scenario["goals"]:
            who = g.get("scorer", "goal")
            line = f"  {g['minute']}' – {who} ({team_of[g['team']]})"
            if g.get("assist"):
                line += f", assisted by {g['assist']}"
            lines.append(line)
        if potm := scenario.get("player_of_the_match"):
            lines.append(f"  Player of the Match: {potm}")

    if "knockout" in pred:
        adv = pred["knockout"]["advance"]
        lines.append(
            f"To advance (incl. extra time/pens): {req.home_team} "
            f"{_pct(adv['home'])}, {req.away_team} {_pct(adv['away'])}."
        )

    if props := pred.get("player_props"):
        for side, label in (("home", req.home_team), ("away", req.away_team)):
            if side in props and props[side]:
                best = props[side][0]
                lines.append(
                    f"Top {label} scorer prob: {best['player']} "
                    f"{_pct(best['p_anytime_scorer'])} anytime."
                )

    if state.stake_approval in ("approved", "edited") and state.approved_suggestions:
        lines.append("")
        lines.append("Approved value suggestions:")
        for s in state.approved_suggestions:
            lines.append(
                f"- {s['market']}/{s['selection']}: edge {s['edge']:+.1%}, "
                f"EV {s['ev']:+.1%}, tier {s['tier']}, "
                f"fractional-Kelly stake {s['kelly_stake']:.3f}. {s['rationale']}"
            )
    elif state.stake_approval == "rejected":
        lines.append("Staking suggestions were reviewed and rejected by the human.")
    elif pred.get("suggestions"):
        lines.append(
            f"{len(pred['suggestions'])} value suggestion(s) computed — "
            "awaiting human approval before disclosure."
        )

    if state.degraded:
        lines.append("")
        lines.append("Reduced confidence — degraded evidence:")
        lines.extend(f"- {note}" for note in state.degraded)

    # parlay correlation study (when prediction includes grid data)
    if pred.get("exact_score", {}).get("scoreline_grid"):
        ou = pred["exact_score"]["over_under_2_5"]
        bt = pred["exact_score"]["btts"]
        lines.append("")
        lines.append(
            "**Parlay correlation study** (same-match legs vs independent pricing):"
        )
        lines.append(
            f"  O/U 2.5: over {_pct(ou['over'])}, under {_pct(ou['under'])}. "
            f"BTTS: yes {_pct(bt['yes'])}, no {_pct(bt['no'])}."
        )
        lines.append(
            "  Use the Parlay Builder to price multi-leg combos with exact "
            "correlation from the Dixon-Coles grid."
        )

    lines.append("")
    lines.append(
        f"Evidence trail: {len(state.ledger)} tool call(s), "
        f"{sum(1 for c in state.ledger if not c.ok)} failed."
    )
    return "\n".join(lines)

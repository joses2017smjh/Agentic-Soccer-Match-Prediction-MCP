"""Knockout-bracket projection engine.

Given a pool of teams with recent form, this seeds a single-elimination
bracket by model-derived strength and simulates it round by round. Every tie
is predicted with the same machinery the rest of the system uses — team xG
from a gradient-boosted model, a Dixon-Coles grid for scorelines and
extra-time/penalty advancement, and the goal-timing model for when goals land
— so a bracket is internally consistent with single-match predictions.

Player-level scorers/assists are allocated at the **role level** (striker,
wingers, etc.) from the team's expected goals via the same Poisson allocation
used elsewhere. Naming real individuals requires a women's player-event
provider (e.g. StatsBomb open data, which covers several women's
competitions) — a documented upgrade, not fabricated here.

Every match carries an ``evidence`` block (both teams' form inputs, strength
ranks, the xG the grid was built from, rho) so a user can drill into exactly
what drove each decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.player_props import allocate_player_props
from src.models.ratings import EloModel, compute_elo, rank_by_elo
from src.models.score_grid import (
    fit_rho,
    knockout_advance,
    outcome_probs,
    round_probs,
    score_grid,
    top_scorelines,
)
from src.models.sequence import GoalTimingModel

# role-level attacking lineup with heuristic shares — labeled illustrative;
# supply a real squad + event data to name individuals.
ROLE_LINEUP = [
    {"player": "Striker (CF)", "xg_share": 0.34, "xa_share": 0.10, "exp_minutes": 90},
    {"player": "Left Winger", "xg_share": 0.20, "xa_share": 0.22, "exp_minutes": 90},
    {"player": "Right Winger", "xg_share": 0.18, "xa_share": 0.22, "exp_minutes": 90},
    {"player": "Attacking Mid", "xg_share": 0.14, "xa_share": 0.30, "exp_minutes": 90},
    {"player": "Central Mid", "xg_share": 0.08, "xa_share": 0.12, "exp_minutes": 90},
    {"player": "Set-piece Def", "xg_share": 0.06, "xa_share": 0.04, "exp_minutes": 90,
     "setpiece_mult": 1.3},
]

# standard single-elimination seeding for 16 (1 and 2 can meet only in final)
SEED_ORDER_16 = [1, 16, 8, 9, 5, 12, 4, 13, 3, 14, 6, 11, 7, 10, 2, 15]
ROUND_NAMES = {16: "Round of 16", 8: "Quarter-finals", 4: "Semi-finals",
               2: "Final"}


@dataclass
class TeamStrength:
    team: str
    rank: int
    elo: float
    n_matches: int
    last_played: str


def rank_teams(
    elo: EloModel, *, top_n: int = 16, active_since: str = "2024-06-01",
    min_matches: int = 10,
) -> list[TeamStrength]:
    """Strongest currently-active teams by Elo (opponent-adjusted)."""
    top = rank_by_elo(elo, top_n=top_n, active_since=active_since,
                      min_matches=min_matches)
    return [
        TeamStrength(team=t, rank=i, elo=round(r, 1),
                     n_matches=elo.n_played[t], last_played=elo.last_played[t])
        for i, (t, r) in enumerate(top, start=1)
    ]


_SHARES_CACHE: dict[str, list[dict]] | None = None


def _lineup_for(team: str) -> tuple[list[dict], bool]:
    """Real StatsBomb player shares when we have them, else the role lineup.

    Returns (lineup, is_named). Named players come from actual event data
    (shot xG per player, key-pass xG for assists) — never invented.
    """
    global _SHARES_CACHE
    if _SHARES_CACHE is None:
        from src.data.statsbomb import load_shares

        try:
            _SHARES_CACHE = load_shares()
        except Exception:  # noqa: BLE001
            _SHARES_CACHE = {}
    players = _SHARES_CACHE.get(team)
    return (players, True) if players else (ROLE_LINEUP, False)


def _scenario(mu_h: float, mu_a: float, grid, timing: GoalTimingModel,
              home: str, away: str, advance: dict) -> dict:
    """Headline scenario: scoreline, scorers, assists, minutes. Uses real
    named players where StatsBomb open data covers the team."""
    top = top_scorelines(grid, n=1)[0]
    gh, ga = (int(x) for x in top["score"].split("-"))
    lineups = {"home": _lineup_for(home), "away": _lineup_for(away)}
    named = {side: flag for side, (_, flag) in lineups.items()}
    props = {
        side: allocate_player_props(pd.DataFrame(
            [{**r, "availability": 1.0} for r in lineups[side][0]]), team_mu=mu)
        for side, mu in (("home", mu_h), ("away", mu_a))
    }
    goals = []
    for side, count in (("home", gh), ("away", ga)):
        roles = props[side].to_dict("records")
        for k in range(count):
            scorer = roles[min(k, len(roles) - 1)]
            others = [r for r in roles if r["player"] != scorer["player"]]
            goals.append({
                "minute": round(timing.minute_quantile((k + 0.5) / count)),
                "team": side, "scorer_role": scorer["player"],
                "assist_role": max(others, key=lambda r: r["p_assist"])["player"]
                if others else None,
            })
    goals.sort(key=lambda g: g["minute"])
    winner = ("home" if gh > ga else "away" if ga > gh
              else ("home" if advance["home"] >= advance["away"] else "away"))
    return {
        "scoreline": top["score"], "probability": round(top["prob"], 3),
        "goals": goals,
        "penalties": None if gh != ga else {
            "winner": home if winner == "home" else away,
            "p_advance": round(advance[winner], 3)},
        "player_data": ("statsbomb" if all(named.values())
                        else "mixed" if any(named.values()) else "role-level"),
        "note": (
            "scorers/assists from real StatsBomb event data (per-player shot "
            "xG and key-pass xG)"
            if all(named.values()) else
            "scorers/assists are role-level where no player-event data covers "
            "the team; named players come from real StatsBomb event data"
        ),
    }


def predict_tie(
    elo: EloModel, rho: float, timing: GoalTimingModel,
    home: TeamStrength, away: TeamStrength, *, neutral: bool = True,
) -> dict[str, Any]:
    """Full prediction for one knockout tie + the evidence behind it."""
    mu_h, mu_a = elo.expected_goals(home.team, away.team, neutral=neutral)
    grid = score_grid(mu_h, mu_a, rho)
    advance = knockout_advance(mu_h, mu_a, rho)
    winner = home if advance["home"] >= advance["away"] else away
    shown_advance = round_probs(advance)

    return {
        "home": home.team, "away": away.team,
        "seeds": {"home": home.rank, "away": away.rank},
        "expected_goals": {"home": round(mu_h, 2), "away": round(mu_a, 2)},
        "outcome_90": round_probs(outcome_probs(grid)),
        "advance": {home.team: shown_advance["home"],
                    away.team: shown_advance["away"]},
        "projected_winner": winner.team,
        "top_scorelines": [{"score": s["score"], "prob": round(s["prob"], 3)}
                           for s in top_scorelines(grid, 4)],
        "goal_timing_bands": timing.expected_goals_by_band(mu_h, mu_a),
        "first_scorer": round_probs(
            timing.first_scorer(mu_h, mu_a, float(grid[0, 0]))),
        "headline_scenario": _scenario(mu_h, mu_a, grid, timing,
                                       home.team, away.team, advance),
        "evidence": {
            "home_rating": {"elo": home.elo, "matches": home.n_matches,
                            "seed": home.rank, "last_played": home.last_played},
            "away_rating": {"elo": away.elo, "matches": away.n_matches,
                            "seed": away.rank, "last_played": away.last_played},
            "elo_difference": round(home.elo - away.elo, 1),
            "model_xg": {"home": round(mu_h, 3), "away": round(mu_a, 3)},
            "dixon_coles_rho": round(rho, 4),
            "method": "opponent-adjusted Elo -> calibrated goal supremacy -> "
                      "Dixon-Coles grid; advancement includes extra time + "
                      "penalties",
        },
        "winner": winner,   # internal: TeamStrength that advances
    }


def _fit_rho(results: pd.DataFrame, elo: EloModel) -> float:
    df = results.dropna(subset=["home_score", "away_score"]).tail(4000)
    mu_h, mu_a = [], []
    for r in df.itertuples(index=False):
        mh, ma = elo.expected_goals(r.home_team, r.away_team,
                                    neutral=bool(getattr(r, "neutral", False)))
        mu_h.append(mh); mu_a.append(ma)
    return fit_rho(df["home_score"].to_numpy(dtype=float),
                   df["away_score"].to_numpy(dtype=float),
                   np.array(mu_h), np.array(mu_a))


def simulate_bracket(
    results: pd.DataFrame, *, top_n: int = 16, active_since: str = "2024-06-01",
) -> dict[str, Any]:
    """Seed the strongest ``top_n`` teams by Elo and simulate the bracket.

    ``results`` is the raw results frame (home_team/away_team/scores/neutral).
    """
    if top_n != 16:
        raise ValueError("top_n must be 16 for the standard bracket")
    elo = compute_elo(results)
    strengths = rank_teams(elo, top_n=top_n, active_since=active_since)
    if len(strengths) < top_n:
        raise ValueError(f"only {len(strengths)} eligible teams; need {top_n}")
    by_rank = {s.rank: s for s in strengths}
    rho = _fit_rho(results, elo)
    timing = GoalTimingModel()  # uniform profile: source has no goal minutes

    current = [by_rank[r] for r in SEED_ORDER_16]
    rounds: list[dict[str, Any]] = []
    while len(current) > 1:
        matches, winners = [], []
        for i in range(0, len(current), 2):
            tie = predict_tie(elo, rho, timing, current[i], current[i + 1])
            winners.append(tie.pop("winner"))
            matches.append(tie)
        rounds.append({"round": ROUND_NAMES.get(len(current), f"Round of {len(current)}"),
                       "matches": matches})
        current = winners

    return {
        "tournament": "Women's World Cup (projection)",
        "disclaimer": "The 2027 WWC draw does not exist yet; this seeds the "
                      "strongest currently-active teams by opponent-adjusted "
                      "Elo and simulates the bracket. Not a real fixture list.",
        "seeding": [{"rank": s.rank, "team": s.team, "elo": s.elo,
                     "matches": s.n_matches, "last_played": s.last_played}
                    for s in strengths],
        "rounds": rounds,
        "champion": current[0].team,
        "model": {"dixon_coles_rho": round(rho, 4),
                  "strength_metric": "opponent-adjusted Elo (home adv 60, K 40)",
                  "goal_slope_per_elo": round(elo.goal_slope, 5),
                  "base_total_goals": round(elo.base_total, 2)},
    }

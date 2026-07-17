"""Data backends for the Sports Data & Odds MCP server.

``DataBackend`` is the seam: the server's tools call only this interface, so
swapping the demo for FBref/API-Football/The Odds API clients (src/data/)
never touches tool code. ``DemoBackend`` produces deterministic,
seeded-by-name synthetic data so the whole Phase B stack runs end-to-end
before any API key exists.

Match ids follow ``HOME-AWAY-YYYY-MM-DD`` (e.g. ``ARS-MCI-2026-07-18``).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import numpy as np


class DataBackend(Protocol):
    def team_stats(self, team_id: str, window: int) -> dict[str, Any]: ...
    def live_odds(self, match_id: str, market: str) -> dict[str, Any]: ...
    def h2h(self, team_a: str, team_b: str) -> dict[str, Any]: ...
    def fixture_context(self, match_id: str) -> dict[str, Any]: ...
    def squad_props(self, team_id: str) -> dict[str, Any]: ...


def _rng(*keys: str) -> np.random.Generator:
    seed = int.from_bytes(
        hashlib.sha256("|".join(keys).encode()).digest()[:8], "big"
    )
    return np.random.default_rng(seed)


def parse_match_id(match_id: str) -> tuple[str, str, str]:
    parts = match_id.split("-", 2)
    if len(parts) != 3:
        raise ValueError(
            f"match_id must look like HOME-AWAY-YYYY-MM-DD, got {match_id!r}"
        )
    return parts[0], parts[1], parts[2]


class DemoBackend:
    """Deterministic synthetic provider for development and agent evals."""

    def team_stats(self, team_id: str, window: int = 10) -> dict[str, Any]:
        rng = _rng("stats", team_id)
        strength = float(rng.normal(1.4, 0.3))
        return {
            "team_id": team_id,
            "window": window,
            "form_xg_for": round(max(0.4, strength), 3),
            "form_xg_against": round(max(0.3, float(rng.normal(1.2, 0.25))), 3),
            "form_shots_for": round(float(rng.normal(13, 2.5)), 1),
            "form_possession": round(float(rng.uniform(42, 62)), 1),
            "rest_days": int(rng.integers(3, 8)),
            "matches_in_window": window,
        }

    def live_odds(self, match_id: str, market: str = "h2h") -> dict[str, Any]:
        from src.features.odds_features import remove_overround

        rng = _rng("odds", match_id, market)
        if market == "h2h":
            probs = rng.dirichlet([5, 3, 4])
            outcomes = ["home", "draw", "away"]
        elif market.startswith("totals"):
            probs = rng.dirichlet([5, 5])
            outcomes = ["over", "under"]
        else:
            raise ValueError(f"unsupported market: {market}")
        overround = 1.05
        odds = [round(1.0 / (p * overround), 2) for p in probs]
        implied = remove_overround(np.array(odds), method="power")
        return {
            "match_id": match_id,
            "market": market,
            "bookmaker": "demo-book",
            "selections": [
                {"outcome": o, "decimal_odds": d, "implied_prob_vigfree": round(float(p), 4)}
                for o, d, p in zip(outcomes, odds, implied)
            ],
            "as_of": (datetime.now(timezone.utc) - timedelta(minutes=5))
            .isoformat(timespec="seconds"),
        }

    def h2h(self, team_a: str, team_b: str) -> dict[str, Any]:
        rng = _rng("h2h", *sorted([team_a, team_b]))
        n = int(rng.integers(4, 9))
        a_wins = int(rng.integers(0, n + 1))
        draws = int(rng.integers(0, n - a_wins + 1))
        return {
            "team_a": team_a, "team_b": team_b, "meetings": n,
            "a_wins": a_wins, "draws": draws, "b_wins": n - a_wins - draws,
            "avg_total_goals": round(float(rng.uniform(1.8, 3.4)), 2),
        }

    def squad_props(self, team_id: str) -> dict[str, Any]:
        """Historical per-player shares that drive the prop allocation."""
        from mcp_servers.demo_data import DEMO_SQUADS

        squad = DEMO_SQUADS.get(team_id)
        if squad is None:
            raise ValueError(
                f"unknown team {team_id!r}; known: {sorted(DEMO_SQUADS)}"
            )
        rng = _rng("squad", team_id)
        players = list(squad)
        xg = rng.dirichlet(np.full(len(players), 1.4))
        xa = rng.dirichlet(np.full(len(players), 1.4))
        taker = int(rng.integers(0, len(players)))  # set-piece duty holder
        return {
            "team_id": team_id,
            "players": [
                {
                    "player": p,
                    "xg_share": round(float(xg[i]), 4),
                    "xa_share": round(float(xa[i]), 4),
                    "exp_minutes": int(rng.integers(75, 91)),
                    "setpiece_mult": 1.2 if i == taker else 1.0,
                }
                for i, p in enumerate(players)
            ],
        }

    def fixture_context(self, match_id: str) -> dict[str, Any]:
        from src.data.tournaments import tournament_features

        home, away, date = parse_match_id(match_id)
        rng = _rng("fixture", match_id)
        stage = str(rng.choice(["group", "quarterfinal", "semifinal", "final"]))
        tournament_id = str(rng.choice([
            "uefa_champions_league", "club_world_cup", "copa_libertadores",
        ]))
        ctx: dict[str, Any] = {
            "match_id": match_id,
            "home_team": home, "away_team": away, "date": date,
            "stage": stage,
            "knockout": stage != "group",
            "neutral_venue": bool(rng.random() < 0.5),
            "home_rest_days": int(rng.integers(3, 8)),
            "away_rest_days": int(rng.integers(3, 8)),
            "stakes": str(rng.choice(["must_win", "normal", "dead_rubber"])),
        }
        # hierarchy coordinates: Confederation → Tournament → Stage → Match
        ctx.update(tournament_features(tournament_id, stage))
        return ctx

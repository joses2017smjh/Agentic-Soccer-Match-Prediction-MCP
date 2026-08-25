"""In-play Bayesian updating (C1).

Condition the pre-match Dixon-Coles grid on the observed match state
(elapsed minute, current score, red cards) to produce live win probabilities.

The key insight: at minute t with score (g_h, g_a), only scorelines where
home >= g_h and away >= g_a are reachable. The remaining goals follow a
scaled Poisson with rate proportional to remaining time. We build a
*remaining-goals* grid, shift it by the current score, and renormalize.

Red cards reduce the affected team's remaining expected goals by a
configurable factor (default 15% per card, from empirical estimates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


MATCH_MINUTES = 90.0
RED_CARD_FACTOR = 0.15


def remaining_grid(
    mu_home: float,
    mu_away: float,
    elapsed: float,
    goals_home: int = 0,
    goals_away: int = 0,
    red_home: int = 0,
    red_away: int = 0,
    rho: float = -0.1,
    max_goals: int = 10,
) -> np.ndarray:
    """Build a scoreline grid conditioned on the current match state.

    Returns a (max_goals+1, max_goals+1) grid where entry [i, j] is
    P(final score = i, j | current state). Only cells reachable from
    the current score have nonzero probability.
    """
    elapsed = min(max(elapsed, 0.0), MATCH_MINUTES)
    frac_remaining = max(1.0 - elapsed / MATCH_MINUTES, 0.01)

    mu_h_rem = mu_home * frac_remaining * (1.0 - RED_CARD_FACTOR * red_home)
    mu_a_rem = mu_away * frac_remaining * (1.0 - RED_CARD_FACTOR * red_away)
    mu_h_rem = max(mu_h_rem, 0.01)
    mu_a_rem = max(mu_a_rem, 0.01)

    size = max_goals + 1
    grid = np.zeros((size, size), dtype=float)

    max_extra_h = size - goals_home
    max_extra_a = size - goals_away
    if max_extra_h <= 0 or max_extra_a <= 0:
        grid[min(goals_home, size - 1), min(goals_away, size - 1)] = 1.0
        return grid

    extra_h = np.arange(max_extra_h)
    extra_a = np.arange(max_extra_a)
    ph = poisson.pmf(extra_h, mu_h_rem)
    pa = poisson.pmf(extra_a, mu_a_rem)

    # rho correction only applies to the remaining-goals (0,0) and (0,1) etc.
    # cells -- but for simplicity and numerical stability we apply a
    # first-order Dixon-Coles correction on the remaining goals
    from src.models.score_grid import _tau
    eh_grid, ea_grid = np.meshgrid(extra_h, extra_a, indexing="ij")
    tau = _tau(eh_grid, ea_grid, mu_h_rem, mu_a_rem, rho)

    outer = np.outer(ph, pa) * tau
    outer = np.clip(outer, 0.0, None)

    grid[goals_home:goals_home + max_extra_h,
         goals_away:goals_away + max_extra_a] = outer

    total = grid.sum()
    if total > 0:
        grid /= total
    return grid


def in_play_probs(
    mu_home: float,
    mu_away: float,
    elapsed: float,
    goals_home: int = 0,
    goals_away: int = 0,
    red_home: int = 0,
    red_away: int = 0,
    rho: float = -0.1,
) -> dict[str, float]:
    """1X2 probabilities conditioned on current match state."""
    grid = remaining_grid(
        mu_home, mu_away, elapsed, goals_home, goals_away,
        red_home, red_away, rho,
    )
    from src.models.score_grid import outcome_probs
    return outcome_probs(grid)


def win_probability_curve(
    mu_home: float,
    mu_away: float,
    events: list[dict],
    rho: float = -0.1,
    resolution: int = 91,
) -> list[dict]:
    """Generate a minute-by-minute win probability curve.

    Args:
        mu_home: pre-match expected goals for home
        mu_away: pre-match expected goals for away
        events: list of dicts with keys "minute", "type" ("goal_home",
                "goal_away", "red_home", "red_away")
        rho: Dixon-Coles dependency
        resolution: number of time points (default: 0-90 inclusive)

    Returns:
        list of dicts with "minute", "home", "draw", "away"
    """
    goals_home = 0
    goals_away = 0
    red_home = 0
    red_away = 0

    sorted_events = sorted(events, key=lambda e: e["minute"])
    event_idx = 0

    curve = []
    for minute in np.linspace(0, MATCH_MINUTES, resolution):
        while event_idx < len(sorted_events) and sorted_events[event_idx]["minute"] <= minute:
            ev = sorted_events[event_idx]
            if ev["type"] == "goal_home":
                goals_home += 1
            elif ev["type"] == "goal_away":
                goals_away += 1
            elif ev["type"] == "red_home":
                red_home += 1
            elif ev["type"] == "red_away":
                red_away += 1
            event_idx += 1

        probs = in_play_probs(
            mu_home, mu_away, minute,
            goals_home, goals_away,
            red_home, red_away, rho,
        )
        curve.append({
            "minute": round(float(minute), 1),
            "home": probs["home"],
            "draw": probs["draw"],
            "away": probs["away"],
        })

    return curve


@dataclass
class InPlayState:
    """Mutable match state tracker for streaming updates."""

    mu_home: float
    mu_away: float
    rho: float = -0.1
    elapsed: float = 0.0
    goals_home: int = 0
    goals_away: int = 0
    red_home: int = 0
    red_away: int = 0

    def update(self, event: dict) -> None:
        """Process a single match event."""
        self.elapsed = event.get("minute", self.elapsed)
        etype = event.get("type", "")
        if etype == "goal_home":
            self.goals_home += 1
        elif etype == "goal_away":
            self.goals_away += 1
        elif etype == "red_home":
            self.red_home += 1
        elif etype == "red_away":
            self.red_away += 1

    def probs(self) -> dict[str, float]:
        return in_play_probs(
            self.mu_home, self.mu_away, self.elapsed,
            self.goals_home, self.goals_away,
            self.red_home, self.red_away, self.rho,
        )

    def snapshot(self) -> dict:
        p = self.probs()
        return {
            "minute": self.elapsed,
            "score": f"{self.goals_home}-{self.goals_away}",
            "red_cards": {"home": self.red_home, "away": self.red_away},
            "probs": p,
        }

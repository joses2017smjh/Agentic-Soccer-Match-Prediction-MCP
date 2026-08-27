"""Bayesian dynamic team strength model (B5).

A discrete-time dynamic linear model for team attack/defense strengths,
inspired by the Bayesian weighted DLM framework (JRSS-C, 2026). Each team
has latent attack and defense abilities that evolve over time via a
random walk with diagonal inflation (system variance grows between
observations to reflect genuine form changes).

Key design decisions:
  - Diagonal inflation: the system variance W_t = delta * C_{t-1} where
    delta > 1 controls how fast old information decays. This is the
    specific mechanism that improves draw calibration -- broader
    posterior uncertainty widens the outcome distribution.
  - The model produces (mu_home, mu_away) expected goals from the
    current strength estimates, which feed into Dixon-Coles for a
    coherent scoreline grid.
  - No PyTorch: uses closed-form Kalman filter updates only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeamState:
    """Latent team abilities with uncertainty."""

    attack: float = 0.0
    defense: float = 0.0
    attack_var: float = 1.0
    defense_var: float = 1.0


@dataclass
class BayesianDynamic:
    """Bayesian dynamic team strength estimator.

    Parameters:
        home_advantage: additive home attack boost (log-goals scale)
        delta: diagonal inflation factor (> 1 widens uncertainty over time)
        obs_var: observation noise variance
        base_rate: league-average log goals per team per match
    """

    home_advantage: float = 0.25
    delta: float = 1.05
    obs_var: float = 0.5
    base_rate: float = 0.3
    _teams: dict[str, TeamState] = field(default_factory=dict)

    def _get_or_init(self, team: str) -> TeamState:
        if team not in self._teams:
            self._teams[team] = TeamState()
        return self._teams[team]

    def predict_xg(
        self,
        home: str,
        away: str,
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Expected goals for each team given current latent states."""
        h = self._get_or_init(home)
        a = self._get_or_init(away)
        adv = 0.0 if neutral else self.home_advantage
        log_mu_h = self.base_rate + h.attack - a.defense + adv
        log_mu_a = self.base_rate + a.attack - h.defense
        return max(np.exp(log_mu_h), 0.1), max(np.exp(log_mu_a), 0.1)

    def predict_probs(
        self,
        home: str,
        away: str,
        neutral: bool = False,
        rho: float = -0.1,
    ) -> dict[str, float]:
        """1X2 probabilities from the dynamic strengths via Dixon-Coles."""
        from src.models.score_grid import outcome_probs, score_grid
        mu_h, mu_a = self.predict_xg(home, away, neutral)
        grid = score_grid(mu_h, mu_a, rho)
        return outcome_probs(grid)

    def uncertainty(self, team: str) -> dict[str, float]:
        """Current posterior uncertainty for a team."""
        s = self._get_or_init(team)
        return {
            "attack_var": s.attack_var,
            "defense_var": s.defense_var,
            "total_var": s.attack_var + s.defense_var,
        }

    def inflate(self) -> None:
        """Apply diagonal inflation: widen all posterior variances.

        Called between matchdays to reflect genuine time-varying ability.
        """
        for s in self._teams.values():
            s.attack_var *= self.delta
            s.defense_var *= self.delta

    def update(
        self,
        home: str,
        away: str,
        goals_home: int,
        goals_away: int,
        neutral: bool = False,
    ) -> None:
        """Kalman-style update after observing a result.

        Observation model: goals_team ~ Poisson(exp(base + atk_team - def_opp + adv))
        We linearize around the current mean (extended Kalman style) and
        update attack/defense for both teams.
        """
        h = self._get_or_init(home)
        a = self._get_or_init(away)

        mu_h, mu_a = self.predict_xg(home, away, neutral)

        # home goals observation -> update home attack, away defense
        resid_h = goals_home - mu_h
        h_h_atk = mu_h  # d(mu_h)/d(atk_home) = mu_h
        h_a_def = -mu_h  # d(mu_h)/d(def_away) = -mu_h

        S_h = h_h_atk**2 * h.attack_var + h_a_def**2 * a.defense_var + self.obs_var
        if S_h > 1e-10:
            K_h_atk = h.attack_var * h_h_atk / S_h
            K_a_def = a.defense_var * h_a_def / S_h
            h.attack += K_h_atk * resid_h
            a.defense += K_a_def * resid_h
            h.attack_var *= (1.0 - K_h_atk * h_h_atk)
            a.defense_var *= (1.0 - K_a_def * (-h_a_def))

        # away goals observation -> update away attack, home defense
        resid_a = goals_away - mu_a
        h_a_atk = mu_a
        h_h_def = -mu_a

        S_a = h_a_atk**2 * a.attack_var + h_h_def**2 * h.defense_var + self.obs_var
        if S_a > 1e-10:
            K_a_atk = a.attack_var * h_a_atk / S_a
            K_h_def = h.defense_var * h_h_def / S_a
            a.attack += K_a_atk * resid_a
            h.defense += K_h_def * resid_a
            a.attack_var *= (1.0 - K_a_atk * h_a_atk)
            h.defense_var *= (1.0 - K_h_def * (-h_h_def))

        # clamp variances to avoid negative values from linearization
        h.attack_var = max(h.attack_var, 0.01)
        h.defense_var = max(h.defense_var, 0.01)
        a.attack_var = max(a.attack_var, 0.01)
        a.defense_var = max(a.defense_var, 0.01)

    def fit(
        self,
        results: "pd.DataFrame",
    ) -> "BayesianDynamic":
        """Fit from a chronologically sorted results frame.

        Expected columns: date, home_team, away_team, home_score,
        away_score, neutral (optional).
        """
        import pandas as pd
        results = results.sort_values("date")
        prev_date = ""
        for r in results.itertuples(index=False):
            d = str(r.date)
            if d != prev_date and prev_date != "":
                self.inflate()
            prev_date = d
            self.update(
                r.home_team, r.away_team,
                int(r.home_score), int(r.away_score),
                neutral=bool(getattr(r, "neutral", False)),
            )
        return self

    def team_rankings(self, top_n: int | None = None) -> list[dict]:
        """Rank teams by attack - defense (net strength)."""
        rankings = []
        for team, s in self._teams.items():
            rankings.append({
                "team": team,
                "attack": round(s.attack, 4),
                "defense": round(s.defense, 4),
                "net": round(s.attack - s.defense, 4),
                "uncertainty": round(s.attack_var + s.defense_var, 4),
            })
        rankings.sort(key=lambda x: -x["net"])
        if top_n is not None:
            return rankings[:top_n]
        return rankings

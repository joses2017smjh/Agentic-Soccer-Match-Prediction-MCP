"""Sequential model: goal ordering and timing as an inhomogeneous Poisson
process with a learned time profile and score-state effects.

Architecture choice (HMM family, not LSTM)
------------------------------------------
The spec allows LSTM/Transformer or a Markov-family model. We use a
piecewise-constant-intensity Poisson process — the continuous-time cousin of
an HMM whose hidden state is the current scoreline — because:

1. *Reconciliation is exact.* Team i's intensity is
   λ_i(t) = mu_i · m(band(t)) with Σ_b m_b · width_b = 90, so the process
   integrates to exactly the Dixon–Coles mean mu_i. Marginal goal counts
   therefore agree with the score grid by construction, not by post-hoc
   patching. A neural sequence model would need explicit re-projection onto
   the grid's marginals every call.
2. *Data volume.* Public event data covers thousands, not millions, of
   matches; a 6-parameter time profile + 2 state multipliers is the right
   capacity, and it trains deterministically in milliseconds.
3. The training loop below (`fit`) is maximum likelihood via band counts —
   swap-in of a neural intensity model later only has to implement the same
   two methods (`fit`, `intensity_multiplier`).

Remaining discrepancy and how it is resolved
--------------------------------------------
Dixon–Coles' τ makes low scores slightly dependent, while a two-stream
Poisson process is independent, so P(no goals) differs by O(rho·mu²).
``first_scorer`` therefore takes the grid's P(0,0) as ground truth and
scales the home-first/away-first split (whose *ratio* comes from the timing
model) to fill exactly 1 − P(0,0). Grid quantities always win; the sequence
model only ever distributes what the grid says exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# 15-minute bands; stoppage time folds into the 6th band (75-90+).
BAND_EDGES: list[int] = [0, 15, 30, 45, 60, 75, 90]
N_BANDS = len(BAND_EDGES) - 1
BAND_LABELS: list[str] = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]


@dataclass
class GoalTimingModel:
    """Learned time profile m_b (mean 1.0) + score-state multipliers."""

    band_multipliers: np.ndarray = field(
        default_factory=lambda: np.ones(N_BANDS)
    )
    trailing_boost: float = 1.0   # intensity multiplier when a team trails
    leading_damp: float = 1.0     # ... and when it leads

    def fit(self, goal_events: pd.DataFrame, smoothing: float = 5.0) -> "GoalTimingModel":
        """MLE from historical goals.

        ``goal_events``: one row per goal with columns ``minute`` (stoppage
        recorded as 45+x → 45.x is fine; clipped into bands) and
        ``scorer_state`` in {"level", "trailing", "leading"} at the moment of
        the goal. Band multipliers are smoothed counts normalized to mean 1;
        state multipliers are relative goal rates vs the "level" state.
        """
        minutes = goal_events["minute"].clip(0, 89.999)
        counts = np.histogram(minutes, bins=BAND_EDGES)[0].astype(float) + smoothing
        rates = counts / counts.sum() * N_BANDS  # mean exactly 1.0
        self.band_multipliers = rates

        state = goal_events.get("scorer_state")
        if state is not None and (state == "level").any():
            level_rate = float((state == "level").mean())
            trail_rate = float((state == "trailing").mean())
            lead_rate = float((state == "leading").mean())
            # normalize by exposure-free proxy: relative shares vs level
            self.trailing_boost = max(trail_rate / level_rate, 0.1) if level_rate else 1.0
            self.leading_damp = max(lead_rate / level_rate, 0.1) if level_rate else 1.0
        return self

    # ---------------------------------------------------------------- serving

    def intensity_multiplier(self, minute: float, state: str = "level") -> float:
        band = int(np.clip(np.searchsorted(BAND_EDGES, minute, side="right") - 1,
                           0, N_BANDS - 1))
        m = float(self.band_multipliers[band])
        if state == "trailing":
            m *= self.trailing_boost
        elif state == "leading":
            m *= self.leading_damp
        return m

    def expected_goals_by_band(
        self, mu_home: float, mu_away: float
    ) -> list[dict]:
        """Expected goals per 15-minute band; sums to mu_home/mu_away exactly."""
        widths = np.diff(BAND_EDGES) / 90.0
        share = self.band_multipliers * widths  # sums to 1.0 by construction
        share = share / share.sum()
        return [
            {"band": lbl, "home": float(mu_home * s), "away": float(mu_away * s)}
            for lbl, s in zip(BAND_LABELS, share)
        ]

    def first_scorer(
        self, mu_home: float, mu_away: float, p_zero_zero: float
    ) -> dict[str, float]:
        """P(home first) / P(away first) / P(no goals).

        For two independent Poisson streams sharing one time profile, the
        first goal is home's with probability mu_h/(mu_h+mu_a) regardless of
        the profile. ``p_zero_zero`` comes from the Dixon–Coles grid and is
        authoritative; the split fills the remainder.
        """
        total = mu_home + mu_away
        if total <= 0:
            return {"home_first": 0.0, "away_first": 0.0, "no_goals": 1.0}
        remainder = 1.0 - p_zero_zero
        return {
            "home_first": remainder * mu_home / total,
            "away_first": remainder * mu_away / total,
            "no_goals": p_zero_zero,
        }

    def next_goal(
        self, mu_home: float, mu_away: float, *,
        minute: float, score_home: int, score_away: int,
    ) -> dict[str, float]:
        """P(next goal is home / away / no more goals) from the current state.

        Remaining intensity = mu_i · (profile mass left after `minute`),
        adjusted by the score-state multipliers.
        """
        widths = np.diff(BAND_EDGES).astype(float)
        remaining = np.clip(np.array(BAND_EDGES[1:]) - minute, 0.0, widths)
        frac_left = float((self.band_multipliers * remaining).sum() / 90.0)

        def _state(for_home: bool) -> str:
            diff = score_home - score_away if for_home else score_away - score_home
            return "trailing" if diff < 0 else ("leading" if diff > 0 else "level")

        mult = {"level": 1.0, "trailing": self.trailing_boost,
                "leading": self.leading_damp}
        lam_h = mu_home * frac_left * mult[_state(True)]
        lam_a = mu_away * frac_left * mult[_state(False)]
        total = lam_h + lam_a
        p_none = float(np.exp(-total))
        if total <= 0:
            return {"home": 0.0, "away": 0.0, "no_more_goals": 1.0}
        return {
            "home": (1.0 - p_none) * lam_h / total,
            "away": (1.0 - p_none) * lam_a / total,
            "no_more_goals": p_none,
        }

    # ------------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "timing.json").write_text(json.dumps({
            "band_multipliers": self.band_multipliers.tolist(),
            "trailing_boost": self.trailing_boost,
            "leading_damp": self.leading_damp,
        }))

    @classmethod
    def load(cls, path: Path) -> "GoalTimingModel":
        d = json.loads((path / "timing.json").read_text())
        return cls(
            band_multipliers=np.array(d["band_multipliers"]),
            trailing_boost=d["trailing_boost"],
            leading_damp=d["leading_damp"],
        )

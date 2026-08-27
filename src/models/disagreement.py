"""Model-vs-market disagreement leaderboard (C5).

Rank fixtures by |model - market| and track how they resolve. The board
surfaces the predictions where the model most disagrees with the market,
which are the highest-information bets: if the model is right, these are
where CLV lives; if the model is wrong, these are where calibration
breaks down.

Resolution tracking: after the match, record whether the model or market
was closer to the outcome, building a running accuracy ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


OUTCOME_MAP = {"home": 0, "draw": 1, "away": 2}


@dataclass
class DisagreementEntry:
    """One fixture's model-vs-market disagreement."""

    match_id: str
    home_team: str
    away_team: str
    model_probs: dict[str, float]
    market_probs: dict[str, float]
    max_diff: float
    max_diff_outcome: str
    model_edge: float  # signed: positive = model sees value
    resolved: bool = False
    actual_outcome: str | None = None
    model_correct: bool | None = None
    market_correct: bool | None = None
    model_log_loss: float | None = None
    market_log_loss: float | None = None


@dataclass
class DisagreementBoard:
    """Track and rank model-vs-market disagreements."""

    entries: list[DisagreementEntry] = field(default_factory=list)
    _resolved_count: int = 0
    _model_wins: int = 0
    _market_wins: int = 0

    def add(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        model_probs: dict[str, float],
        market_probs: dict[str, float],
    ) -> DisagreementEntry:
        """Register a fixture's disagreement."""
        diffs = {
            k: model_probs.get(k, 0) - market_probs.get(k, 0)
            for k in ("home", "draw", "away")
        }
        abs_diffs = {k: abs(v) for k, v in diffs.items()}
        max_key = max(abs_diffs, key=abs_diffs.get)

        entry = DisagreementEntry(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            model_probs=dict(model_probs),
            market_probs=dict(market_probs),
            max_diff=abs_diffs[max_key],
            max_diff_outcome=max_key,
            model_edge=diffs[max_key],
        )
        self.entries.append(entry)
        return entry

    def resolve(
        self,
        match_id: str,
        actual_outcome: str,
    ) -> DisagreementEntry | None:
        """Record the actual outcome and score both forecasters."""
        entry = next((e for e in self.entries if e.match_id == match_id), None)
        if entry is None or entry.resolved:
            return entry

        entry.resolved = True
        entry.actual_outcome = actual_outcome
        self._resolved_count += 1

        outcome_idx = OUTCOME_MAP.get(actual_outcome, 0)
        model_p = max(entry.model_probs.get(actual_outcome, 0), 1e-12)
        market_p = max(entry.market_probs.get(actual_outcome, 0), 1e-12)

        entry.model_log_loss = -float(np.log(model_p))
        entry.market_log_loss = -float(np.log(market_p))

        entry.model_correct = entry.model_log_loss < entry.market_log_loss
        entry.market_correct = entry.market_log_loss < entry.model_log_loss

        if entry.model_correct:
            self._model_wins += 1
        elif entry.market_correct:
            self._market_wins += 1

        return entry

    def leaderboard(self, top_n: int | None = None) -> list[dict]:
        """Rank unresolved fixtures by disagreement magnitude."""
        unresolved = [e for e in self.entries if not e.resolved]
        ranked = sorted(unresolved, key=lambda e: -e.max_diff)
        if top_n is not None:
            ranked = ranked[:top_n]
        return [
            {
                "match_id": e.match_id,
                "fixture": f"{e.home_team} vs {e.away_team}",
                "max_diff": round(e.max_diff, 4),
                "outcome": e.max_diff_outcome,
                "model_edge": round(e.model_edge, 4),
                "model": {k: round(v, 4) for k, v in e.model_probs.items()},
                "market": {k: round(v, 4) for k, v in e.market_probs.items()},
            }
            for e in ranked
        ]

    def resolution_summary(self) -> dict:
        """Summary of resolved disagreements."""
        resolved = [e for e in self.entries if e.resolved]
        if not resolved:
            return {
                "total": 0,
                "model_wins": 0,
                "market_wins": 0,
                "ties": 0,
                "model_win_rate": 0.0,
                "avg_model_ll": 0.0,
                "avg_market_ll": 0.0,
            }

        ties = sum(
            1 for e in resolved
            if not e.model_correct and not e.market_correct
        )
        model_lls = [e.model_log_loss for e in resolved if e.model_log_loss is not None]
        market_lls = [e.market_log_loss for e in resolved if e.market_log_loss is not None]

        return {
            "total": len(resolved),
            "model_wins": self._model_wins,
            "market_wins": self._market_wins,
            "ties": ties,
            "model_win_rate": self._model_wins / max(len(resolved), 1),
            "avg_model_ll": float(np.mean(model_lls)) if model_lls else 0.0,
            "avg_market_ll": float(np.mean(market_lls)) if market_lls else 0.0,
        }

    def biggest_misses(self, top_n: int = 5) -> list[dict]:
        """Resolved entries where the model was most wrong vs market."""
        resolved = [e for e in self.entries if e.resolved and e.market_correct]
        ranked = sorted(
            resolved,
            key=lambda e: (e.model_log_loss or 0) - (e.market_log_loss or 0),
            reverse=True,
        )
        return [
            {
                "match_id": e.match_id,
                "fixture": f"{e.home_team} vs {e.away_team}",
                "actual": e.actual_outcome,
                "model_ll": round(e.model_log_loss, 4) if e.model_log_loss else None,
                "market_ll": round(e.market_log_loss, 4) if e.market_log_loss else None,
                "ll_gap": round((e.model_log_loss or 0) - (e.market_log_loss or 0), 4),
            }
            for e in ranked[:top_n]
        ]

    def biggest_wins(self, top_n: int = 5) -> list[dict]:
        """Resolved entries where the model beat the market most."""
        resolved = [e for e in self.entries if e.resolved and e.model_correct]
        ranked = sorted(
            resolved,
            key=lambda e: (e.market_log_loss or 0) - (e.model_log_loss or 0),
            reverse=True,
        )
        return [
            {
                "match_id": e.match_id,
                "fixture": f"{e.home_team} vs {e.away_team}",
                "actual": e.actual_outcome,
                "model_ll": round(e.model_log_loss, 4) if e.model_log_loss else None,
                "market_ll": round(e.market_log_loss, 4) if e.market_log_loss else None,
                "ll_gap": round((e.market_log_loss or 0) - (e.model_log_loss or 0), 4),
            }
            for e in ranked[:top_n]
        ]

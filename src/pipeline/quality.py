"""Quality gates: count what is dropped, and fail when too much is.

The loaders this pipeline reuses are forgiving by design -- the CSV reader
skips malformed lines, the canonical-frame builder drops rows with missing
results or unparseable dates. Forgiving is fine; *silent* is not. Every drop
here lands in a RejectLog under a named reason, and a reject rate above the
configured limit raises instead of producing a quietly smaller dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

RAW_REQUIRED = ("HomeTeam", "AwayTeam", "FTHG", "FTAG")


class PipelineError(RuntimeError):
    """A stage refused to produce output (reject rate, missing input, ...)."""


@dataclass
class RejectLog:
    """Rows checked and rows rejected, by reason, for one stage."""

    checked: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str, count: int) -> None:
        if count:
            self.reasons[reason] = self.reasons.get(reason, 0) + int(count)

    @property
    def rejected(self) -> int:
        return sum(self.reasons.values())

    @property
    def rate(self) -> float:
        return self.rejected / self.checked if self.checked else 0.0

    def merge(self, other: "RejectLog") -> None:
        self.checked += other.checked
        for reason, count in other.reasons.items():
            self.add(reason, count)

    def enforce(self, stage: str, max_rate: float) -> None:
        if self.rate > max_rate:
            raise PipelineError(
                f"{stage}: rejected {self.rejected} of {self.checked} rows "
                f"({self.rate:.1%}), over the {max_rate:.1%} limit: {self.reasons}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"checked": self.checked, "rejected": self.rejected,
                "rate": round(self.rate, 6), "reasons": dict(sorted(self.reasons.items()))}


def read_raw_csv(path: Path) -> tuple[pd.DataFrame, int]:
    """Read one raw season file; returns (frame, number of malformed lines).

    Same tolerance as the cached-download reader, but the skipped lines are
    counted rather than discarded without trace.
    """
    malformed: list[list[str]] = []

    def _count(line: list[str]) -> None:
        malformed.append(line)

    frame = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines=_count, engine="python")
    return frame, len(malformed)


def _season_window(start_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (pd.Timestamp(year=start_year, month=7, day=1, tz="UTC"),
            pd.Timestamp(year=start_year + 1, month=8, day=31, tz="UTC"))


def gate_team_matches(df: pd.DataFrame, start_year: int) -> tuple[pd.DataFrame, RejectLog]:
    """Reject whole matches that break an invariant of the TEAM_MATCH frame.

    A match is counted under the first reason it fails, so the reasons add up
    to the number of matches removed.
    """
    log = RejectLog(checked=int(df["match_id"].nunique()))
    bad: dict[str, str] = {}

    def flag(reason: str, match_ids: pd.Series) -> None:
        for mid in pd.unique(match_ids):
            bad.setdefault(mid, reason)

    flag("duplicate_match", df.loc[df.duplicated(["match_id", "team"], keep=False), "match_id"])
    sides = df.groupby("match_id")["is_home"].agg(["size", "sum"])
    flag("missing_side", sides.index[(sides["size"] != 2) | (sides["sum"] != 1)].to_series())
    goals = df[["goals_for", "goals_against"]]
    flag("invalid_goals", df.loc[((goals < 0) | (goals % 1 != 0)).any(axis=1), "match_id"])
    shots = df[["shots_for", "shots_against"]]
    flag("negative_shots", df.loc[(shots < 0).any(axis=1), "match_id"])
    lo, hi = _season_window(start_year)
    flag("kickoff_outside_season",
         df.loc[(df["kickoff_utc"] < lo) | (df["kickoff_utc"] > hi), "match_id"])

    for reason in bad.values():
        log.add(reason, 1)
    return df[~df["match_id"].isin(bad.keys())].reset_index(drop=True), log


def gate_odds(df: pd.DataFrame, valid_match_ids: set[str]) -> tuple[pd.DataFrame, RejectLog]:
    """Reject odds rows that are duplicated, unpriced, unnormalized or orphaned."""
    log = RejectLog(checked=len(df))
    if df.empty:
        return df, log
    keep = pd.Series(True, index=df.index)

    def reject(reason: str, mask: pd.Series) -> None:
        nonlocal keep
        hit = mask & keep
        log.add(reason, int(hit.sum()))
        keep = keep & ~hit

    reject("duplicate_match", df.duplicated("match_id", keep="first"))
    reject("orphan_match", ~df["match_id"].isin(valid_match_ids))
    prices = df[["odds_home", "odds_draw", "odds_away"]]
    reject("price_not_above_one", ~(prices > 1.0).all(axis=1))
    implied = df[["odds_imp_home", "odds_imp_draw", "odds_imp_away"]]
    reject("implied_not_normalized", (implied.sum(axis=1) - 1.0).abs() > 1e-6)
    return df[keep].reset_index(drop=True), log

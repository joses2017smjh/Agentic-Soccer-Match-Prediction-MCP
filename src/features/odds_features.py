"""Betting-odds features: vig removal and timestamp-aligned snapshot selection.

Odds play two roles: an anchor feature for the GBM, and the benchmark for the
suggestion layer. Both consume the output of ``odds_at_cutoff``, which selects,
per match/market/outcome, the last snapshot taken at or before the prediction
cutoff — never after.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.interfaces import ODDS_SNAPSHOT_COLUMNS, validate_frame
from src.features.leakage import assert_all_before, prediction_cutoff


def implied_probabilities(decimal_odds: np.ndarray) -> np.ndarray:
    """Raw implied probabilities 1/odds; sums to >1 by the overround."""
    odds = np.asarray(decimal_odds, dtype=float)
    if (odds <= 1.0).any():
        raise ValueError("decimal odds must be > 1.0")
    return 1.0 / odds


def remove_overround(decimal_odds: np.ndarray, method: str = "proportional") -> np.ndarray:
    """Strip the bookmaker margin from one market's odds; result sums to 1.

    proportional — divide each raw probability by their sum. Simple, standard.
    power — solve k so that sum((1/o_i)^k) = 1. Attributes more of the margin
        to longshots, correcting favourite–longshot bias; preferred for the
        value benchmark on markets with big price spreads.
    """
    raw = implied_probabilities(decimal_odds)
    if method == "proportional":
        return raw / raw.sum()
    if method == "power":
        lo, hi = 0.5, 3.0
        for _ in range(100):
            k = (lo + hi) / 2
            total = float((raw**k).sum())
            if abs(total - 1.0) < 1e-12:
                break
            # raw sums to >1, so larger k lowers the total
            lo, hi = (lo, k) if total < 1.0 else (k, hi)
        return raw**k / (raw**k).sum()
    raise ValueError(f"unknown overround method: {method}")


def odds_at_cutoff(
    snapshots: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    cutoff_minutes: int = 60,
) -> pd.DataFrame:
    """Last odds snapshot per (match, bookmaker, market, outcome) before cutoff.

    ``schedule`` needs match_id and kickoff_utc. Snapshots after a match's
    cutoff are dropped up front, and the survivors are re-asserted against the
    cutoff so no code path can hand later prices to the model.
    """
    snaps = validate_frame(snapshots, ODDS_SNAPSHOT_COLUMNS, "ODDS_SNAPSHOT")
    snaps = snaps.merge(schedule[["match_id", "kickoff_utc"]], on="match_id", how="inner")
    snaps["cutoff_utc"] = prediction_cutoff(snaps["kickoff_utc"], cutoff_minutes)

    usable = snaps[snaps["snapshot_time_utc"] <= snaps["cutoff_utc"]]
    latest = (
        usable.sort_values("snapshot_time_utc")
        .groupby(["match_id", "bookmaker", "market", "outcome"], as_index=False)
        .tail(1)
    )
    assert_all_before(latest, "snapshot_time_utc", "cutoff_utc", context="odds_at_cutoff")
    return latest.drop(columns=["cutoff_utc"]).reset_index(drop=True)


def consensus_market_probabilities(
    latest_odds: pd.DataFrame, *, method: str = "power"
) -> pd.DataFrame:
    """Median price across bookmakers, then de-vig within each match/market.

    Returns one row per (match_id, market, outcome) with columns
    ``consensus_odds`` and ``market_prob`` (vig-free, sums to 1 per market).
    """
    consensus = (
        latest_odds.groupby(["match_id", "market", "outcome"], as_index=False)
        .agg(consensus_odds=("decimal_odds", "median"))
    )

    def _devig(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group["market_prob"] = remove_overround(
            group["consensus_odds"].to_numpy(), method=method
        )
        return group

    return (
        consensus.groupby(["match_id", "market"], group_keys=False)
        .apply(_devig, include_groups=False)
        .join(consensus[["match_id", "market"]])
        .reset_index(drop=True)
    )


def h2h_odds_features(market_probs: pd.DataFrame) -> pd.DataFrame:
    """Wide per-match anchor features from the 1X2 market.

    Returns match_id, odds_imp_home, odds_imp_draw, odds_imp_away.
    """
    h2h = market_probs[market_probs["market"] == "h2h"]
    wide = h2h.pivot_table(
        index="match_id", columns="outcome", values="market_prob"
    ).rename(
        columns={"home": "odds_imp_home", "draw": "odds_imp_draw", "away": "odds_imp_away"}
    )
    wide.columns.name = None
    return wide.reset_index()

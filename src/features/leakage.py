"""Look-ahead leakage guards.

Every feature that joins time-stamped information (odds snapshots, news,
lineups, form stats) must go through these helpers rather than a raw merge.
The rule: for a match with kickoff K and configured cutoff C minutes, only
records whose availability timestamp is <= K - C may contribute, at train
and serve time alike.
"""

from __future__ import annotations

import pandas as pd


class LeakageError(RuntimeError):
    """Raised when a feature join would use information from after the cutoff."""


def prediction_cutoff(
    kickoff_utc: pd.Series | pd.Timestamp, minutes_before: int
) -> pd.Series | pd.Timestamp:
    """The latest instant whose information may be used for a prediction."""
    return kickoff_utc - pd.Timedelta(minutes=minutes_before)


def assert_all_before(
    df: pd.DataFrame, ts_col: str, cutoff_col: str, context: str = ""
) -> pd.DataFrame:
    """Hard-fail if any row's timestamp is after its cutoff. Returns df."""
    late = df[ts_col] > df[cutoff_col]
    if late.any():
        sample = df.loc[late, [ts_col, cutoff_col]].head(3)
        raise LeakageError(
            f"{int(late.sum())} rows use information after the cutoff"
            f"{f' in {context}' if context else ''}:\n{sample}"
        )
    return df


def merge_asof_guarded(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_cutoff: str,
    right_ts: str,
    by: str | list[str],
    context: str = "",
) -> pd.DataFrame:
    """As-of join: for each left row, the latest right row at or before its cutoff.

    Wraps ``pd.merge_asof(direction="backward")`` and then re-asserts that no
    joined timestamp exceeds the cutoff, so a refactor of the join can never
    silently introduce look-ahead.
    """
    left_sorted = left.sort_values(left_cutoff).reset_index(drop=True)
    right_sorted = right.sort_values(right_ts).reset_index(drop=True)
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=left_cutoff,
        right_on=right_ts,
        by=by,
        direction="backward",
        allow_exact_matches=True,
    )
    matched = merged[merged[right_ts].notna()]
    if not matched.empty:
        assert_all_before(matched, right_ts, left_cutoff, context=context)
    return merged

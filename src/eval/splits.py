"""Temporal splits and walk-forward folds.

A fold's training data always ends before its test data begins — asserted,
not assumed. Groups are typically seasons or tournaments, so the backtest
answers the question the system will actually face: "trained on everything
through last season, how do we do on the next one?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class Fold:
    group: str
    train_idx: pd.Index
    test_idx: pd.Index


def walk_forward_folds(
    df: pd.DataFrame,
    *,
    time_col: str = "kickoff_utc",
    group_col: str = "season",
    min_train_groups: int = 1,
) -> Iterator[Fold]:
    """Yield one fold per group (after the first ``min_train_groups``),
    training on all strictly earlier groups."""
    order = (
        df.groupby(group_col)[time_col].min().sort_values().index.tolist()
    )
    for i in range(min_train_groups, len(order)):
        train_groups, test_group = order[:i], order[i]
        train_idx = df.index[df[group_col].isin(train_groups)]
        test_idx = df.index[df[group_col] == test_group]

        latest_train = df.loc[train_idx, time_col].max()
        earliest_test = df.loc[test_idx, time_col].min()
        if latest_train >= earliest_test:
            raise ValueError(
                f"temporal overlap: train group(s) {train_groups} reach "
                f"{latest_train}, test group {test_group} starts {earliest_test}"
            )
        yield Fold(group=str(test_group), train_idx=train_idx, test_idx=test_idx)

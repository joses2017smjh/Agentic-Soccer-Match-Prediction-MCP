"""Tests for configurable episode shapes."""

from __future__ import annotations

import numpy as np
import pytest

from envs.episode import (
    EpisodeShape,
    SeasonResult,
    measure_instruction_adherence,
    run_shaped_episode,
)
from envs.real_market import load_season
from pathlib import Path
from training.evaluate import AbstainerPolicy, FavouritePolicy

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "football_data_uk"


@pytest.fixture(scope="module")
def fixtures():
    return load_season(_DATA_DIR / "E0_2425.csv")


def test_gameweek_shape(fixtures):
    """Gameweek shape must produce multiple episodes."""
    result = run_shaped_episode(fixtures, AbstainerPolicy(), EpisodeShape.GAMEWEEK, max_gameweeks=3)
    assert result.n_gameweeks == 3
    assert result.shape == EpisodeShape.GAMEWEEK


def test_single_match_shape(fixtures):
    """Single-match shape must produce one episode per fixture."""
    result = run_shaped_episode(fixtures[:5], AbstainerPolicy(), EpisodeShape.SINGLE_MATCH)
    assert result.n_gameweeks == 5


def test_season_shape_carries_bankroll(fixtures):
    """Season shape must carry bankroll across gameweeks."""
    result = run_shaped_episode(
        fixtures, FavouritePolicy(), EpisodeShape.SEASON,
        bankroll=1000.0, max_gameweeks=5,
    )
    assert result.n_gameweeks == 5
    assert result.total_bets > 0


def test_season_result_summary(fixtures):
    """Summary dict must contain all required keys."""
    result = run_shaped_episode(fixtures, AbstainerPolicy(), EpisodeShape.GAMEWEEK, max_gameweeks=2)
    summary = result.summary()
    required = {"shape", "n_gameweeks", "total_bets", "total_rejected", "total_pnl",
                "mean_clv", "mean_reward", "max_drawdown"}
    assert required.issubset(summary.keys())


def test_abstainer_zero_bets(fixtures):
    """Abstainer must place zero bets in all shapes."""
    for shape in EpisodeShape:
        result = run_shaped_episode(fixtures, AbstainerPolicy(), shape, max_gameweeks=2)
        assert result.total_bets == 0


def test_instruction_adherence_abstainer(fixtures):
    """Abstainer should have perfect adherence."""
    result = run_shaped_episode(fixtures, AbstainerPolicy(), EpisodeShape.GAMEWEEK, max_gameweeks=5)
    adherence = measure_instruction_adherence(result.episodes)
    assert all(a == 1.0 for a in adherence)


def test_instruction_adherence_length(fixtures):
    """Adherence list must have one entry per episode."""
    result = run_shaped_episode(fixtures, FavouritePolicy(), EpisodeShape.GAMEWEEK, max_gameweeks=5)
    adherence = measure_instruction_adherence(result.episodes)
    assert len(adherence) == result.n_gameweeks

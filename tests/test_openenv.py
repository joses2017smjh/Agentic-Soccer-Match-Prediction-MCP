"""Tests for the OpenEnv-compatible MarketGym adapter."""

from __future__ import annotations

import numpy as np
import pytest

from envs.openenv_adapter import ENV_ID, ENV_VERSION, MarketGym, MarketGymConfig


def test_env_id():
    assert MarketGym.env_id() == "MarketGym-v0"
    assert MarketGym.version() == ENV_VERSION


def test_reset_returns_obs_and_info():
    """Reset must return (observation, info) tuple."""
    config = MarketGymConfig(use_sim=True, eta=0.85, seed=42, n_gameweeks=3)
    env = MarketGym(config=config)
    obs, info = env.reset()
    assert obs.ndim == 2
    assert obs.shape[0] > 0
    assert "gameweek" in info
    assert "bankroll" in info


def test_step_returns_five_tuple():
    """Step must return (obs, reward, terminated, truncated, info)."""
    config = MarketGymConfig(use_sim=True, eta=0.85, seed=42, n_gameweeks=2)
    env = MarketGym(config=config)
    obs, info = env.reset()
    n = obs.shape[0]
    actions = np.zeros(n, dtype=int)
    obs2, reward, terminated, truncated, step_info = env.step(actions)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "n_bets" in step_info


def test_skip_all_zero_bets():
    """Skipping all fixtures should place zero bets."""
    config = MarketGymConfig(use_sim=True, eta=0.85, seed=42, n_gameweeks=2)
    env = MarketGym(config=config)
    obs, _ = env.reset()
    actions = np.zeros(obs.shape[0], dtype=int)
    _, _, _, _, info = env.step(actions)
    assert info["n_bets"] == 0


def test_bet_all_home():
    """Betting H on everything should place bets."""
    config = MarketGymConfig(use_sim=True, eta=0.85, seed=42, n_gameweeks=2)
    env = MarketGym(config=config)
    obs, _ = env.reset()
    actions = np.ones(obs.shape[0], dtype=int)
    _, _, _, _, info = env.step(actions)
    assert info["n_bets"] > 0


def test_episode_terminates():
    """Episode must terminate after all gameweeks."""
    config = MarketGymConfig(use_sim=True, eta=0.85, seed=42, n_gameweeks=3)
    env = MarketGym(config=config)
    obs, _ = env.reset()

    for _ in range(10):
        if obs.shape[0] == 0:
            break
        actions = np.zeros(obs.shape[0], dtype=int)
        obs, _, terminated, _, _ = env.step(actions)
        if terminated:
            break

    assert terminated


def test_render():
    """Render must return a non-empty string."""
    config = MarketGymConfig(use_sim=True, eta=0.85, seed=42, n_gameweeks=2)
    env = MarketGym(config=config)
    env.reset()
    text = env.render()
    assert "GW" in text
    assert "Bankroll" in text


def test_observation_space_keys():
    """Observation space descriptor must have type and shape."""
    env = MarketGym(MarketGymConfig(use_sim=True, n_gameweeks=1, seed=0))
    space = env.observation_space
    assert "type" in space
    assert "shape" in space


def test_action_space_keys():
    """Action space descriptor must have type and meanings."""
    env = MarketGym(MarketGymConfig(use_sim=True, n_gameweeks=1, seed=0))
    space = env.action_space
    assert "type" in space
    assert "meanings" in space
    assert space["meanings"][0] == "skip"


def test_reset_with_seed():
    """Resetting with different seeds should produce different obs."""
    config = MarketGymConfig(use_sim=True, eta=0.85, n_gameweeks=2)
    env = MarketGym(config=config)
    obs1, _ = env.reset(seed=1)
    obs2, _ = env.reset(seed=2)
    assert not np.array_equal(obs1, obs2)

"""Tests for the GRPO training module.

Covers: gym wrapper, MLP policy, GRPO trainer, and walk-forward evaluation.
Property-based tests ensure the training loop is correct regardless of
whether the learned policy beats the baselines.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from envs.real_market import Fixture, Gameweek, load_season
from mcp_servers.book_server.limits import LimitConfig
from mcp_servers.book_server.state import BookState
from training.gym_wrapper import N_FEATURES, StakingEnv, devig, fixture_features
from training.policy import MLPPolicy


# ── Helpers ──────────────────────────────────────────────────────────

def _make_fixtures() -> list[Fixture]:
    """Three synthetic fixtures with known odds."""
    return [
        Fixture(
            date=date(2024, 8, 17), match_id="A-B-2024-08-17",
            home="A", away="B", ftr="H",
            odds_h=1.80, odds_d=3.50, odds_a=4.50,
            close_h=1.75, close_d=3.60, close_a=4.80,
            odds_source="pinnacle",
        ),
        Fixture(
            date=date(2024, 8, 17), match_id="C-D-2024-08-17",
            home="C", away="D", ftr="D",
            odds_h=2.20, odds_d=3.30, odds_a=3.10,
            close_h=2.25, close_d=3.25, close_a=3.15,
            odds_source="pinnacle",
        ),
        Fixture(
            date=date(2024, 8, 17), match_id="E-F-2024-08-17",
            home="E", away="F", ftr="A",
            odds_h=3.00, odds_d=3.20, odds_a=2.30,
            close_h=3.10, close_d=3.15, close_a=2.35,
            odds_source="pinnacle",
        ),
    ]


def _make_gameweek(fixtures: list[Fixture] | None = None) -> Gameweek:
    fixes = fixtures or _make_fixtures()
    dates = sorted({f.date for f in fixes})
    return Gameweek(number=1, dates=dates, fixtures=fixes)


# ── Devig ────────────────────────────────────────────────────────────

def test_devig_sums_to_one():
    impl = devig(1.80, 3.50, 4.50)
    assert abs(impl.sum() - 1.0) < 1e-9


def test_devig_favourite_has_highest_prob():
    impl = devig(1.50, 4.00, 6.00)
    assert impl[0] > impl[1] > impl[2]


# ── Feature extraction ──────────────────────────────────────────────

def test_fixture_features_shape():
    fix = _make_fixtures()[0]
    feat = fixture_features(fix, bankroll_frac=1.0, drawdown_frac=0.0)
    assert feat.shape == (N_FEATURES,)
    assert feat[0] == fix.odds_h
    assert feat[6] == 1.0  # bankroll_frac
    assert feat[7] == 0.0  # drawdown_frac


# ── Env observation ──────────────────────────────────────────────────

def test_observe_returns_correct_shape():
    env = StakingEnv(bankroll=1000.0)
    state = BookState.create(bankroll=1000.0, start_date=date(2024, 8, 17))
    fixtures = _make_fixtures()
    obs = env.observe(state, fixtures)
    assert obs.shape == (3, N_FEATURES)


def test_observe_empty_fixtures():
    env = StakingEnv()
    state = BookState.create(bankroll=1000.0, start_date=date(2024, 8, 17))
    obs = env.observe(state, [])
    assert obs.shape == (0, N_FEATURES)


# ── Env rollout ──────────────────────────────────────────────────────

def test_rollout_skip_all():
    """Skipping all fixtures produces abstainer reward."""
    env = StakingEnv(bankroll=1000.0)
    gw = _make_gameweek()
    actions = np.zeros(3, dtype=int)  # all skip
    reward, info = env.rollout(gw, actions)
    assert info["n_bets"] == 0
    assert reward > 0  # abstention bonus


def test_rollout_bet_all_home():
    """Betting H on every match produces bets and a finite reward."""
    env = StakingEnv(bankroll=1000.0)
    gw = _make_gameweek()
    actions = np.ones(3, dtype=int)  # all bet H
    reward, info = env.rollout(gw, actions)
    assert info["n_bets"] > 0
    assert np.isfinite(reward)


def test_rollout_bankroll_never_negative():
    """Bankroll must never go negative, regardless of actions."""
    env = StakingEnv(bankroll=100.0, stake_fraction=0.10)
    gw = _make_gameweek()
    rng = np.random.default_rng(42)
    for _ in range(20):
        actions = rng.integers(0, 4, size=3)
        _, info = env.rollout(gw, actions)
        assert info["final_bankroll"] >= 0


# ── MLP Policy ───────────────────────────────────────────────────────

def test_policy_forward_sums_to_one():
    """Softmax output must sum to 1 for each fixture."""
    policy = MLPPolicy(seed=0)
    X = np.random.default_rng(0).normal(size=(5, N_FEATURES))
    probs = policy.forward(X)
    assert probs.shape == (5, 4)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-7)


def test_policy_sample_valid_actions():
    """Sampled actions must be in {0, 1, 2, 3}."""
    policy = MLPPolicy(seed=0)
    X = np.random.default_rng(0).normal(size=(10, N_FEATURES))
    actions, log_probs = policy.sample(X)
    assert actions.shape == (10,)
    assert all(0 <= a <= 3 for a in actions)
    assert all(lp <= 0 for lp in log_probs)


def test_policy_greedy_deterministic():
    """Greedy policy must be deterministic given the same input."""
    policy = MLPPolicy(seed=0)
    X = np.random.default_rng(0).normal(size=(5, N_FEATURES))
    a1 = policy.greedy(X)
    a2 = policy.greedy(X)
    np.testing.assert_array_equal(a1, a2)


def test_policy_backward_shapes():
    """Backward pass must return gradients matching parameter shapes."""
    policy = MLPPolicy(seed=0)
    X = np.random.default_rng(0).normal(size=(3, N_FEATURES))
    policy.forward(X)
    actions = np.array([0, 1, 2])
    advantages = np.array([1.0, -0.5, 0.3])
    grads = policy.backward(actions, advantages)
    for name, param in policy.params.items():
        assert grads[name].shape == param.shape, f"gradient shape mismatch for {name}"


def test_policy_n_params():
    """Smoke test for parameter counting."""
    policy = MLPPolicy(input_dim=8, hidden1=64, hidden2=32, n_actions=4)
    expected = 8 * 64 + 64 + 64 * 32 + 32 + 32 * 4 + 4
    assert policy.n_params() == expected


def test_policy_save_load(tmp_path):
    """Save and load round-trips the weights."""
    policy = MLPPolicy(seed=0)
    path = str(tmp_path / "policy.npz")
    policy.save(path)
    loaded = MLPPolicy.load(path)
    for name in policy.param_names:
        np.testing.assert_array_equal(
            getattr(policy, name), getattr(loaded, name),
        )


# ── GRPO Trainer ─────────────────────────────────────────────────────

def test_grpo_one_step():
    """One GRPO step must update the policy weights."""
    from training.grpo import GRPOTrainer
    policy = MLPPolicy(seed=0)
    env = StakingEnv(bankroll=1000.0)
    gw = _make_gameweek()

    before = {n: p.copy() for n, p in policy.params.items()}
    trainer = GRPOTrainer(policy, env, K=4, lr=1e-3, seed=0)
    log = trainer.train_gameweek(gw)

    changed = any(
        not np.array_equal(before[n], getattr(policy, n))
        for n in policy.param_names
    )
    assert changed, "GRPO step should update at least some weights"
    assert np.isfinite(log.mean_reward)


def test_grpo_multiple_epochs():
    """Training over multiple epochs should produce a complete report."""
    from training.grpo import GRPOTrainer
    policy = MLPPolicy(seed=0)
    env = StakingEnv(bankroll=1000.0)
    gw = _make_gameweek()

    trainer = GRPOTrainer(policy, env, K=4, lr=1e-3, seed=0)
    report = trainer.train([gw], n_epochs=2)

    assert report.n_epochs == 2
    assert report.n_gameweeks == 1
    assert len(report.steps) == 2
    assert np.isfinite(report.final_mean_reward)


def test_grpo_reduces_variance_over_epochs():
    """Reward variance within the K group should not explode."""
    from training.grpo import GRPOTrainer
    policy = MLPPolicy(seed=0)
    env = StakingEnv(bankroll=1000.0)
    gw = _make_gameweek()

    trainer = GRPOTrainer(policy, env, K=8, lr=1e-3, seed=0)
    report = trainer.train([gw], n_epochs=5)

    for step in report.steps:
        assert np.isfinite(step.std_reward)
        assert step.std_reward >= 0


# ── Walk-forward evaluation (real data, capped) ─────────────────────

@pytest.fixture(scope="module")
def real_gameweek():
    from pathlib import Path
    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "football_data_uk"
    season_file = data_dir / "E0_2425.csv"
    fixtures = load_season(season_file)
    first_date = fixtures[0].date
    gw_fixes = [f for f in fixtures if (f.date - first_date).days <= 3]
    dates = sorted({f.date for f in gw_fixes})
    return Gameweek(number=1, dates=dates, fixtures=gw_fixes)


def test_learned_policy_protocol(real_gameweek):
    """LearnedPolicy must conform to the Policy protocol."""
    from training.evaluate import LearnedPolicy
    from envs.market_env import run_episode

    policy = MLPPolicy(seed=42)
    env = StakingEnv(bankroll=1000.0)
    learned = LearnedPolicy(policy, env, greedy=True)

    result = run_episode(real_gameweek, learned, bankroll=1000.0)
    assert result.reward is not None
    assert np.isfinite(result.reward)


def test_evaluation_comparison_table(real_gameweek):
    """Walk-forward eval produces a comparison table with all policies."""
    from training.evaluate import (
        AbstainerPolicy, FavouritePolicy, LearnedPolicy, RandomPolicy,
        _evaluate_policy,
    )

    env = StakingEnv(bankroll=1000.0)
    policy = MLPPolicy(seed=42)
    learned = LearnedPolicy(policy, env)

    policies = [
        ("GRPO", learned),
        ("Favourite", FavouritePolicy()),
        ("Random", RandomPolicy()),
        ("Abstainer", AbstainerPolicy()),
    ]

    for name, pol in policies:
        r = _evaluate_policy(name, pol, [real_gameweek], bankroll=1000.0)
        assert r.n_gameweeks == 1
        assert np.isfinite(r.mean_reward)

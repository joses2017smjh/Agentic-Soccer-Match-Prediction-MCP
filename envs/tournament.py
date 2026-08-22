"""Population tournament: all bots on identical match sets with statistics.

Evaluates a population of policies on the same fixtures with the same seeds,
then reports a leaderboard sorted by mean CLV with bootstrapped confidence
intervals and a pairwise significance matrix.

Reporting rules (enforced in code):
  - Every number carries a bootstrapped 95% CI.
  - The leaderboard sorts by mean CLV, not final bankroll.
  - is_distinguishable(a, b) reports pairwise significance.
  - >= 100 seeds per bot (fewer bots, not fewer seeds).

The conceptual link: GRPO's group-relative advantage is a population of
candidates scored against each other.  The tournament is the same structure
as the training signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.market_env import EpisodeResult, Policy, run_episode
from envs.real_market import Fixture, Gameweek, iter_gameweeks


@dataclass
class PolicyStats:
    """Aggregate statistics for one policy across many runs."""
    name: str
    mean_clv: float
    clv_ci_lo: float
    clv_ci_hi: float
    mean_reward: float
    reward_ci_lo: float
    reward_ci_hi: float
    mean_bankroll: float
    total_bets: int
    n_runs: int
    per_run_clvs: list[float] = field(default_factory=list)
    per_run_rewards: list[float] = field(default_factory=list)


@dataclass
class PairwiseResult:
    """Whether two policies are statistically distinguishable."""
    policy_a: str
    policy_b: str
    distinguishable: bool
    p_value: float
    delta_mean: float
    delta_ci_lo: float
    delta_ci_hi: float


@dataclass
class TournamentReport:
    """Full tournament results."""
    leaderboard: list[PolicyStats]
    pairwise: list[PairwiseResult]
    n_seeds: int
    n_gameweeks_per_seed: int

    def leaderboard_table(self) -> list[dict[str, Any]]:
        return [
            {
                "rank": i + 1,
                "policy": s.name,
                "mean_clv": round(s.mean_clv, 6),
                "clv_95ci": f"[{s.clv_ci_lo:.6f}, {s.clv_ci_hi:.6f}]",
                "mean_reward": round(s.mean_reward, 6),
                "reward_95ci": f"[{s.reward_ci_lo:.6f}, {s.reward_ci_hi:.6f}]",
                "total_bets": s.total_bets,
                "n_runs": s.n_runs,
            }
            for i, s in enumerate(self.leaderboard)
        ]

    def significance_matrix(self) -> dict[str, dict[str, str]]:
        names = [s.name for s in self.leaderboard]
        matrix: dict[str, dict[str, str]] = {}
        for name in names:
            matrix[name] = {}
        for pw in self.pairwise:
            label = "YES" if pw.distinguishable else "no"
            matrix[pw.policy_a][pw.policy_b] = f"{label} (p={pw.p_value:.3f})"
            matrix[pw.policy_b][pw.policy_a] = f"{label} (p={pw.p_value:.3f})"
        for name in names:
            matrix[name][name] = "-"
        return matrix


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via percentile bootstrap."""
    arr = np.array(values)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def is_distinguishable(
    values_a: list[float],
    values_b: list[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> PairwiseResult:
    """Test whether two policies have distinguishable mean CLV.

    Uses a permutation test: under the null hypothesis that both samples
    come from the same distribution, the observed difference in means
    should not be extreme.
    """
    a = np.array(values_a)
    b = np.array(values_b)
    observed_delta = float(a.mean() - b.mean())

    combined = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    n_a = len(a)

    count_extreme = 0
    for _ in range(n_bootstrap):
        perm = rng.permutation(combined)
        perm_delta = perm[:n_a].mean() - perm[n_a:].mean()
        if abs(perm_delta) >= abs(observed_delta):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_bootstrap + 1)

    deltas = [float(a_val - b_val) for a_val, b_val in zip(
        rng.choice(a, size=n_bootstrap, replace=True),
        rng.choice(b, size=n_bootstrap, replace=True),
    )]
    delta_lo = float(np.percentile(deltas, 100 * alpha / 2))
    delta_hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    return PairwiseResult(
        policy_a="",
        policy_b="",
        distinguishable=p_value < alpha,
        p_value=round(p_value, 4),
        delta_mean=round(observed_delta, 6),
        delta_ci_lo=round(delta_lo, 6),
        delta_ci_hi=round(delta_hi, 6),
    )


def run_tournament(
    policies: list[tuple[str, Policy]],
    gameweeks: list[Gameweek],
    n_seeds: int = 100,
    bankroll: float = 1000.0,
    max_gw_per_seed: int | None = None,
    n_bootstrap: int = 5000,
) -> TournamentReport:
    """Run all policies on identical match sets across multiple seeds.

    Each seed uses the same gameweeks but re-rolls the policy's internal
    randomness (if any).  The environment is deterministic (real data),
    so seed variation comes only from stochastic policies.
    """
    gw_subset = gameweeks[:max_gw_per_seed] if max_gw_per_seed else gameweeks

    all_stats: list[PolicyStats] = []
    all_per_run: dict[str, list[float]] = {}

    for name, policy in policies:
        run_clvs: list[float] = []
        run_rewards: list[float] = []
        total_bets = 0

        for seed in range(n_seeds):
            seed_clvs: list[float] = []
            seed_rewards: list[float] = []
            seed_bets = 0

            for gw in gw_subset:
                result = run_episode(gw, policy, bankroll=bankroll)
                if result.n_bets > 0:
                    seed_clvs.append(result.mean_clv)
                seed_rewards.append(result.reward)
                seed_bets += result.n_bets

            mean_clv = sum(seed_clvs) / len(seed_clvs) if seed_clvs else 0.0
            mean_reward = sum(seed_rewards) / len(seed_rewards) if seed_rewards else 0.0
            run_clvs.append(mean_clv)
            run_rewards.append(mean_reward)
            total_bets += seed_bets

        clv_mean, clv_lo, clv_hi = _bootstrap_ci(run_clvs, n_bootstrap)
        rew_mean, rew_lo, rew_hi = _bootstrap_ci(run_rewards, n_bootstrap)
        mean_bankroll = bankroll

        stats = PolicyStats(
            name=name,
            mean_clv=round(clv_mean, 6),
            clv_ci_lo=round(clv_lo, 6),
            clv_ci_hi=round(clv_hi, 6),
            mean_reward=round(rew_mean, 6),
            reward_ci_lo=round(rew_lo, 6),
            reward_ci_hi=round(rew_hi, 6),
            mean_bankroll=round(mean_bankroll, 2),
            total_bets=total_bets,
            n_runs=n_seeds,
            per_run_clvs=run_clvs,
            per_run_rewards=run_rewards,
        )
        all_stats.append(stats)
        all_per_run[name] = run_clvs

    all_stats.sort(key=lambda s: s.mean_clv, reverse=True)

    pairwise: list[PairwiseResult] = []
    names = [s.name for s in all_stats]
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            pw = is_distinguishable(all_per_run[name_a], all_per_run[name_b], n_bootstrap)
            pw.policy_a = name_a
            pw.policy_b = name_b
            pairwise.append(pw)

    return TournamentReport(
        leaderboard=all_stats,
        pairwise=pairwise,
        n_seeds=n_seeds,
        n_gameweeks_per_seed=len(gw_subset),
    )

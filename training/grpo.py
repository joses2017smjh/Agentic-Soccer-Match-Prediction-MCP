"""Group Relative Policy Optimization for the staking environment.

GRPO (Shao et al., 2024) replaces the learned value baseline in PPO with
a group-relative baseline: for each state, sample K action sequences,
score each with the environment reward, and normalize advantages within
the group.  This eliminates the value network entirely while reducing
gradient variance compared to vanilla REINFORCE.

    A_i = (R_i - mean(R_{1..K})) / (std(R_{1..K}) + eps)

The policy gradient is then the standard REINFORCE estimator weighted
by these group-relative advantages.

Why GRPO fits this problem:
- The staking environment has a stochastic reward (match outcomes vary),
  so multiple rollouts per state reduce variance.
- The state space is modest (8 features x ~10 fixtures), so K=8
  rollouts per gameweek are cheap.
- No value network means fewer moving parts and cleaner attribution of
  what the policy learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.real_market import Gameweek
from mcp_servers.book_server.state import BookState
from training.gym_wrapper import StakingEnv
from training.policy import MLPPolicy


# ── Adam optimizer ───────────────────────────────────────────────────

@dataclass
class AdamState:
    m: dict[str, np.ndarray] = field(default_factory=dict)
    v: dict[str, np.ndarray] = field(default_factory=dict)
    t: int = 0


def adam_step(
    policy: MLPPolicy,
    grads: dict[str, np.ndarray],
    state: AdamState,
    lr: float = 3e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """In-place Adam update on the policy parameters."""
    state.t += 1
    for name in policy.param_names:
        param = getattr(policy, name)
        g = grads[name]

        if name not in state.m:
            state.m[name] = np.zeros_like(param)
            state.v[name] = np.zeros_like(param)

        state.m[name] = beta1 * state.m[name] + (1 - beta1) * g
        state.v[name] = beta2 * state.v[name] + (1 - beta2) * g ** 2

        m_hat = state.m[name] / (1 - beta1 ** state.t)
        v_hat = state.v[name] / (1 - beta2 ** state.t)

        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ── Training logs ────────────────────────────────────────────────────

@dataclass
class StepLog:
    gameweek: int
    epoch: int
    mean_reward: float
    std_reward: float
    best_reward: float
    mean_n_bets: float
    mean_clv: float


@dataclass
class TrainReport:
    steps: list[StepLog]
    n_epochs: int
    n_gameweeks: int
    final_mean_reward: float

    def summary_table(self) -> list[dict[str, Any]]:
        """Per-epoch aggregate for display."""
        from collections import defaultdict
        by_epoch: dict[int, list[StepLog]] = defaultdict(list)
        for s in self.steps:
            by_epoch[s.epoch].append(s)
        rows = []
        for epoch, logs in sorted(by_epoch.items()):
            rows.append({
                "epoch": epoch,
                "mean_reward": round(np.mean([l.mean_reward for l in logs]), 6),
                "mean_clv": round(np.mean([l.mean_clv for l in logs]), 6),
                "mean_bets": round(np.mean([l.mean_n_bets for l in logs]), 1),
            })
        return rows


# ── GRPO Trainer ─────────────────────────────────────────────────────

class GRPOTrainer:
    """GRPO training loop over gameweek episodes.

    For each gameweek in each epoch:
    1. Observe the initial state and build the feature matrix.
    2. Sample K action sets from the current policy.
    3. Execute each through the environment to obtain rewards.
    4. Compute group-relative advantages.
    5. Accumulate REINFORCE gradients weighted by advantages.
    6. Update the policy with Adam.
    """

    def __init__(
        self,
        policy: MLPPolicy,
        env: StakingEnv,
        K: int = 8,
        lr: float = 3e-4,
        seed: int = 42,
    ):
        self.policy = policy
        self.env = env
        self.K = K
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.adam = AdamState()

    def train_gameweek(
        self, gameweek: Gameweek, epoch: int = 0,
    ) -> StepLog:
        """One GRPO update on a single gameweek."""
        initial_state = BookState.create(
            bankroll=self.env.initial_bankroll,
            start_date=gameweek.dates[0],
        )
        obs = self.env.observe(initial_state, gameweek.fixtures)

        if obs.shape[0] == 0:
            return StepLog(
                gameweek=gameweek.number, epoch=epoch,
                mean_reward=0.0, std_reward=0.0, best_reward=0.0,
                mean_n_bets=0.0, mean_clv=0.0,
            )

        # sample K trajectories
        all_actions: list[np.ndarray] = []
        rewards: list[float] = []
        infos: list[dict] = []

        for _ in range(self.K):
            actions, _ = self.policy.sample(obs, rng=self.rng)
            reward, info = self.env.rollout(gameweek, actions)
            all_actions.append(actions)
            rewards.append(reward)
            infos.append(info)

        r_arr = np.array(rewards)
        mean_r = r_arr.mean()
        std_r = r_arr.std() + 1e-8

        # group-relative advantages
        advantages = (r_arr - mean_r) / std_r

        # accumulate gradients across K samples
        total_grads = {n: np.zeros_like(getattr(self.policy, n))
                       for n in self.policy.param_names}

        for k in range(self.K):
            self.policy.forward(obs)
            adv_vec = np.full(obs.shape[0], advantages[k])
            grads = self.policy.backward(all_actions[k], adv_vec)
            for name in total_grads:
                total_grads[name] += grads[name]

        for name in total_grads:
            total_grads[name] /= self.K

        adam_step(self.policy, total_grads, self.adam, lr=self.lr)

        mean_clv = float(np.mean([info["mean_clv"] for info in infos]))
        mean_bets = float(np.mean([info["n_bets"] for info in infos]))

        return StepLog(
            gameweek=gameweek.number,
            epoch=epoch,
            mean_reward=float(mean_r),
            std_reward=float(std_r - 1e-8),
            best_reward=float(r_arr.max()),
            mean_n_bets=mean_bets,
            mean_clv=mean_clv,
        )

    def train(
        self,
        gameweeks: list[Gameweek],
        n_epochs: int = 3,
    ) -> TrainReport:
        """Full training loop: n_epochs passes over all gameweeks."""
        steps: list[StepLog] = []
        for epoch in range(n_epochs):
            for gw in gameweeks:
                log = self.train_gameweek(gw, epoch=epoch)
                steps.append(log)

        final_rewards = [s.mean_reward for s in steps[-len(gameweeks):]]
        return TrainReport(
            steps=steps,
            n_epochs=n_epochs,
            n_gameweeks=len(gameweeks),
            final_mean_reward=float(np.mean(final_rewards)) if final_rewards else 0.0,
        )

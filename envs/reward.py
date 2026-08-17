"""CLV-primary composite reward for the gameweek environment.

The reward is designed to be:
- **CLV-dominated**: closing-line value is the signal, not P&L (which is
  mostly variance over a short horizon).
- **Abstention-aware**: a small positive reward for not betting when no
  edge exists, but not enough to make always-abstain competitive.
- **Churn-penalized**: many tiny bets to farm per-bet reward are penalized.
- **Drawdown-penalized**: large drawdowns reduce reward.

Composite formula:
  R = w_clv * mean_clv
    + w_abstain * abstention_bonus
    - w_churn * churn_penalty
    - w_drawdown * drawdown_penalty

Weights are chosen so that:
- A policy with mean CLV of +5% and reasonable bet count dominates.
- An abstainer scores ~0.01 (small positive, not zero).
- A churner who places 50+ tiny bets per gameweek is penalized.
- A martingale that draws down 30% is penalized even if CLV is positive.
"""

from __future__ import annotations

from mcp_servers.book_server.state import BookState

W_CLV = 1.0
W_ABSTAIN = 0.01
W_CHURN = 0.002
W_DRAWDOWN = 0.5

CHURN_THRESHOLD = 15


def compute_episode_reward(state: BookState) -> tuple[float, dict[str, float]]:
    """Compute the composite reward from a completed episode.

    Returns (reward, component_dict) for transparency and debugging.
    """
    settled = state.settled_positions()
    n_bets = len(settled)

    mean_clv = sum(p.clv for p in settled) / n_bets if n_bets > 0 else 0.0
    clv_term = W_CLV * mean_clv

    abstention_bonus = W_ABSTAIN if n_bets == 0 else 0.0

    excess_bets = max(0, n_bets - CHURN_THRESHOLD)
    churn_penalty = W_CHURN * excess_bets

    dd_fraction = state.max_drawdown() / state.initial_bankroll if state.initial_bankroll > 0 else 0.0
    drawdown_penalty = W_DRAWDOWN * dd_fraction ** 2

    reward = clv_term + abstention_bonus - churn_penalty - drawdown_penalty

    components = {
        "clv_term": round(clv_term, 6),
        "abstention_bonus": round(abstention_bonus, 6),
        "churn_penalty": round(churn_penalty, 6),
        "drawdown_penalty": round(drawdown_penalty, 6),
        "mean_clv": round(mean_clv, 6),
        "n_bets": n_bets,
        "max_drawdown_frac": round(dd_fraction, 4),
    }

    return round(reward, 6), components

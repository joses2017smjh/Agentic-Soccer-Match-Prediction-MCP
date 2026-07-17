"""Player props: Poisson allocation of team xG to individual players.

The team's Dixon–Coles mean mu is split across players by allocation weights

    w_i = xg_share_i · minutes_i/90 · availability_i · setpiece_i · role_i

and renormalized so Σ λ_i = mu · (1 − own_goal_share): players collectively
account for the team's goals, so ruling a striker out (availability 0)
redistributes his share to teammates instead of deleting it — exactly the
lineup-shock propagation the suggestion layer trades on. Anytime-scorer
probability is then 1 − exp(−λ_i); assists run identically on xA shares
against expected assisted goals.

Inputs per player (from historical event data + the availability report):
    xg_share        historical share of the team's xG while on pitch
    xa_share        same for expected assists
    exp_minutes     projected minutes for this match
    availability    availability_pct from TeamAvailabilityReport (0..1)
    setpiece_mult   >1 for penalty/free-kick takers (default 1)
    role_mult       manager-specific usage adjustment (default 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLS = ["player", "xg_share", "xa_share", "exp_minutes", "availability"]
OWN_GOAL_SHARE = 0.02  # league-typical share of goals that are own goals


def _weights(players: pd.DataFrame, share_col: str) -> np.ndarray:
    w = (
        players[share_col].to_numpy(dtype=float)
        * (players["exp_minutes"].to_numpy(dtype=float) / 90.0).clip(0.0, 1.0)
        * players["availability"].to_numpy(dtype=float)
        * players.get("setpiece_mult", pd.Series(1.0, index=players.index)).to_numpy(dtype=float)
        * players.get("role_mult", pd.Series(1.0, index=players.index)).to_numpy(dtype=float)
    )
    return np.clip(w, 0.0, None)


def allocate_player_props(
    players: pd.DataFrame,
    team_mu: float,
    *,
    assist_ratio: float = 0.75,
) -> pd.DataFrame:
    """Return players + goal_lambda, p_anytime_scorer, assist_lambda, p_assist.

    ``assist_ratio`` is the fraction of goals that carry an assist (~0.7-0.8
    in top leagues); the assist pool is team_mu · assist_ratio.
    """
    missing = [c for c in REQUIRED_COLS if c not in players.columns]
    if missing:
        raise ValueError(f"player frame missing columns: {missing}")
    out = players.copy()

    goal_pool = team_mu * (1.0 - OWN_GOAL_SHARE)
    for share_col, lam_col, prob_col, pool in [
        ("xg_share", "goal_lambda", "p_anytime_scorer", goal_pool),
        ("xa_share", "assist_lambda", "p_assist", team_mu * assist_ratio),
    ]:
        w = _weights(out, share_col)
        total = w.sum()
        lam = pool * w / total if total > 0 else np.zeros_like(w)
        out[lam_col] = lam
        out[prob_col] = 1.0 - np.exp(-lam)

    return out.sort_values("p_anytime_scorer", ascending=False).reset_index(drop=True)

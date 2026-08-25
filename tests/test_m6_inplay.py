"""Tests for M6: in-play Bayesian updating (C1) and counterfactual lab (C2)."""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------
# C1: in-play Bayesian updating
# --------------------------------------------------------------------------

class TestRemainingGrid:
    def test_pre_match_matches_full_grid(self):
        from src.models.in_play import remaining_grid
        from src.models.score_grid import score_grid
        full = score_grid(1.5, 1.0, rho=-0.1)
        rem = remaining_grid(1.5, 1.0, elapsed=0.0, rho=-0.1)
        np.testing.assert_allclose(rem, full, atol=1e-6)

    def test_sums_to_one(self):
        from src.models.in_play import remaining_grid
        grid = remaining_grid(1.5, 1.2, elapsed=45.0, goals_home=1, goals_away=0)
        assert grid.sum() == pytest.approx(1.0, abs=1e-6)

    def test_only_reachable_scorelines(self):
        from src.models.in_play import remaining_grid
        grid = remaining_grid(1.5, 1.0, elapsed=60.0,
                              goals_home=2, goals_away=1)
        assert grid[:2, :].sum() == pytest.approx(0.0, abs=1e-10)
        assert grid[:, :1].sum() == pytest.approx(0.0, abs=1e-10)

    def test_late_game_concentrates_on_current(self):
        from src.models.in_play import remaining_grid
        grid = remaining_grid(1.5, 1.0, elapsed=89.0,
                              goals_home=1, goals_away=0)
        assert grid[1, 0] > 0.5

    def test_red_card_reduces_attack(self):
        from src.models.in_play import remaining_grid
        grid_no_red = remaining_grid(1.5, 1.0, elapsed=30.0)
        grid_red = remaining_grid(1.5, 1.0, elapsed=30.0, red_home=1)
        # with a red card, home should score fewer goals on average
        size = grid_no_red.shape[0]
        goals = np.arange(size)
        avg_home_no_red = (grid_no_red.sum(axis=1) * goals).sum()
        avg_home_red = (grid_red.sum(axis=1) * goals).sum()
        assert avg_home_red < avg_home_no_red


class TestInPlayProbs:
    def test_pre_match_consistent(self):
        from src.models.in_play import in_play_probs
        from src.models.score_grid import outcome_probs, score_grid
        pre = outcome_probs(score_grid(1.5, 1.0))
        live = in_play_probs(1.5, 1.0, elapsed=0.0)
        for k in ("home", "draw", "away"):
            assert live[k] == pytest.approx(pre[k], abs=1e-6)

    def test_sums_to_one(self):
        from src.models.in_play import in_play_probs
        p = in_play_probs(1.5, 1.0, elapsed=50.0,
                          goals_home=1, goals_away=1)
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)

    def test_goal_shifts_probability(self):
        from src.models.in_play import in_play_probs
        p0 = in_play_probs(1.5, 1.0, elapsed=30.0,
                           goals_home=0, goals_away=0)
        p1 = in_play_probs(1.5, 1.0, elapsed=30.0,
                           goals_home=1, goals_away=0)
        assert p1["home"] > p0["home"]

    def test_full_time_reflects_score(self):
        from src.models.in_play import in_play_probs
        p = in_play_probs(1.5, 1.0, elapsed=90.0,
                          goals_home=2, goals_away=1)
        assert p["home"] > 0.9


class TestWinProbabilityCurve:
    def test_curve_length(self):
        from src.models.in_play import win_probability_curve
        curve = win_probability_curve(1.5, 1.0, events=[], resolution=91)
        assert len(curve) == 91

    def test_curve_with_goal(self):
        from src.models.in_play import win_probability_curve
        events = [{"minute": 30, "type": "goal_home"}]
        curve = win_probability_curve(1.5, 1.0, events=events, resolution=91)
        pre_goal = [c for c in curve if c["minute"] < 30]
        post_goal = [c for c in curve if c["minute"] > 30]
        assert post_goal[0]["home"] > pre_goal[-1]["home"]

    def test_curve_sums_to_one(self):
        from src.models.in_play import win_probability_curve
        events = [
            {"minute": 20, "type": "goal_home"},
            {"minute": 55, "type": "goal_away"},
        ]
        curve = win_probability_curve(1.5, 1.0, events=events)
        for point in curve:
            total = point["home"] + point["draw"] + point["away"]
            assert total == pytest.approx(1.0, abs=1e-5)


class TestInPlayState:
    def test_initial_state(self):
        from src.models.in_play import InPlayState
        s = InPlayState(mu_home=1.5, mu_away=1.0)
        snap = s.snapshot()
        assert snap["score"] == "0-0"
        assert snap["minute"] == 0.0

    def test_update_goal(self):
        from src.models.in_play import InPlayState
        s = InPlayState(mu_home=1.5, mu_away=1.0)
        s.update({"minute": 35, "type": "goal_home"})
        assert s.goals_home == 1
        assert s.elapsed == 35
        snap = s.snapshot()
        assert snap["score"] == "1-0"

    def test_update_red_card(self):
        from src.models.in_play import InPlayState
        s = InPlayState(mu_home=1.5, mu_away=1.0)
        s.update({"minute": 40, "type": "red_away"})
        assert s.red_away == 1
        p_after = s.probs()
        # away team weakened, home should be more likely to win
        s2 = InPlayState(mu_home=1.5, mu_away=1.0, elapsed=40.0)
        p_no_red = s2.probs()
        assert p_after["home"] > p_no_red["home"]


# --------------------------------------------------------------------------
# C2: counterfactual lab
# --------------------------------------------------------------------------

class TestCounterfactualRemovePlayer:
    def test_removes_player_reduces_xg(self):
        from src.models.counterfactual import Perturbation, run_counterfactual
        r = run_counterfactual(
            1.5, 1.0,
            Perturbation("remove_player", {"side": "home", "xg_contribution": 0.2}),
        )
        assert r.perturbed_mu[0] < r.baseline_mu[0]
        assert r.perturbed_mu[1] == r.baseline_mu[1]

    def test_reduces_home_win_prob(self):
        from src.models.counterfactual import Perturbation, run_counterfactual
        r = run_counterfactual(
            1.5, 1.0,
            Perturbation("remove_player", {"side": "home"}),
        )
        assert r.deltas["home"] < 0

    def test_away_side(self):
        from src.models.counterfactual import Perturbation, run_counterfactual
        r = run_counterfactual(
            1.5, 1.0,
            Perturbation("remove_player", {"side": "away", "xg_contribution": 0.15}),
        )
        assert r.perturbed_mu[1] < r.baseline_mu[1]
        assert r.deltas["home"] > 0


class TestCounterfactualRestDays:
    def test_fewer_rest_reduces_xg(self):
        from src.models.counterfactual import Perturbation, run_counterfactual
        r = run_counterfactual(
            1.5, 1.0,
            Perturbation("rest_days", {"side": "home", "days": 2, "baseline_days": 5}),
        )
        assert r.perturbed_mu[0] < r.baseline_mu[0]

    def test_more_rest_increases_xg(self):
        from src.models.counterfactual import Perturbation, run_counterfactual
        r = run_counterfactual(
            1.5, 1.0,
            Perturbation("rest_days", {"side": "home", "days": 7, "baseline_days": 4}),
        )
        assert r.perturbed_mu[0] > r.baseline_mu[0]


class TestCounterfactualNeutralVenue:
    def test_neutral_reduces_home_xg(self):
        from src.models.counterfactual import Perturbation, run_counterfactual
        r = run_counterfactual(
            1.5, 1.0,
            Perturbation("neutral_venue"),
        )
        assert r.perturbed_mu[0] < r.baseline_mu[0]
        assert r.perturbed_mu[1] > r.baseline_mu[1]

    def test_neutral_reduces_home_win_prob(self):
        from src.models.counterfactual import Perturbation, run_counterfactual
        r = run_counterfactual(
            1.5, 1.0,
            Perturbation("neutral_venue"),
        )
        assert r.deltas["home"] < 0


class TestCounterfactualAdjustXg:
    def test_boost_home(self):
        from src.models.counterfactual import Perturbation, run_counterfactual
        r = run_counterfactual(
            1.5, 1.0,
            Perturbation("adjust_xg", {"delta_home": 0.5}),
        )
        assert r.perturbed_mu[0] == pytest.approx(2.0)
        assert r.deltas["home"] > 0

    def test_reduce_away(self):
        from src.models.counterfactual import Perturbation, run_counterfactual
        r = run_counterfactual(
            1.5, 1.0,
            Perturbation("adjust_xg", {"delta_away": -0.3}),
        )
        assert r.perturbed_mu[1] == pytest.approx(0.7)
        assert r.deltas["home"] > 0


class TestCounterfactualBatch:
    def test_batch_returns_all(self):
        from src.models.counterfactual import (
            Perturbation, run_counterfactual_batch,
        )
        perturbs = [
            Perturbation("remove_player", {"side": "home"}),
            Perturbation("neutral_venue"),
            Perturbation("adjust_xg", {"delta_home": 0.3}),
        ]
        results = run_counterfactual_batch(1.5, 1.0, perturbs)
        assert len(results) == 3
        for r in results:
            total = sum(r.perturbed_probs.values())
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_deltas_sum_to_zero(self):
        from src.models.counterfactual import Perturbation, run_counterfactual
        r = run_counterfactual(
            1.5, 1.0,
            Perturbation("remove_player", {"side": "home"}),
        )
        assert sum(r.deltas.values()) == pytest.approx(0.0, abs=1e-6)


class TestCounterfactualFormat:
    def test_format_table(self):
        from src.models.counterfactual import (
            Perturbation, format_counterfactual_table,
            run_counterfactual_batch,
        )
        results = run_counterfactual_batch(1.5, 1.0, [
            Perturbation("neutral_venue"),
        ])
        table = format_counterfactual_table(results)
        assert len(table) == 1
        row = table[0]
        assert "perturbation" in row
        assert "delta" in row
        assert "mu_change" in row


class TestCounterfactualUnknownKind:
    def test_unknown_raises(self):
        from src.models.counterfactual import Perturbation, run_counterfactual
        with pytest.raises(ValueError, match="Unknown perturbation"):
            run_counterfactual(1.5, 1.0, Perturbation("teleport"))

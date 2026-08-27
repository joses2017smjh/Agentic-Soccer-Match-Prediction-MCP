"""Tests for M7: Bayesian dynamic (B5), debate critic (C4),
disagreement board (C5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------
# B5: Bayesian dynamic team strength
# --------------------------------------------------------------------------

class TestBayesianDynamicPredict:
    def test_predict_xg_positive(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic()
        mu_h, mu_a = m.predict_xg("A", "B")
        assert mu_h > 0
        assert mu_a > 0

    def test_home_advantage(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic()
        mu_h, mu_a = m.predict_xg("A", "B", neutral=False)
        mu_hn, mu_an = m.predict_xg("A", "B", neutral=True)
        assert mu_h > mu_hn  # home advantage boosts home xG

    def test_predict_probs_sum_to_one(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic()
        p = m.predict_probs("A", "B")
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)

    def test_unknown_teams_get_base(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic()
        u = m.uncertainty("NewTeam")
        assert u["attack_var"] == 1.0
        assert u["defense_var"] == 1.0


class TestBayesianDynamicUpdate:
    def test_update_shifts_attack(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic()
        mu_before, _ = m.predict_xg("A", "B")
        m.update("A", "B", goals_home=5, goals_away=0)
        mu_after, _ = m.predict_xg("A", "B")
        assert mu_after > mu_before

    def test_update_reduces_variance(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic()
        var_before = m.uncertainty("A")["attack_var"]
        m.update("A", "B", goals_home=2, goals_away=1)
        var_after = m.uncertainty("A")["attack_var"]
        assert var_after < var_before

    def test_variance_stays_positive(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic()
        for _ in range(50):
            m.update("A", "B", goals_home=3, goals_away=0)
        u = m.uncertainty("A")
        assert u["attack_var"] >= 0.01
        assert u["defense_var"] >= 0.01


class TestBayesianDynamicInflate:
    def test_inflate_widens_variance(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic(delta=1.1)
        m.update("A", "B", goals_home=1, goals_away=1)
        var_before = m.uncertainty("A")["attack_var"]
        m.inflate()
        var_after = m.uncertainty("A")["attack_var"]
        assert var_after > var_before

    def test_delta_one_no_change(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic(delta=1.0)
        m.update("A", "B", goals_home=1, goals_away=0)
        var_before = m.uncertainty("A")["attack_var"]
        m.inflate()
        assert m.uncertainty("A")["attack_var"] == pytest.approx(var_before)


class TestBayesianDynamicFit:
    def _results(self):
        return pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-08",
                     "2024-01-08", "2024-01-15"],
            "home_team": ["A", "C", "B", "A", "C"],
            "away_team": ["B", "D", "C", "D", "A"],
            "home_score": [3, 1, 0, 2, 1],
            "away_score": [0, 1, 2, 0, 1],
            "neutral": [False, False, False, False, False],
        })

    def test_fit_returns_self(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic()
        assert m.fit(self._results()) is m

    def test_fit_populates_teams(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic().fit(self._results())
        assert len(m._teams) == 4

    def test_fit_strong_team_higher_attack(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic().fit(self._results())
        # A scored 3-0, 2-0 at home, should have high attack
        assert m._teams["A"].attack > m._teams["D"].attack

    def test_rankings(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic().fit(self._results())
        r = m.team_rankings(top_n=2)
        assert len(r) == 2
        assert r[0]["net"] >= r[1]["net"]

    def test_rankings_all(self):
        from src.models.bayesian_dynamic import BayesianDynamic
        m = BayesianDynamic().fit(self._results())
        r = m.team_rankings()
        assert len(r) == 4


# --------------------------------------------------------------------------
# C4: debate critic
# --------------------------------------------------------------------------

class TestMarketAdvocate:
    def test_support_when_close(self):
        from src.models.debate import market_advocate
        model = {"home": 0.50, "draw": 0.25, "away": 0.25}
        market = {"home": 0.48, "draw": 0.26, "away": 0.26}
        arg = market_advocate(model, market, threshold=0.05)
        assert arg.position == "support"
        assert arg.agent == "market_advocate"

    def test_challenge_when_far(self):
        from src.models.debate import market_advocate
        model = {"home": 0.70, "draw": 0.15, "away": 0.15}
        market = {"home": 0.45, "draw": 0.28, "away": 0.27}
        arg = market_advocate(model, market, threshold=0.05)
        assert arg.position == "challenge"
        assert len(arg.points) > 0
        assert arg.proposed_adjustment is not None


class TestStatisticalSkeptic:
    def test_support_when_reasonable(self):
        from src.models.debate import statistical_skeptic
        model = {"home": 0.45, "draw": 0.28, "away": 0.27}
        arg = statistical_skeptic(model)
        assert arg.position == "support"

    def test_challenge_overconfident(self):
        from src.models.debate import statistical_skeptic
        model = {"home": 0.90, "draw": 0.05, "away": 0.05}
        arg = statistical_skeptic(model)
        assert arg.position == "challenge"
        assert any("overconfidence" in p for p in arg.points)

    def test_challenge_extreme_base_rate_deviation(self):
        from src.models.debate import statistical_skeptic
        model = {"home": 0.05, "draw": 0.90, "away": 0.05}
        arg = statistical_skeptic(model)
        assert arg.position == "challenge"
        assert any("base rate" in p for p in arg.points)

    def test_calibration_error_flagged(self):
        from src.models.debate import statistical_skeptic
        model = {"home": 0.45, "draw": 0.28, "away": 0.27}
        arg = statistical_skeptic(model, calibration_error=0.1)
        assert arg.position == "challenge"
        assert any("ECE" in p for p in arg.points)


class TestJudgeDebate:
    def test_no_challengers_no_adjustment(self):
        from src.models.debate import DebateArgument, judge_debate
        model = {"home": 0.50, "draw": 0.25, "away": 0.25}
        args = [
            DebateArgument("a", "support", 0.9, ["ok"]),
            DebateArgument("b", "support", 0.8, ["fine"]),
        ]
        v = judge_debate(model, args)
        assert not v.adjustment_applied
        assert v.final_probs == model
        assert v.consistency_score == 1.0

    def test_low_confidence_no_adjustment(self):
        from src.models.debate import DebateArgument, judge_debate
        model = {"home": 0.50, "draw": 0.25, "away": 0.25}
        args = [
            DebateArgument("a", "challenge", 0.2, ["minor"],
                           {"home": 0.48, "draw": 0.26, "away": 0.26}),
        ]
        v = judge_debate(model, args)
        assert not v.adjustment_applied

    def test_high_confidence_adjusts(self):
        from src.models.debate import DebateArgument, judge_debate
        model = {"home": 0.70, "draw": 0.15, "away": 0.15}
        args = [
            DebateArgument("a", "challenge", 0.8, ["big gap"],
                           {"home": 0.45, "draw": 0.28, "away": 0.27}),
            DebateArgument("b", "challenge", 0.6, ["overconfident"],
                           {"home": 0.50, "draw": 0.25, "away": 0.25}),
        ]
        v = judge_debate(model, args)
        assert v.adjustment_applied
        assert v.final_probs["home"] < model["home"]

    def test_final_probs_sum_to_one(self):
        from src.models.debate import DebateArgument, judge_debate
        model = {"home": 0.70, "draw": 0.15, "away": 0.15}
        args = [
            DebateArgument("a", "challenge", 0.9, ["gap"],
                           {"home": 0.45, "draw": 0.28, "away": 0.27}),
        ]
        v = judge_debate(model, args)
        assert sum(v.final_probs.values()) == pytest.approx(1.0, abs=1e-6)


class TestRunDebate:
    def test_full_debate_pipeline(self):
        from src.models.debate import run_debate
        model = {"home": 0.65, "draw": 0.18, "away": 0.17}
        market = {"home": 0.45, "draw": 0.28, "away": 0.27}
        v = run_debate(model, market)
        assert len(v.arguments) == 2
        assert sum(v.final_probs.values()) == pytest.approx(1.0, abs=1e-6)

    def test_agreement_no_adjustment(self):
        from src.models.debate import run_debate
        model = {"home": 0.46, "draw": 0.27, "away": 0.27}
        market = {"home": 0.45, "draw": 0.28, "away": 0.27}
        v = run_debate(model, market)
        assert v.final_probs == model


class TestSelfConsistencyBaseline:
    def test_returns_valid_probs(self):
        from src.models.debate import self_consistency_baseline
        model = {"home": 0.50, "draw": 0.25, "away": 0.25}
        out = self_consistency_baseline(model)
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(v > 0 for v in out.values())

    def test_close_to_original(self):
        from src.models.debate import self_consistency_baseline
        model = {"home": 0.50, "draw": 0.25, "away": 0.25}
        out = self_consistency_baseline(model, noise_std=0.01)
        for k in model:
            assert abs(out[k] - model[k]) < 0.05

    def test_deterministic_with_seed(self):
        from src.models.debate import self_consistency_baseline
        model = {"home": 0.50, "draw": 0.25, "away": 0.25}
        a = self_consistency_baseline(model, seed=7)
        b = self_consistency_baseline(model, seed=7)
        for k in model:
            assert a[k] == pytest.approx(b[k])


# --------------------------------------------------------------------------
# C5: disagreement board
# --------------------------------------------------------------------------

class TestDisagreementBoard:
    def _board_with_entries(self):
        from src.models.disagreement import DisagreementBoard
        board = DisagreementBoard()
        board.add("m1", "Arsenal", "Chelsea",
                  {"home": 0.60, "draw": 0.22, "away": 0.18},
                  {"home": 0.45, "draw": 0.28, "away": 0.27})
        board.add("m2", "Liverpool", "ManCity",
                  {"home": 0.35, "draw": 0.30, "away": 0.35},
                  {"home": 0.33, "draw": 0.28, "away": 0.39})
        board.add("m3", "Spurs", "Everton",
                  {"home": 0.55, "draw": 0.25, "away": 0.20},
                  {"home": 0.50, "draw": 0.26, "away": 0.24})
        return board

    def test_add_entry(self):
        from src.models.disagreement import DisagreementBoard
        board = DisagreementBoard()
        e = board.add("m1", "A", "B",
                      {"home": 0.60, "draw": 0.20, "away": 0.20},
                      {"home": 0.45, "draw": 0.28, "away": 0.27})
        assert e.max_diff == pytest.approx(0.15)
        assert e.max_diff_outcome == "home"
        assert len(board.entries) == 1

    def test_leaderboard_ranked(self):
        board = self._board_with_entries()
        lb = board.leaderboard()
        assert len(lb) == 3
        assert lb[0]["max_diff"] >= lb[1]["max_diff"]
        assert lb[1]["max_diff"] >= lb[2]["max_diff"]

    def test_leaderboard_top_n(self):
        board = self._board_with_entries()
        lb = board.leaderboard(top_n=2)
        assert len(lb) == 2

    def test_resolve_scores_both(self):
        board = self._board_with_entries()
        e = board.resolve("m1", "home")
        assert e is not None
        assert e.resolved
        assert e.model_log_loss is not None
        assert e.market_log_loss is not None
        # model gave 0.60 to home, market gave 0.45 -> model wins
        assert e.model_correct

    def test_resolve_removes_from_leaderboard(self):
        board = self._board_with_entries()
        board.resolve("m1", "home")
        lb = board.leaderboard()
        assert all(r["match_id"] != "m1" for r in lb)

    def test_resolution_summary(self):
        board = self._board_with_entries()
        board.resolve("m1", "home")
        board.resolve("m2", "away")
        s = board.resolution_summary()
        assert s["total"] == 2
        assert s["model_wins"] + s["market_wins"] + s["ties"] == 2

    def test_empty_summary(self):
        from src.models.disagreement import DisagreementBoard
        board = DisagreementBoard()
        s = board.resolution_summary()
        assert s["total"] == 0

    def test_biggest_wins_and_misses(self):
        board = self._board_with_entries()
        board.resolve("m1", "home")  # model wins (gave 0.60 vs 0.45)
        board.resolve("m2", "away")  # market wins (gave 0.39 vs 0.35)
        board.resolve("m3", "draw")
        wins = board.biggest_wins()
        misses = board.biggest_misses()
        assert isinstance(wins, list)
        assert isinstance(misses, list)

    def test_double_resolve_no_op(self):
        board = self._board_with_entries()
        board.resolve("m1", "home")
        e = board.resolve("m1", "away")
        assert e.actual_outcome == "home"  # first resolution sticks

    def test_resolve_unknown_returns_none(self):
        from src.models.disagreement import DisagreementBoard
        board = DisagreementBoard()
        assert board.resolve("nonexistent", "home") is None

    def test_model_edge_sign(self):
        from src.models.disagreement import DisagreementBoard
        board = DisagreementBoard()
        e = board.add("m1", "A", "B",
                      {"home": 0.60, "draw": 0.20, "away": 0.20},
                      {"home": 0.45, "draw": 0.28, "away": 0.27})
        assert e.model_edge > 0  # model sees more home value than market

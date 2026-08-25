"""Tests for M5: expanded features (B3), grid reconciliation (B4a),
and log-opinion pool ensemble (B6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------
# B3: expanded features
# --------------------------------------------------------------------------

class TestEloFeatures:
    def _make_results(self):
        return pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03",
                     "2024-01-04", "2024-01-05"],
            "home_team": ["A", "B", "A", "C", "B"],
            "away_team": ["B", "C", "C", "A", "A"],
            "home_score": [2, 1, 0, 1, 3],
            "away_score": [1, 0, 0, 2, 1],
            "neutral": [False, False, False, False, False],
        })

    def test_elo_features_shape(self):
        from src.features.expanded import elo_features
        results = self._make_results()
        matches = pd.DataFrame({
            "home_team": ["A", "B"],
            "away_team": ["C", "A"],
            "kickoff_utc": ["2024-01-06", "2024-01-07"],
        })
        out = elo_features(matches, results)
        assert out.shape == (2, 3)
        assert list(out.columns) == ["elo_home", "elo_away", "elo_diff"]

    def test_elo_diff_sign(self):
        from src.features.expanded import elo_features
        results = self._make_results()
        # A beat B in the first match, so A should have higher Elo
        matches = pd.DataFrame({
            "home_team": ["A"],
            "away_team": ["B"],
            "kickoff_utc": ["2024-01-02"],
        })
        out = elo_features(matches, results)
        assert out.iloc[0]["elo_diff"] > 0  # A ahead after beating B

    def test_elo_walk_forward(self):
        """Elo for a match uses only results before that match's date."""
        from src.features.expanded import elo_features
        results = self._make_results()
        matches = pd.DataFrame({
            "home_team": ["A"],
            "away_team": ["B"],
            "kickoff_utc": ["2024-01-01"],  # before any result
        })
        out = elo_features(matches, results)
        # both should be at base Elo
        assert out.iloc[0]["elo_home"] == pytest.approx(1500.0)
        assert out.iloc[0]["elo_away"] == pytest.approx(1500.0)
        assert out.iloc[0]["elo_diff"] == pytest.approx(0.0)


class TestBookDisagreement:
    def test_basic_disagreement(self):
        from src.features.expanded import book_disagreement_features
        odds = pd.DataFrame([{
            "B365H": 2.0, "B365D": 3.5, "B365A": 4.0,
            "BWH": 2.1, "BWD": 3.3, "BWA": 3.8,
        }])
        cols = [("B365H", "B365D", "B365A"), ("BWH", "BWD", "BWA")]
        out = book_disagreement_features(odds, book_columns=cols)
        assert out.shape == (1, 4)
        assert out.iloc[0]["book_std_home"] >= 0
        assert out.iloc[0]["overround"] >= 0

    def test_missing_book_zeros(self):
        from src.features.expanded import book_disagreement_features
        odds = pd.DataFrame([{
            "B365H": np.nan, "B365D": np.nan, "B365A": np.nan,
        }])
        cols = [("B365H", "B365D", "B365A")]
        out = book_disagreement_features(odds, book_columns=cols)
        assert out.iloc[0]["book_std_home"] == 0.0
        assert out.iloc[0]["overround"] == 0.0

    def test_identical_books_zero_std(self):
        from src.features.expanded import book_disagreement_features
        odds = pd.DataFrame([{
            "B365H": 2.5, "B365D": 3.0, "B365A": 3.0,
            "BWH": 2.5, "BWD": 3.0, "BWA": 3.0,
        }])
        cols = [("B365H", "B365D", "B365A"), ("BWH", "BWD", "BWA")]
        out = book_disagreement_features(odds, book_columns=cols)
        assert out.iloc[0]["book_std_home"] == pytest.approx(0.0, abs=1e-10)


class TestHomeAwaySplitForm:
    def test_split_form_output(self):
        from src.features.expanded import home_away_split_form
        from src.features.team_form import FORM_STATS
        rows = []
        for i in range(20):
            rows.append({
                "match_id": i,
                "team": "TeamA",
                "kickoff_utc": f"2024-01-{i+1:02d}",
                "is_home": i % 2 == 0,
                **{s: np.random.default_rng(i).random() for s in FORM_STATS},
            })
        df = pd.DataFrame(rows)
        out = home_away_split_form(df, window=5)
        assert "match_id" in out.columns
        assert f"split_form_{FORM_STATS[0]}" in out.columns
        assert len(out) == 20

    def test_split_form_nan_fallback(self):
        """With fewer than 3 same-venue matches, should return NaN."""
        from src.features.expanded import home_away_split_form
        from src.features.team_form import FORM_STATS
        rows = [
            {"match_id": 0, "team": "X", "kickoff_utc": "2024-01-01",
             "is_home": True, **{s: 1.0 for s in FORM_STATS}},
            {"match_id": 1, "team": "X", "kickoff_utc": "2024-01-02",
             "is_home": True, **{s: 1.0 for s in FORM_STATS}},
        ]
        df = pd.DataFrame(rows)
        out = home_away_split_form(df, window=5)
        # first match has no prior same-venue, second has only 1
        assert np.isnan(out.iloc[0][f"split_form_{FORM_STATS[0]}"])


# --------------------------------------------------------------------------
# B4a: grid reconciliation
# --------------------------------------------------------------------------

class TestGridOutcome:
    def test_grid_outcome_sums_to_one(self):
        from src.models.reconcile import grid_outcome
        p = grid_outcome(1.5, 1.0)
        total = p["home"] + p["draw"] + p["away"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_grid_outcome_home_advantage(self):
        from src.models.reconcile import grid_outcome
        p = grid_outcome(2.0, 0.8)
        assert p["home"] > p["away"]

    def test_grid_outcome_symmetric(self):
        from src.models.reconcile import grid_outcome
        p = grid_outcome(1.0, 1.0, rho=0.0)
        assert p["home"] == pytest.approx(p["away"], abs=1e-6)


class TestReconcile:
    def test_reconcile_grid_when_close(self):
        from src.models.reconcile import grid_outcome, reconcile
        g = grid_outcome(1.5, 1.0)
        # head matches grid within epsilon
        head = {k: v + 0.01 for k, v in g.items()}
        total = sum(head.values())
        head = {k: v / total for k, v in head.items()}
        result = reconcile(head, 1.5, 1.0, epsilon=0.05)
        assert result["reconciled_via"] == "grid"
        assert result["max_diff"] <= 0.05

    def test_reconcile_blend_when_far(self):
        from src.models.reconcile import reconcile
        head = {"home": 0.80, "draw": 0.10, "away": 0.10}
        result = reconcile(head, 1.0, 1.0, epsilon=0.05)
        assert result["reconciled_via"] == "blend"
        assert len(result["discrepancies"]) > 0

    def test_reconcile_blend_sums_to_one(self):
        from src.models.reconcile import reconcile
        head = {"home": 0.80, "draw": 0.10, "away": 0.10}
        result = reconcile(head, 1.0, 1.0)
        rec = result["reconciled"]
        total = rec["home"] + rec["draw"] + rec["away"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_reconcile_returns_grid_and_head(self):
        from src.models.reconcile import reconcile
        head = {"home": 0.50, "draw": 0.25, "away": 0.25}
        result = reconcile(head, 1.3, 0.9)
        assert "grid_probs" in result
        assert "head_probs" in result
        assert result["head_probs"] == head

    def test_reconcile_blend_weight(self):
        from src.models.reconcile import reconcile
        head = {"home": 0.80, "draw": 0.10, "away": 0.10}
        r1 = reconcile(head, 1.0, 1.0, blend_weight=0.9)
        r2 = reconcile(head, 1.0, 1.0, blend_weight=0.1)
        # higher blend_weight -> closer to grid
        from src.models.reconcile import grid_outcome
        g = grid_outcome(1.0, 1.0)
        diff1 = abs(r1["reconciled"]["home"] - g["home"])
        diff2 = abs(r2["reconciled"]["home"] - g["home"])
        assert diff1 < diff2


# --------------------------------------------------------------------------
# B6: log-opinion pool ensemble
# --------------------------------------------------------------------------

class TestLogOpinionPool:
    def _make_members(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        m1 = rng.dirichlet([3, 1, 1], size=n)
        m2 = rng.dirichlet([1, 3, 1], size=n)
        m3 = rng.dirichlet([2, 2, 2], size=n)
        y = np.array([rng.choice(3, p=p) for p in m1])
        return [m1, m2, m3], y

    def test_pool_sums_to_one(self):
        from src.models.ensemble import log_opinion_pool
        members, _ = self._make_members()
        w = np.array([0.5, 0.3, 0.2])
        combined = log_opinion_pool(members, w)
        np.testing.assert_allclose(combined.sum(axis=1), 1.0, atol=1e-6)

    def test_pool_shape(self):
        from src.models.ensemble import log_opinion_pool
        members, _ = self._make_members(n=100)
        w = np.array([0.4, 0.4, 0.2])
        combined = log_opinion_pool(members, w)
        assert combined.shape == (100, 3)

    def test_equal_weights_geometric_mean(self):
        from src.models.ensemble import log_opinion_pool
        rng = np.random.default_rng(99)
        m1 = rng.dirichlet([2, 2, 2], size=50)
        m2 = rng.dirichlet([2, 2, 2], size=50)
        w = np.array([0.5, 0.5])
        combined = log_opinion_pool([m1, m2], w)
        # log-opinion pool with equal weights = geometric mean (normalized)
        geo = np.sqrt(m1 * m2)
        geo_norm = geo / geo.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(combined, geo_norm, atol=1e-5)

    def test_single_member_identity(self):
        from src.models.ensemble import log_opinion_pool
        rng = np.random.default_rng(7)
        m = rng.dirichlet([3, 2, 1], size=30)
        combined = log_opinion_pool([m], np.array([1.0]))
        np.testing.assert_allclose(combined, m, atol=1e-6)

    def test_zero_weight_ignored(self):
        from src.models.ensemble import log_opinion_pool
        rng = np.random.default_rng(5)
        m1 = rng.dirichlet([3, 2, 1], size=30)
        m2 = rng.dirichlet([1, 1, 5], size=30)
        combined = log_opinion_pool([m1, m2], np.array([1.0, 0.0]))
        np.testing.assert_allclose(combined, m1, atol=1e-6)


class TestFitEnsembleWeights:
    def test_fit_returns_valid_weights(self):
        from src.models.ensemble import fit_ensemble_weights
        rng = np.random.default_rng(42)
        m1 = rng.dirichlet([3, 1, 1], size=200)
        m2 = rng.dirichlet([1, 3, 1], size=200)
        y = np.array([rng.choice(3, p=p) for p in m1])
        w = fit_ensemble_weights([m1, m2], y)
        assert w.shape == (2,)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= 0).all()

    def test_fit_favors_better_member(self):
        from src.models.ensemble import fit_ensemble_weights
        rng = np.random.default_rng(11)
        n = 400
        true_probs = rng.dirichlet([3, 2, 1], size=n)
        y = np.array([rng.choice(3, p=p) for p in true_probs])
        good = true_probs * 0.9 + 0.1 / 3  # close to true
        bad = rng.dirichlet([1, 1, 1], size=n)  # uninformative
        w = fit_ensemble_weights([good, bad], y)
        assert w[0] > w[1]  # good member gets higher weight

    def test_fit_improves_log_loss(self):
        from src.models.ensemble import fit_ensemble_weights, log_opinion_pool
        rng = np.random.default_rng(33)
        n = 300
        m1 = rng.dirichlet([3, 1, 1], size=n)
        m2 = rng.dirichlet([1, 2, 2], size=n)
        y = np.array([rng.choice(3, p=p) for p in m1])
        w = fit_ensemble_weights([m1, m2], y)
        combined = log_opinion_pool([m1, m2], w)
        # ensemble LL should be at least as good as uniform weights
        uniform = log_opinion_pool([m1, m2], np.array([0.5, 0.5]))
        ll_opt = np.mean(np.log(combined[np.arange(n), y] + 1e-12))
        ll_uni = np.mean(np.log(uniform[np.arange(n), y] + 1e-12))
        assert ll_opt >= ll_uni - 0.01


class TestLogOpinionPoolClass:
    def test_fit_and_predict(self):
        from src.models.ensemble import LogOpinionPool
        rng = np.random.default_rng(42)
        m1 = rng.dirichlet([3, 1, 1], size=200)
        m2 = rng.dirichlet([1, 3, 1], size=200)
        y = np.array([rng.choice(3, p=p) for p in m1])
        pool = LogOpinionPool(["model_a", "model_b"])
        pool.fit([m1, m2], y)
        out = pool.predict([m1, m2])
        assert out.shape == (200, 3)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_summary(self):
        from src.models.ensemble import LogOpinionPool
        pool = LogOpinionPool(
            ["market", "gbm", "grid"],
            weights=np.array([0.4, 0.35, 0.25]),
        )
        s = pool.summary()
        assert set(s.keys()) == {"market", "gbm", "grid"}
        assert sum(s.values()) == pytest.approx(1.0, abs=1e-6)

    def test_default_uniform_weights(self):
        from src.models.ensemble import LogOpinionPool
        pool = LogOpinionPool(["a", "b", "c"])
        np.testing.assert_allclose(
            pool.weights, np.array([1/3, 1/3, 1/3]), atol=1e-10,
        )

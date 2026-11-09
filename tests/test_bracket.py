"""Tests: Elo ratings and the knockout-bracket projection engine.

Elo and bracket structure are tested on a deterministic synthetic
round-robin (no network). A separate network-gated test sanity-checks the
real women's data (Spain/USA in the top tier).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.bracket import ROLE_LINEUP, simulate_bracket
from src.models.ratings import compute_elo, rank_by_elo


def _round_robin(n_teams: int = 18) -> pd.DataFrame:
    """Teams named T00..; higher index is strictly stronger (wins by 1).
    Double round-robin across recent dates."""
    rows = []
    day = pd.Timestamp("2025-01-01")
    teams = [f"T{i:02d}" for i in range(n_teams)]
    for i in range(n_teams):
        for j in range(n_teams):
            if i == j:
                continue
            # stronger (higher index) team scores more
            hs, as_ = (2, 0) if i > j else (0, 2)
            rows.append({
                "date": str((day).date()), "home_team": teams[i],
                "away_team": teams[j], "home_score": hs, "away_score": as_,
                "neutral": True,
            })
            day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- elo

def test_elo_recovers_strength_order() -> None:
    elo = compute_elo(_round_robin(12))
    ranked = [t for t, _ in rank_by_elo(elo, top_n=12, active_since="2024-01-01",
                                        min_matches=5)]
    # strongest team (highest index) should top the ranking
    assert ranked[0] == "T11"
    assert ranked[-1] == "T00"
    # ratings strictly decrease with the seed order
    ratings = [elo.ratings[t] for t in ranked]
    assert ratings == sorted(ratings, reverse=True)


def test_elo_expected_goals_favours_stronger() -> None:
    elo = compute_elo(_round_robin(12))
    mu_strong, mu_weak = elo.expected_goals("T11", "T00", neutral=True)
    assert mu_strong > mu_weak
    # symmetric on neutral ground
    a, b = elo.expected_goals("T05", "T05", neutral=True)
    assert a == pytest.approx(b)


def test_elo_home_advantage() -> None:
    elo = compute_elo(_round_robin(12))
    neutral = elo.expected_goals("T05", "T06", neutral=True)
    at_home = elo.expected_goals("T05", "T06", neutral=False)
    assert at_home[0] - at_home[1] > neutral[0] - neutral[1]  # home lifts supremacy


# --------------------------------------------------------------- bracket

@pytest.fixture(scope="module")
def synthetic_bracket() -> dict:
    return simulate_bracket(_round_robin(20), active_since="2024-01-01")


def test_bracket_structure(synthetic_bracket: dict) -> None:
    b = synthetic_bracket
    assert [r["round"] for r in b["rounds"]] == [
        "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
    assert len(b["seeding"]) == 16
    assert len(b["rounds"][0]["matches"]) == 8
    assert len(b["rounds"][-1]["matches"]) == 1
    # seeds ordered by Elo (descending)
    elos = [s["elo"] for s in b["seeding"]]
    assert elos == sorted(elos, reverse=True)
    # standard seeding: top seed faces the weakest in the field in Ro16
    first = b["rounds"][0]["matches"][0]
    assert first["seeds"] == {"home": 1, "away": 16}


def test_bracket_champion_is_top_seed_in_deterministic_field(synthetic_bracket) -> None:
    # strongest team wins a strictly-ordered field
    assert synthetic_bracket["champion"] == synthetic_bracket["seeding"][0]["team"]


def test_each_match_has_scorers_assists_timing_and_evidence(synthetic_bracket) -> None:
    m = synthetic_bracket["rounds"][0]["matches"][0]
    assert set(m) >= {"expected_goals", "outcome_90", "advance",
                      "top_scorelines", "goal_timing_bands", "first_scorer",
                      "headline_scenario", "evidence", "projected_winner"}
    assert sum(m["outcome_90"].values()) == pytest.approx(1.0, abs=1e-6)
    # scorers/assists present at role level with minutes
    for g in m["headline_scenario"]["goals"]:
        assert "scorer_role" in g and 0 <= g["minute"] <= 90
        assert g["scorer_role"] in {r["player"] for r in ROLE_LINEUP}
    # drill-down evidence exposes Elo + the xG the grid was built on
    ev = m["evidence"]
    assert "elo" in ev["home_rating"] and "model_xg" in ev
    assert ev["elo_difference"] == pytest.approx(
        ev["home_rating"]["elo"] - ev["away_rating"]["elo"], abs=0.1)


def test_advance_probabilities_normalized(synthetic_bracket) -> None:
    for rnd in synthetic_bracket["rounds"]:
        for m in rnd["matches"]:
            assert sum(m["advance"].values()) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------ real data (gated)

def test_real_womens_field_is_credible() -> None:
    try:
        from src.data.womens_international import fetch_results

        elo = compute_elo(fetch_results())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"women's data unavailable: {exc}")
    top = [t for t, _ in rank_by_elo(elo, top_n=8, active_since="2024-06-01")]
    # the perennial powers must surface near the top on real data
    assert "Spain" in top and "United States" in top

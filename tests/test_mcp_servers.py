"""Tests: artifact bundle, prediction composer, and the three MCP servers.

Server tools are tested through their underlying logic functions (the
FastMCP decorators add transport, not behavior) plus a registration check
that every expected tool is exposed with a docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

import scripts.build_demo_artifacts as demo
from mcp_servers.common import ttl_cache, with_as_of
from mcp_servers.data_server import server as data_srv
from mcp_servers.data_server.backend import parse_match_id
from mcp_servers.ml_server import server as ml_srv
from mcp_servers.news_server import server as news_srv
from src.models.artifact import ArtifactBundle

MATCH_ID = "ARS-MCI-2026-07-18"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> ArtifactBundle:
    root = tmp_path_factory.mktemp("artifacts")
    old = demo.ARTIFACT_ROOT
    demo.ARTIFACT_ROOT = root
    try:
        demo.build("v-test")
    finally:
        demo.ARTIFACT_ROOT = old
    loaded = ArtifactBundle.load(root)
    ml_srv._bundle = loaded  # point the server at the test bundle
    ml_srv._prediction_store.clear()
    return loaded


def _demo_context(**overrides) -> dict:
    ctx = {
        "odds_imp_home": 0.42, "odds_imp_draw": 0.26, "odds_imp_away": 0.32,
        "form_xg_for_home": 1.7, "form_xg_against_home": 1.0,
        "form_xg_for_away": 1.5, "form_xg_against_away": 1.1,
        "availability_home": 0.92, "availability_away": 1.0,
        "sentiment_home": 0.2, "sentiment_away": -0.1,
        "neutral_venue": 1.0,
    }
    ctx.update(overrides)
    return ctx


# ------------------------------------------------------------------ common

def test_with_as_of_stamps_but_never_overwrites() -> None:
    assert "as_of" in with_as_of({})
    assert with_as_of({"as_of": "2024-01-01T00:00:00+00:00"})["as_of"].startswith("2024")


def test_ttl_cache_caches_and_expires() -> None:
    calls = {"n": 0}

    @ttl_cache(seconds=1000)
    def fn(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert fn(2) == 4 and fn(2) == 4
    assert calls["n"] == 1
    fn.cache_clear()
    fn(2)
    assert calls["n"] == 2


# ------------------------------------------------------------- data server

def test_data_server_tools_deterministic_and_stamped() -> None:
    s1, s2 = data_srv._team_stats("ARS", 10), data_srv._team_stats("ARS", 10)
    assert s1 == s2  # cached AND seeded
    assert s1["form_xg_for"] > 0 and "as_of" in s1

    odds = data_srv._live_odds(MATCH_ID, "h2h")
    probs = [s["implied_prob_vigfree"] for s in odds["selections"]]
    assert sum(probs) == pytest.approx(1.0, abs=1e-3)
    assert all(s["decimal_odds"] > 1.0 for s in odds["selections"])

    ctx = data_srv._fixture_context(MATCH_ID)
    assert ctx["home_team"] == "ARS" and ctx["away_team"] == "MCI"
    assert ctx["knockout"] == (ctx["stage"] != "group")


def test_bad_match_id_rejected() -> None:
    with pytest.raises(ValueError, match="HOME-AWAY"):
        parse_match_id("nonsense")


# --------------------------------------------------------------- ml server

def test_predict_match_full_payload(bundle: ArtifactBundle) -> None:
    result = ml_srv.run_predict(MATCH_ID, _demo_context(
        players_home=[
            {"player": "Striker", "xg_share": 0.5, "xa_share": 0.2,
             "exp_minutes": 90, "availability": 1.0},
            {"player": "Mid", "xg_share": 0.5, "xa_share": 0.8,
             "exp_minutes": 90, "availability": 1.0},
        ],
        market_quotes=[{"market": "h2h", "selection": "home",
                        "model_prob": 0.0, "market_prob": 0.40,
                        "decimal_odds": 2.4}],
        knockout=True,
    ))
    mo = result["match_outcome"]
    assert mo["home"] + mo["draw"] + mo["away"] == pytest.approx(1.0, abs=1e-6)
    assert set(mo["conformal_set"]) <= {"home", "draw", "away"}
    assert result["exact_score"]["top_scorelines"][0]["prob"] > 0
    fs = result["event_sequence"]["first_scorer"]
    assert sum(fs.values()) == pytest.approx(1.0, abs=1e-9)
    assert result["knockout"]["advance"]["home"] + result["knockout"]["advance"]["away"] == pytest.approx(1.0)
    assert len(result["player_props"]["home"]) == 2
    assert "as_of" in result


def test_predict_refuses_on_missing_features(bundle: ArtifactBundle) -> None:
    ctx = _demo_context()
    del ctx["availability_away"]
    with pytest.raises(ValueError, match="REFUSED.*availability_away"):
        ml_srv.run_predict(MATCH_ID, ctx)


def test_explain_prediction_after_predict(bundle: ArtifactBundle) -> None:
    ml_srv.run_predict(MATCH_ID, _demo_context())
    exp = ml_srv.run_explain(MATCH_ID, top_k=3)
    assert exp["explained_class"] in ("home", "draw", "away")
    assert len(exp["top_features"]) == 3
    mags = [abs(f["contribution"]) for f in exp["top_features"]]
    assert mags == sorted(mags, reverse=True)


def test_explain_without_predict_fails(bundle: ArtifactBundle) -> None:
    with pytest.raises(ValueError, match="predict_match first"):
        ml_srv.run_explain("NEVER-SEEN-2026-01-01")


# ------------------------------------------------------------- news server

def test_availability_tool_hard_features() -> None:
    news_srv._availability.cache_clear()
    report = news_srv._availability("ARS")
    by_player = {p["player"]: p for p in report["players"]}
    assert by_player["Gabriel Jesus"]["status"] == "out"
    assert by_player["Martin Odegaard"]["availability_pct"] == 0.75
    assert report["availability_index"] < 1.0
    assert all(len(p["evidence"]) <= 160 for p in report["players"])


def test_sentiment_tool_bounded_and_stamped() -> None:
    news_srv._sentiment.cache_clear()
    result = news_srv._sentiment("MCI")
    assert -1.0 <= result["score"] <= 1.0
    assert result["n_articles"] == 2
    assert result["sources"] == ["demo-wire"]


def test_unknown_team_rejected() -> None:
    with pytest.raises(ValueError, match="unknown team"):
        news_srv._availability("XXX")


# ---------------------------------------------------- tool registration

@pytest.mark.parametrize("srv,expected", [
    (data_srv.server, {"get_team_stats", "get_live_odds", "get_h2h",
                       "get_fixture_context"}),
    (ml_srv.server, {"predict_match", "explain_prediction", "get_model_card"}),
    (news_srv.server, {"get_availability_report", "analyze_team_sentiment"}),
])
def test_all_tools_registered_with_descriptions(srv, expected) -> None:
    import anyio

    tools = anyio.run(srv.list_tools)
    names = {t.name for t in tools}
    assert expected <= names
    assert all(t.description for t in tools)

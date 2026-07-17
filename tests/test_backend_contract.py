"""Contract tests across DataBackend implementations.

The architecture claim is that providers are drop-in swappable behind the
protocol with zero tool-code changes. These tests hold every backend to the
same output contract; a new provider either satisfies it or fails loudly.

The football-data backend needs its season CSVs (downloaded once, then
cached in data/raw/football_data_uk/); the fixture skips cleanly when
neither cache nor network is available.
"""

from __future__ import annotations

import pytest

from mcp_servers.data_server.backend import DemoBackend, FootballDataBackend


def _football_data_backend() -> FootballDataBackend:
    try:
        return FootballDataBackend(start_years=(2023, 2024))
    except Exception as exc:  # noqa: BLE001 — offline and cacheless
        pytest.skip(f"football-data.co.uk unavailable: {exc}")


@pytest.fixture(params=["demo", "football_data"])
def backend(request):
    if request.param == "demo":
        return DemoBackend()
    return _football_data_backend()


def test_team_stats_contract(backend) -> None:
    stats = backend.team_stats("ARS", window=10)
    assert set(stats) >= {"form_xg_for", "form_xg_against", "rest_days",
                          "matches_in_window", "window"}
    assert stats["form_xg_for"] > 0
    assert stats["form_xg_against"] > 0
    assert stats["rest_days"] >= 0
    assert 1 <= stats["matches_in_window"] <= 10


def test_team_stats_unknown_team_fails_loudly(backend) -> None:
    if isinstance(backend, DemoBackend):
        pytest.skip("demo backend synthesizes stats for any team id")
    with pytest.raises(ValueError, match="unknown team"):
        backend.team_stats("ZZZ", window=5)


def test_h2h_contract(backend) -> None:
    rec = backend.h2h("ARS", "MCI")
    assert set(rec) >= {"meetings", "a_wins", "draws", "b_wins"}
    assert rec["meetings"] == rec["a_wins"] + rec["draws"] + rec["b_wins"]
    assert rec["meetings"] >= 0


def test_historical_backend_declares_missing_capabilities() -> None:
    """No silent stubs: unsupported tools raise with guidance."""
    backend = _football_data_backend()
    with pytest.raises(ValueError, match="live odds"):
        backend.live_odds("ARS-MCI-2026-07-18")
    with pytest.raises(ValueError, match="player-level"):
        backend.squad_props("ARS")
    with pytest.raises(ValueError, match="fixture"):
        backend.fixture_context("ARS-MCI-2026-07-18")


def test_real_form_is_not_the_demo_form() -> None:
    """The two backends must disagree — real data is not seeded noise."""
    real = _football_data_backend().team_stats("ARS", window=10)
    demo = DemoBackend().team_stats("ARS", window=10)
    assert real["form_xg_for"] != pytest.approx(demo["form_xg_for"])
    assert real.get("source") == "football-data.co.uk"

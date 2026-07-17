"""Unit tests: edge/EV math, Kelly, tiers, conformal cap, rationale."""

from __future__ import annotations

import pytest

from src.models.suggestions import Driver, MarketQuote, make_suggestions


def _quote(p_model=0.50, p_market=0.40, odds=2.40, market="h2h", sel="home"):
    return MarketQuote(market=market, selection=sel, model_prob=p_model,
                       market_prob=p_market, decimal_odds=odds)


def test_ev_and_edge_math() -> None:
    (s,) = make_suggestions([_quote()])
    assert s.edge == pytest.approx(0.10)
    assert s.ev == pytest.approx(0.50 * 1.40 - 0.50)  # 0.20
    assert s.flagged and s.tier == "high"


def test_kelly_stake_fractional() -> None:
    (s,) = make_suggestions([_quote()], kelly_fraction=0.25)
    full_kelly = (0.50 * 1.40 - 0.50) / 1.40
    assert s.kelly_stake == pytest.approx(0.25 * full_kelly)


def test_negative_ev_never_flagged() -> None:
    (s,) = make_suggestions([_quote(p_model=0.30, p_market=0.40, odds=2.40)])
    assert not s.flagged
    assert s.tier == "none" and s.kelly_stake == 0.0 and s.rationale == ""


def test_threshold_boundary() -> None:
    # ev = p(o-1) - (1-p); pick p so ev just below threshold 0.03
    (s,) = make_suggestions([_quote(p_model=0.43, p_market=0.42, odds=2.395)])
    assert s.ev < 0.03 and not s.flagged


def test_tiers_from_edge() -> None:
    tiers = {"high": 0.08, "medium": 0.05, "low": 0.03}
    med = make_suggestions([_quote(p_model=0.46, p_market=0.40)], tiers=tiers)[0]
    assert med.tier == "medium"


def test_conformal_set_caps_tier() -> None:
    (s,) = make_suggestions(
        [_quote()], h2h_conformal_set=["draw", "away"]  # home excluded
    )
    assert s.flagged and s.tier == "low"
    assert "conformal" in s.rationale


def test_rationale_cites_strongest_drivers() -> None:
    drivers = {"h2h/home": [
        Driver("availability", -0.02, "Away starting striker ruled out 2h ago."),
        Driver("form_xg", 0.01, "Home xG trend +0.4 over last 5."),
        Driver("minor", 0.001, "Should not appear."),
    ]}
    (s,) = make_suggestions([_quote()], drivers=drivers)
    assert "ruled out 2h ago" in s.rationale
    assert "xG trend" in s.rationale
    assert "Should not appear" not in s.rationale
    assert "+10.0% edge" in s.rationale


def test_flagged_sort_first_by_ev() -> None:
    out = make_suggestions([
        _quote(p_model=0.30, p_market=0.40),            # not flagged
        _quote(p_model=0.46, p_market=0.40, sel="draw"), # flagged, small ev
        _quote(p_model=0.55, p_market=0.40, sel="away"), # flagged, big ev
    ])
    assert [s.selection for s in out] == ["away", "draw", "home"]

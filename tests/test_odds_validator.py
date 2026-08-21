"""Tests for the computer-use odds validator."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pytest

from agents.odds_validator import (
    CapturedPage,
    ExtractedOdds,
    MockCapture,
    OddsExtractor,
    OddsValidationAgent,
    OddsValidator,
    ValidationReport,
)
from envs.real_market import Fixture


# ── Helpers ──────────────────────────────────────────────────────────

def _make_fixtures() -> list[Fixture]:
    return [
        Fixture(
            date=date(2024, 8, 17), match_id="MUN-FUL-2024-08-17",
            home="Man United", away="Fulham", ftr="H",
            odds_h=1.67, odds_d=4.10, odds_a=5.00,
            close_h=1.65, close_d=4.20, close_a=5.28,
            odds_source="pinnacle",
        ),
        Fixture(
            date=date(2024, 8, 17), match_id="IPS-LIV-2024-08-17",
            home="Ipswich", away="Liverpool", ftr="A",
            odds_h=8.14, odds_d=6.09, odds_a=1.34,
            close_h=8.14, close_d=6.09, close_a=1.34,
            odds_source="pinnacle",
        ),
        Fixture(
            date=date(2024, 8, 17), match_id="ARS-WOL-2024-08-17",
            home="Arsenal", away="Wolves", ftr="H",
            odds_h=1.15, odds_d=9.05, odds_a=18.76,
            close_h=1.15, close_d=9.05, close_a=18.76,
            odds_source="pinnacle",
        ),
    ]


# ── MockCapture ──────────────────────────────────────────────────────

def test_mock_capture_returns_all_fixtures():
    fixtures = _make_fixtures()
    capture = MockCapture(fixtures=fixtures)
    page = capture.capture("https://example.com/odds")
    assert page.content_type == "application/json"
    assert page.metadata["n_fixtures"] == 3


def test_mock_capture_with_drift():
    fixtures = _make_fixtures()
    capture = MockCapture(fixtures=fixtures, drift_pct=0.05)
    page = capture.capture()
    extractor = OddsExtractor()
    extracted = extractor.extract(page)
    assert len(extracted) == 3
    for ext, ref in zip(extracted, fixtures):
        assert ext.odds_h != ref.odds_h or ext.odds_d != ref.odds_d


def test_mock_capture_drop_fixtures():
    fixtures = _make_fixtures()
    capture = MockCapture(fixtures=fixtures, drop_indices=[1])
    page = capture.capture()
    extractor = OddsExtractor()
    extracted = extractor.extract(page)
    assert len(extracted) == 2


def test_mock_capture_with_anomaly():
    fixtures = _make_fixtures()
    capture = MockCapture(fixtures=fixtures, anomaly_indices=[0])
    page = capture.capture()
    extractor = OddsExtractor()
    extracted = extractor.extract(page)
    assert extracted[0].odds_h == 0.5  # anomalous


# ── OddsExtractor ────────────────────────────────────────────────────

def test_extractor_parses_json():
    fixtures = _make_fixtures()
    capture = MockCapture(fixtures=fixtures)
    page = capture.capture()
    extractor = OddsExtractor()
    extracted = extractor.extract(page)
    assert len(extracted) == 3
    assert extracted[0].match_id == "MUN-FUL-2024-08-17"
    assert extracted[0].home == "Man United"


def test_extractor_html_returns_empty():
    """HTML extraction is a placeholder -- returns empty list."""
    page = CapturedPage(
        url="https://example.com",
        captured_at=datetime.now(timezone.utc),
        content=b"<html></html>",
        content_type="text/html",
    )
    extractor = OddsExtractor()
    assert extractor.extract(page) == []


def test_extractor_screenshot_returns_empty():
    """Screenshot extraction is a placeholder -- returns empty list."""
    page = CapturedPage(
        url="https://example.com",
        captured_at=datetime.now(timezone.utc),
        content=b"\x89PNG",
        content_type="image/png",
    )
    extractor = OddsExtractor()
    assert extractor.extract(page) == []


# ── OddsValidator ────────────────────────────────────────────────────

def test_validator_perfect_match():
    """Identical odds should pass validation."""
    fixtures = _make_fixtures()
    now = datetime.now(timezone.utc)
    extracted = [
        ExtractedOdds(
            match_id=f.match_id, home=f.home, away=f.away,
            odds_h=f.odds_h, odds_d=f.odds_d, odds_a=f.odds_a,
            source="test", extracted_at=now,
        )
        for f in fixtures
    ]
    validator = OddsValidator()
    report = validator.validate(extracted, fixtures, reference_timestamp=now)
    assert report.passed is True
    assert report.n_valid == 3
    assert report.n_drift == 0
    assert report.n_anomaly == 0
    assert report.n_missing == 0
    assert report.max_drift == 0.0


def test_validator_detects_drift():
    """Odds with >10% drift should be flagged."""
    fixtures = _make_fixtures()
    now = datetime.now(timezone.utc)
    extracted = [
        ExtractedOdds(
            match_id=fixtures[0].match_id,
            home=fixtures[0].home, away=fixtures[0].away,
            odds_h=fixtures[0].odds_h * 1.15,  # 15% drift
            odds_d=fixtures[0].odds_d,
            odds_a=fixtures[0].odds_a,
            source="test", extracted_at=now,
        ),
    ]
    validator = OddsValidator(drift_threshold=0.10)
    report = validator.validate(extracted, fixtures[:1], reference_timestamp=now)
    assert report.n_drift == 1
    assert report.passed is False


def test_validator_detects_anomaly():
    """Odds below 1.01 should be flagged as anomalous."""
    fixtures = _make_fixtures()
    now = datetime.now(timezone.utc)
    extracted = [
        ExtractedOdds(
            match_id=fixtures[0].match_id,
            home=fixtures[0].home, away=fixtures[0].away,
            odds_h=0.50,  # impossible odds
            odds_d=fixtures[0].odds_d,
            odds_a=fixtures[0].odds_a,
            source="test", extracted_at=now,
        ),
    ]
    validator = OddsValidator()
    report = validator.validate(extracted, fixtures[:1], reference_timestamp=now)
    assert report.n_anomaly == 1
    assert report.passed is False


def test_validator_detects_missing():
    """Reference fixtures not in extraction should be flagged as missing."""
    fixtures = _make_fixtures()
    now = datetime.now(timezone.utc)
    extracted = [
        ExtractedOdds(
            match_id=fixtures[0].match_id,
            home=fixtures[0].home, away=fixtures[0].away,
            odds_h=fixtures[0].odds_h, odds_d=fixtures[0].odds_d,
            odds_a=fixtures[0].odds_a,
            source="test", extracted_at=now,
        ),
    ]
    validator = OddsValidator()
    report = validator.validate(extracted, fixtures, reference_timestamp=now)
    assert report.n_missing == 2
    assert report.passed is False


def test_validator_staleness():
    """Data older than the staleness limit should fail validation."""
    fixtures = _make_fixtures()
    now = datetime.now(timezone.utc)
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    extracted = [
        ExtractedOdds(
            match_id=fixtures[0].match_id,
            home=fixtures[0].home, away=fixtures[0].away,
            odds_h=fixtures[0].odds_h, odds_d=fixtures[0].odds_d,
            odds_a=fixtures[0].odds_a,
            source="test", extracted_at=now,
        ),
    ]
    validator = OddsValidator(staleness_limit_seconds=3600)
    report = validator.validate(extracted, fixtures[:1], reference_timestamp=old)
    assert report.staleness_seconds > 3600
    assert report.passed is False


def test_validator_summary_keys():
    """Summary dict must contain all required keys."""
    fixtures = _make_fixtures()
    now = datetime.now(timezone.utc)
    extracted = [
        ExtractedOdds(
            match_id=f.match_id, home=f.home, away=f.away,
            odds_h=f.odds_h, odds_d=f.odds_d, odds_a=f.odds_a,
            source="test", extracted_at=now,
        )
        for f in fixtures
    ]
    validator = OddsValidator()
    report = validator.validate(extracted, fixtures, reference_timestamp=now)
    summary = report.summary()
    required = {
        "n_extracted", "n_reference", "n_matched", "n_valid",
        "n_drift", "n_anomaly", "n_missing", "mean_drift",
        "max_drift", "staleness_seconds", "passed",
    }
    assert required.issubset(summary.keys())


# ── Full pipeline (OddsValidationAgent) ──────────────────────────────

def test_agent_pipeline_pass():
    """Full pipeline with matching data should pass."""
    fixtures = _make_fixtures()
    capture = MockCapture(fixtures=fixtures)
    agent = OddsValidationAgent(capture=capture)
    report = agent.run(
        url="https://example.com/odds",
        reference=fixtures,
        reference_timestamp=datetime.now(timezone.utc),
    )
    assert report.passed is True
    assert report.n_matched == 3


def test_agent_pipeline_drift():
    """Full pipeline with drifted data should detect drift."""
    fixtures = _make_fixtures()
    capture = MockCapture(fixtures=fixtures, drift_pct=0.20)
    agent = OddsValidationAgent(
        capture=capture,
        validator=OddsValidator(drift_threshold=0.05),
    )
    report = agent.run(
        url="https://example.com/odds",
        reference=fixtures,
        reference_timestamp=datetime.now(timezone.utc),
    )
    assert report.n_drift > 0


def test_agent_pipeline_missing():
    """Full pipeline with dropped fixtures should detect missing."""
    fixtures = _make_fixtures()
    capture = MockCapture(fixtures=fixtures, drop_indices=[0, 2])
    agent = OddsValidationAgent(capture=capture)
    report = agent.run(
        url="https://example.com/odds",
        reference=fixtures,
        reference_timestamp=datetime.now(timezone.utc),
    )
    assert report.n_missing == 2
    assert report.passed is False

"""Computer-use odds validator: capture, extract, validate.

Demonstrates the Halluminate-style computer-use agent pattern:
a three-stage pipeline that captures a web page, extracts structured
data from the capture using a vision-language model, and validates the
extracted data against a trusted reference source.

Architecture:
    PageCapture (protocol) -> CapturedPage
    OddsExtractor           -> list[ExtractedOdds]
    OddsValidator           -> ValidationReport

Two capture implementations:
    MockCapture   -- returns cached fixture data for testing
    BrowserCapture -- uses Playwright for real browser automation (optional)

The validator detects:
    - Price drift:    |extracted - reference| / reference > threshold
    - Stale data:     capture timestamp vs reference timestamp exceeds limit
    - Missing matches: fixtures in reference but absent from capture
    - Anomalous odds:  odds outside reasonable range [1.01, 100]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np


# ── Data types ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class CapturedPage:
    """Result of capturing a web page."""
    url: str
    captured_at: datetime
    content: bytes
    content_type: str  # "image/png", "text/html"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedOdds:
    """Odds extracted from a captured page."""
    match_id: str
    home: str
    away: str
    odds_h: float
    odds_d: float
    odds_a: float
    source: str
    extracted_at: datetime


@dataclass
class MatchValidation:
    """Validation result for one match."""
    match_id: str
    home: str
    away: str
    status: str  # "valid", "drift", "anomaly", "missing"
    drift_h: float | None = None
    drift_d: float | None = None
    drift_a: float | None = None
    max_drift: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregate validation results."""
    n_extracted: int
    n_reference: int
    n_matched: int
    n_valid: int
    n_drift: int
    n_anomaly: int
    n_missing: int
    mean_drift: float
    max_drift: float
    staleness_seconds: float
    matches: list[MatchValidation]
    passed: bool

    def summary(self) -> dict[str, Any]:
        return {
            "n_extracted": self.n_extracted,
            "n_reference": self.n_reference,
            "n_matched": self.n_matched,
            "n_valid": self.n_valid,
            "n_drift": self.n_drift,
            "n_anomaly": self.n_anomaly,
            "n_missing": self.n_missing,
            "mean_drift": round(self.mean_drift, 4),
            "max_drift": round(self.max_drift, 4),
            "staleness_seconds": round(self.staleness_seconds, 1),
            "passed": self.passed,
        }


# ── Capture protocol ─────────────────────────────────────────────────

class PageCapture(Protocol):
    """Protocol for web page capture implementations."""

    def capture(self, url: str) -> CapturedPage:
        """Capture a web page and return its content."""
        ...


# ── Mock capture ─────────────────────────────────────────────────────

class MockCapture:
    """Test-friendly capture that returns cached fixture data.

    Simulates browser capture by converting Fixture objects into a
    CapturedPage with JSON content.  Supports injecting price drift
    and anomalies for testing the validator.
    """

    def __init__(
        self,
        fixtures: list[Any] | None = None,
        drift_pct: float = 0.0,
        anomaly_indices: list[int] | None = None,
        drop_indices: list[int] | None = None,
    ):
        self.fixtures = fixtures or []
        self.drift_pct = drift_pct
        self.anomaly_indices = set(anomaly_indices or [])
        self.drop_indices = set(drop_indices or [])

    def capture(self, url: str = "") -> CapturedPage:
        import json
        now = datetime.now(timezone.utc)
        extracted: list[dict] = []

        for i, fix in enumerate(self.fixtures):
            if i in self.drop_indices:
                continue

            odds_h = fix.odds_h
            odds_d = fix.odds_d
            odds_a = fix.odds_a

            if self.drift_pct != 0.0:
                rng = np.random.default_rng(i)
                odds_h *= 1.0 + rng.uniform(-self.drift_pct, self.drift_pct)
                odds_d *= 1.0 + rng.uniform(-self.drift_pct, self.drift_pct)
                odds_a *= 1.0 + rng.uniform(-self.drift_pct, self.drift_pct)

            if i in self.anomaly_indices:
                odds_h = 0.5  # impossible odds

            extracted.append({
                "match_id": fix.match_id,
                "home": fix.home,
                "away": fix.away,
                "odds_h": round(odds_h, 2),
                "odds_d": round(odds_d, 2),
                "odds_a": round(odds_a, 2),
            })

        content = json.dumps(extracted).encode("utf-8")
        return CapturedPage(
            url=url,
            captured_at=now,
            content=content,
            content_type="application/json",
            metadata={"source": "mock", "n_fixtures": len(extracted)},
        )


# ── Odds extractor ───────────────────────────────────────────────────

class OddsExtractor:
    """Extract structured odds from a captured page.

    In production, this would call a VLM (e.g., Claude with vision)
    to extract odds from a screenshot.  The structured extraction
    interface is the same regardless of the underlying method:

        VLM extraction:  screenshot -> Claude vision -> structured JSON
        HTML parsing:    HTML content -> CSS selectors -> structured JSON
        Mock:            JSON content -> direct parse

    The extractor validates that the output conforms to the expected
    schema before returning.
    """

    def extract(self, page: CapturedPage) -> list[ExtractedOdds]:
        import json

        if page.content_type == "application/json":
            return self._from_json(page)

        if page.content_type == "text/html":
            return self._from_html(page)

        if page.content_type == "image/png":
            return self._from_screenshot(page)

        return []

    def _from_json(self, page: CapturedPage) -> list[ExtractedOdds]:
        """Parse pre-structured JSON (mock capture path)."""
        import json
        data = json.loads(page.content.decode("utf-8"))
        results = []
        for item in data:
            results.append(ExtractedOdds(
                match_id=item["match_id"],
                home=item["home"],
                away=item["away"],
                odds_h=float(item["odds_h"]),
                odds_d=float(item["odds_d"]),
                odds_a=float(item["odds_a"]),
                source="mock",
                extracted_at=page.captured_at,
            ))
        return results

    def _from_html(self, page: CapturedPage) -> list[ExtractedOdds]:
        """Parse odds from HTML using structured selectors.

        In production: use CSS selectors tuned to the target site.
        Halluminate's Westworld uses deterministic structured selectors
        for exactly this reason -- they survive across resets.
        """
        return []

    def _from_screenshot(self, page: CapturedPage) -> list[ExtractedOdds]:
        """Extract odds from a screenshot using a VLM.

        Production flow:
            1. Send screenshot to Claude with vision
            2. Prompt: "Extract all match odds from this screenshot.
               Return as JSON: [{match_id, home, away, odds_h, odds_d, odds_a}]"
            3. Parse the structured response
            4. Validate schema

        This method is a placeholder -- the VLM call requires an API
        key and network access.  The architecture is the artifact.
        """
        return []


# ── Validator ────────────────────────────────────────────────────────

class OddsValidator:
    """Validate extracted odds against a trusted reference.

    Checks:
    1. Price drift: relative difference between extracted and reference odds.
    2. Anomalies: odds outside the valid range [1.01, 100].
    3. Missing matches: reference fixtures not found in the extraction.
    4. Staleness: time gap between capture and reference data.
    """

    def __init__(
        self,
        drift_threshold: float = 0.10,
        min_odds: float = 1.01,
        max_odds: float = 100.0,
        staleness_limit_seconds: float = 3600.0,
    ):
        self.drift_threshold = drift_threshold
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.staleness_limit = staleness_limit_seconds

    def validate(
        self,
        extracted: list[ExtractedOdds],
        reference: list[Any],
        reference_timestamp: datetime | None = None,
    ) -> ValidationReport:
        """Compare extracted odds against reference fixtures.

        Args:
            extracted: odds from the page capture.
            reference: Fixture objects from the data server.
            reference_timestamp: when the reference data was fetched.
        """
        ref_by_id = {f.match_id: f for f in reference}
        ext_by_id = {e.match_id: e for e in extracted}

        matches: list[MatchValidation] = []
        all_drifts: list[float] = []

        for ext in extracted:
            if ext.match_id not in ref_by_id:
                matches.append(MatchValidation(
                    match_id=ext.match_id,
                    home=ext.home,
                    away=ext.away,
                    status="anomaly",
                    details={"reason": "not in reference data"},
                ))
                continue

            ref = ref_by_id[ext.match_id]

            if not self._valid_odds(ext.odds_h, ext.odds_d, ext.odds_a):
                matches.append(MatchValidation(
                    match_id=ext.match_id,
                    home=ext.home,
                    away=ext.away,
                    status="anomaly",
                    details={
                        "reason": "odds outside valid range",
                        "odds": [ext.odds_h, ext.odds_d, ext.odds_a],
                    },
                ))
                continue

            drift_h = self._relative_drift(ext.odds_h, ref.odds_h)
            drift_d = self._relative_drift(ext.odds_d, ref.odds_d)
            drift_a = self._relative_drift(ext.odds_a, ref.odds_a)
            max_d = max(abs(drift_h), abs(drift_d), abs(drift_a))

            all_drifts.append(max_d)

            status = "drift" if max_d > self.drift_threshold else "valid"
            matches.append(MatchValidation(
                match_id=ext.match_id,
                home=ext.home,
                away=ext.away,
                status=status,
                drift_h=round(drift_h, 4),
                drift_d=round(drift_d, 4),
                drift_a=round(drift_a, 4),
                max_drift=round(max_d, 4),
            ))

        for ref_id, ref in ref_by_id.items():
            if ref_id not in ext_by_id:
                matches.append(MatchValidation(
                    match_id=ref_id,
                    home=ref.home,
                    away=ref.away,
                    status="missing",
                    details={"reason": "not found in capture"},
                ))

        n_valid = sum(1 for m in matches if m.status == "valid")
        n_drift = sum(1 for m in matches if m.status == "drift")
        n_anomaly = sum(1 for m in matches if m.status == "anomaly")
        n_missing = sum(1 for m in matches if m.status == "missing")

        staleness = 0.0
        if reference_timestamp and extracted:
            staleness = abs(
                (extracted[0].extracted_at - reference_timestamp).total_seconds()
            )

        passed = (
            n_anomaly == 0
            and n_missing == 0
            and (not all_drifts or max(all_drifts) <= self.drift_threshold)
            and staleness <= self.staleness_limit
        )

        return ValidationReport(
            n_extracted=len(extracted),
            n_reference=len(reference),
            n_matched=len(ext_by_id.keys() & ref_by_id.keys()),
            n_valid=n_valid,
            n_drift=n_drift,
            n_anomaly=n_anomaly,
            n_missing=n_missing,
            mean_drift=float(np.mean(all_drifts)) if all_drifts else 0.0,
            max_drift=float(max(all_drifts)) if all_drifts else 0.0,
            staleness_seconds=staleness,
            matches=matches,
            passed=passed,
        )

    def _relative_drift(self, extracted: float, reference: float) -> float:
        if reference <= 0:
            return 0.0
        return (extracted - reference) / reference

    def _valid_odds(self, *odds: float) -> bool:
        return all(self.min_odds <= o <= self.max_odds for o in odds)


# ── Orchestrator ─────────────────────────────────────────────────────

class OddsValidationAgent:
    """Full capture-extract-validate pipeline.

    This is the top-level agent that a human or automated system
    triggers.  It orchestrates the three stages and returns a
    structured report.

    In the Halluminate pattern:
    - State-based verification:  did the page load correctly?
    - Component-level:           are individual odds valid?
    - Ground-truth matching:     do extracted odds match the reference?
    """

    def __init__(
        self,
        capture: PageCapture,
        extractor: OddsExtractor | None = None,
        validator: OddsValidator | None = None,
    ):
        self.capture = capture
        self.extractor = extractor or OddsExtractor()
        self.validator = validator or OddsValidator()

    def run(
        self,
        url: str,
        reference: list[Any],
        reference_timestamp: datetime | None = None,
    ) -> ValidationReport:
        """Execute the full validation pipeline.

        1. Capture the page (screenshot or HTML)
        2. Extract structured odds from the capture
        3. Validate extracted odds against reference data
        """
        page = self.capture.capture(url)
        extracted = self.extractor.extract(page)
        report = self.validator.validate(
            extracted, reference,
            reference_timestamp=reference_timestamp,
        )
        return report

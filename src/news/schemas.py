"""Schema-validated news structures — the trust boundary for scraped text.

Everything scraped from the web (RSS bodies, NewsAPI articles) is untrusted
input. The parsers in this package reduce it to the typed records below:
enums, floats, timestamps, and player names canonicalized against a squad
list we control. Raw article text never travels past this module — downstream
features, models, and (in Phase B) the agent's context only ever see these
records, which is the first line of defense against indirect prompt injection
and MCP tool-poisoning via planted article content.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
MAX_SNIPPET_CHARS = 160


def sanitize_text(raw: str, max_chars: int = 4000) -> str:
    """Normalize untrusted text before pattern matching: strip HTML tags,
    control characters, and zero-width/bidi characters used to hide
    instructions; collapse whitespace; hard-cap length."""
    text = _TAG_RE.sub(" ", raw)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" and ch not in "​‌‍⁠﻿"
    )
    return _WS_RE.sub(" ", text).strip()[:max_chars]


class PlayerStatus(str, Enum):
    OUT = "out"                          # injury or suspension, ruled out
    DOUBTFUL = "doubtful"                # fitness doubt, availability_pct applies
    FIT = "fit"                          # returned / passed fit, not yet confirmed
    CONFIRMED_STARTER = "confirmed_starter"
    CONFIRMED_BENCH = "confirmed_bench"


# Default availability probability per status when no percentage was parsed.
DEFAULT_AVAILABILITY: dict[PlayerStatus, float] = {
    PlayerStatus.OUT: 0.0,
    PlayerStatus.DOUBTFUL: 0.5,
    PlayerStatus.FIT: 0.9,
    PlayerStatus.CONFIRMED_STARTER: 1.0,
    PlayerStatus.CONFIRMED_BENCH: 0.25,  # may appear as a sub
}


class NewsItem(BaseModel):
    """One raw article/report, as delivered by a NewsProvider."""

    team: str
    title: str
    body: str
    source: str
    published_utc: datetime

    @field_validator("published_utc")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("published_utc must be timezone-aware UTC")
        return v


class PlayerAvailability(BaseModel):
    """The hard availability fact extracted for one player.

    ``player`` is always a canonical name from the squad list we supplied to
    the parser — never a string lifted from article text. ``evidence`` is the
    name of the rule/source that fired plus a sanitized, hard-capped snippet
    for human audit only; it is never fed to models or LLM prompts.
    """

    player: str
    team: str
    status: PlayerStatus
    availability_pct: float = Field(ge=0.0, le=1.0)
    is_expected_starter: bool = False
    source: str
    as_of_utc: datetime
    evidence: str = Field(default="", max_length=MAX_SNIPPET_CHARS)


class TeamAvailabilityReport(BaseModel):
    """All availability facts for one team at one moment, plus derived features."""

    team: str
    as_of_utc: datetime
    players: list[PlayerAvailability]
    n_out: int
    n_doubtful: int
    starters_out: int
    availability_index: float = Field(
        ge=0.0, le=1.0,
        description="Minutes-weighted mean availability of the expected XI; "
                    "1.0 = full-strength squad.",
    )


class TeamSentiment(BaseModel):
    """Soft morale signal for one team, aggregated with recency decay."""

    team: str
    as_of_utc: datetime
    score: float = Field(ge=-1.0, le=1.0)
    n_articles: int
    half_life_days: float
    sources: list[str]

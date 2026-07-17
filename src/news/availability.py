"""Structured availability parser: injuries, suspensions, confirmed lineups.

Two extraction paths feed one report:

1. ``from_structured`` — clean provider payloads (API-Football injuries and
   lineups endpoints). Preferred whenever present.
2. ``from_text`` — rule-based extraction over sanitized article text, for
   facts that break in the press before the APIs update ("ruled out two
   hours ago"). Player mentions are matched only against the squad list we
   pass in, so no string from an untrusted article ever becomes a player
   identity downstream.

When both paths report the same player, structured data wins unless the text
report is *newer* and *more severe* (e.g. API still lists DOUBTFUL, press
says ruled out) — recency of hard news is exactly the market-lag signal the
suggestion layer wants.

An injured starter is a lineup change, not a sentiment signal: the report's
``availability_index`` directly multiplies team-strength inputs, and the
per-player records feed the player-prop allocation in Step 4.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable

from src.news.schemas import (
    DEFAULT_AVAILABILITY,
    MAX_SNIPPET_CHARS,
    NewsItem,
    PlayerAvailability,
    PlayerStatus,
    TeamAvailabilityReport,
    sanitize_text,
)

# Ordered: first match wins. Severity is used when merging text vs structured.
_TEXT_RULES: list[tuple[str, re.Pattern[str], PlayerStatus]] = [
    ("ruled_out", re.compile(
        r"\b(ruled out|out for|sidelined|will miss|misses (the|this)|"
        r"suspended|out injured|season.ending)\b", re.I), PlayerStatus.OUT),
    ("doubtful", re.compile(
        r"\b(doubt(ful)?|fitness (test|doubt)|50.50|questionable|"
        r"race against time|game.time decision)\b", re.I), PlayerStatus.DOUBTFUL),
    ("returning", re.compile(
        r"\b(passed fit|returns to|back in training|declared fit|"
        r"available again|back in contention)\b", re.I), PlayerStatus.FIT),
]

_PCT_RE = re.compile(r"(\d{1,3})\s?(?:%|percent|per cent)", re.I)

_SEVERITY: dict[PlayerStatus, int] = {
    PlayerStatus.CONFIRMED_STARTER: 0,
    PlayerStatus.FIT: 1,
    PlayerStatus.CONFIRMED_BENCH: 2,
    PlayerStatus.DOUBTFUL: 3,
    PlayerStatus.OUT: 4,
}


def from_structured(
    payload: dict[str, Any], team: str, as_of_utc: datetime,
    expected_starters: set[str] | None = None,
) -> list[PlayerAvailability]:
    """Parse an API-Football-style payload.

    Expected shape::

        {"injuries": [{"player": ..., "type": "out"|"doubtful", "pct": 0.3?}],
         "lineup":   {"starting": [...], "bench": [...]}}   # once released

    Unknown players are kept (the provider is trusted for identity), but
    unknown status strings raise rather than defaulting to something playable.
    """
    starters = expected_starters or set()
    records: list[PlayerAvailability] = []

    for inj in payload.get("injuries", []):
        status = PlayerStatus(inj["type"])
        pct = float(inj.get("pct", DEFAULT_AVAILABILITY[status]))
        records.append(PlayerAvailability(
            player=inj["player"], team=team, status=status,
            availability_pct=pct,
            is_expected_starter=inj["player"] in starters,
            source="structured", as_of_utc=as_of_utc,
            evidence=f"provider:{inj.get('reason', 'unspecified')}"[:MAX_SNIPPET_CHARS],
        ))

    lineup = payload.get("lineup") or {}
    for name in lineup.get("starting", []):
        records.append(PlayerAvailability(
            player=name, team=team, status=PlayerStatus.CONFIRMED_STARTER,
            availability_pct=1.0, is_expected_starter=True,
            source="structured", as_of_utc=as_of_utc, evidence="provider:lineup",
        ))
    for name in lineup.get("bench", []):
        records.append(PlayerAvailability(
            player=name, team=team, status=PlayerStatus.CONFIRMED_BENCH,
            availability_pct=DEFAULT_AVAILABILITY[PlayerStatus.CONFIRMED_BENCH],
            is_expected_starter=False,
            source="structured", as_of_utc=as_of_utc, evidence="provider:lineup",
        ))
    return records


def from_text(
    items: Iterable[NewsItem], squad: dict[str, list[str]],
    expected_starters: set[str] | None = None,
) -> list[PlayerAvailability]:
    """Rule-based extraction from sanitized article text.

    ``squad`` maps canonical player name -> aliases (surname, common short
    forms). A rule only produces a record when a squad player is mentioned in
    the same sentence as the pattern — identity always comes from the squad
    list, never from the article.
    """
    starters = expected_starters or set()
    records: list[PlayerAvailability] = []

    for item in items:
        text = sanitize_text(f"{item.title}. {item.body}")
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            for rule_name, pattern, status in _TEXT_RULES:
                if not pattern.search(sentence):
                    continue
                for canonical, aliases in squad.items():
                    names = [canonical, *aliases]
                    if not any(
                        re.search(rf"\b{re.escape(n)}\b", sentence, re.I)
                        for n in names
                    ):
                        continue
                    pct = DEFAULT_AVAILABILITY[status]
                    if status is PlayerStatus.DOUBTFUL:
                        if m := _PCT_RE.search(sentence):
                            pct = min(100, int(m.group(1))) / 100.0
                    records.append(PlayerAvailability(
                        player=canonical, team=item.team, status=status,
                        availability_pct=pct,
                        is_expected_starter=canonical in starters,
                        source=f"text:{item.source}",
                        as_of_utc=item.published_utc,
                        evidence=f"{rule_name}: {sentence[:MAX_SNIPPET_CHARS - 20]}",
                    ))
                break  # first matching rule per sentence
    return records


def merge_reports(
    structured: list[PlayerAvailability], text: list[PlayerAvailability]
) -> list[PlayerAvailability]:
    """One record per player. Structured wins, except a strictly newer AND
    strictly more severe text report overrides it (press beats a stale API)."""
    best: dict[str, PlayerAvailability] = {}
    for rec in structured:
        cur = best.get(rec.player)
        if cur is None or rec.as_of_utc >= cur.as_of_utc:
            best[rec.player] = rec
    for rec in text:
        cur = best.get(rec.player)
        if cur is None:
            best[rec.player] = rec
        elif (
            cur.source == "structured"
            and rec.as_of_utc > cur.as_of_utc
            and _SEVERITY[rec.status] > _SEVERITY[cur.status]
        ):
            best[rec.player] = rec
        elif cur.source.startswith("text:") and (
            rec.as_of_utc > cur.as_of_utc
            or (rec.as_of_utc == cur.as_of_utc
                and _SEVERITY[rec.status] > _SEVERITY[cur.status])
        ):
            best[rec.player] = rec
    return sorted(best.values(), key=lambda r: r.player)


def build_report(
    team: str, as_of_utc: datetime, records: list[PlayerAvailability],
    expected_xi: list[str],
) -> TeamAvailabilityReport:
    """Derive the hard features. ``availability_index`` is the mean
    availability over the expected XI; players with no record count as 1.0."""
    by_player = {r.player: r for r in records}
    xi_avail = [
        by_player[p].availability_pct if p in by_player else 1.0
        for p in expected_xi
    ]
    return TeamAvailabilityReport(
        team=team,
        as_of_utc=as_of_utc,
        players=records,
        n_out=sum(r.status is PlayerStatus.OUT for r in records),
        n_doubtful=sum(r.status is PlayerStatus.DOUBTFUL for r in records),
        starters_out=sum(
            r.status is PlayerStatus.OUT and r.player in set(expected_xi)
            for r in records
        ),
        availability_index=sum(xi_avail) / len(xi_avail) if xi_avail else 1.0,
    )

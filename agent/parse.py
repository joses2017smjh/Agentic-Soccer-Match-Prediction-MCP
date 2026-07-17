"""Natural-language request parsing (the cheap-model tier).

FrugalGPT-style routing: extraction/argument-filling is the easy tier, so it
runs on rules here (and would route to a small LLM, never the synthesis
model, when wired to an API). Output is schema-validated ParsedRequest —
a malformed request fails loudly before any tool is called.
"""

from __future__ import annotations

import re

from agent.state import ParsedRequest

TEAM_ALIASES: dict[str, str] = {
    "arsenal": "ARS", "man city": "MCI", "manchester city": "MCI",
    "city": "MCI", "gunners": "ARS", "liverpool": "LIV", "real madrid": "RMA",
    "barcelona": "BAR", "bayern": "BAY", "psg": "PSG",
}

_VS_RE = re.compile(
    r"(?P<home>[\w .'-]+?)\s+(?:vs\.?|v|against|-)\s+(?P<away>[\w .'-]+?)"
    r"(?=$|[,.?!—–]|\s+[—–-]\s|\s+match|\s+game|\s+on\s)", re.I,
)
_LEAD_WORDS = frozenset({
    "predict", "the", "match", "game", "please", "who", "wins", "will",
    "about", "what", "for", "how", "does", "do",
})
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_MATCH_ID_RE = re.compile(r"\b([A-Z]{2,4}-[A-Z]{2,4}-\d{4}-\d{2}-\d{2})\b")
_STAKE_RE = re.compile(r"\b(bet|stake|value|edge|wager|kelly|suggest)", re.I)
DEFAULT_DATE = "2026-07-18"  # demo fixture date when the request names none


def _team_code(name: str) -> str:
    words = [w for w in name.strip().lower().split() if w]
    # longest alias match anchored at the END ("predict arsenal" → "arsenal")
    for i in range(len(words)):
        candidate = " ".join(words[i:])
        if candidate in TEAM_ALIASES:
            return TEAM_ALIASES[candidate]
    while words and words[0] in _LEAD_WORDS:
        words = words[1:]
    code = re.sub(r"[^a-z]", "", "".join(words))[:3].upper()
    if len(code) < 2:
        raise ValueError(f"cannot derive a team code from {name!r}")
    return code


def parse_request(text: str) -> ParsedRequest:
    wants_stakes = bool(_STAKE_RE.search(text))

    if m := _MATCH_ID_RE.search(text):
        home, away, _ = m.group(1).split("-", 2)
        return ParsedRequest(match_id=m.group(1), home_team=home,
                             away_team=away, wants_stakes=wants_stakes,
                             raw_text=text)

    m = _VS_RE.search(text)
    if not m:
        raise ValueError(
            "could not identify two teams; phrase the request like "
            "'Arsenal vs Man City' or give a match id HOME-AWAY-YYYY-MM-DD"
        )
    home, away = _team_code(m.group("home")), _team_code(m.group("away"))
    date_m = _DATE_RE.search(text)
    date = date_m.group(1) if date_m else DEFAULT_DATE
    return ParsedRequest(
        match_id=f"{home}-{away}-{date}", home_team=home, away_team=away,
        wants_stakes=wants_stakes, raw_text=text,
    )

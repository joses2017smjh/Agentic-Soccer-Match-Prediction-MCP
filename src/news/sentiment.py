"""Soft sentiment layer: per-team morale score with recency decay.

Model choice
------------
FinBERT is trained on financial filings and analyst language; its polarity
cues ("miss", "loss", "downgrade") collide head-on with football vocabulary —
"a heavy loss at home" is routine sports text, not a bearish filing — so it
is explicitly rejected here. The default scorer is
``cardiffnlp/twitter-roberta-base-sentiment-latest``: a RoBERTa model tuned
on ~124M social/news-register posts whose informal, event-driven language is
much closer to sports coverage, and which emits the three-way
negative/neutral/positive distribution we fold into a single [-1, 1] score.
The scorer sits behind a Protocol, so a sports-tuned model or zero-shot LLM
scoring (Phase B, via the news MCP server) can be swapped in without touching
the aggregation logic — and unit tests run on a dependency-free lexicon stub.

Only the *numeric* score and source names leave this module; article text is
sanitized before scoring and never propagated (see src/news/schemas.py).
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Protocol

from src.news.schemas import NewsItem, TeamSentiment, sanitize_text


class SentimentModel(Protocol):
    def score(self, text: str) -> float:
        """Return sentiment in [-1, 1] for one sanitized text."""
        ...


class LexiconSentiment:
    """Tiny deterministic scorer for tests and torch-free environments."""

    _POSITIVE = frozenset({
        "win", "wins", "victory", "unbeaten", "brilliant", "confident",
        "boost", "fit", "returns", "extends", "harmony", "settled",
    })
    _NEGATIVE = frozenset({
        "crisis", "sacked", "sacking", "pressure", "rift", "controversy",
        "injury", "injured", "loss", "losses", "unrest", "protest", "feud",
    })

    def score(self, text: str) -> float:
        words = text.lower().split()
        pos = sum(w.strip(".,!?") in self._POSITIVE for w in words)
        neg = sum(w.strip(".,!?") in self._NEGATIVE for w in words)
        total = pos + neg
        return 0.0 if total == 0 else (pos - neg) / total


class TransformerSentiment:
    """Default production scorer (requires the ``sentiment`` extra)."""

    def __init__(
        self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ) -> None:
        from transformers import pipeline  # lazy: torch is an optional extra

        self._pipe = pipeline(
            "sentiment-analysis", model=model_name, top_k=None, truncation=True
        )

    def score(self, text: str) -> float:
        scores = {d["label"].lower(): d["score"] for d in self._pipe(text)[0]}
        return scores.get("positive", 0.0) - scores.get("negative", 0.0)


def team_sentiment(
    team: str,
    items: Iterable[NewsItem],
    *,
    model: SentimentModel,
    as_of_utc: datetime,
    half_life_days: float = 3.0,
) -> TeamSentiment:
    """Decay-weighted mean sentiment over the team's articles.

    Articles published after ``as_of_utc`` are excluded — the leakage cutoff
    applies to news exactly as it does to odds and stats. Weight halves every
    ``half_life_days``, so a sacking three days ago counts half as much as one
    this morning.
    """
    weighted = total_weight = 0.0
    sources: list[str] = []
    n = 0
    for item in items:
        if item.team != team or item.published_utc > as_of_utc:
            continue
        age_days = (as_of_utc - item.published_utc).total_seconds() / 86400.0
        weight = 0.5 ** (age_days / half_life_days)
        text = sanitize_text(f"{item.title}. {item.body}")
        weighted += weight * model.score(text)
        total_weight += weight
        sources.append(item.source)
        n += 1

    score = weighted / total_weight if total_weight > 0 else 0.0
    return TeamSentiment(
        team=team,
        as_of_utc=as_of_utc,
        score=max(-1.0, min(1.0, score)),
        n_articles=n,
        half_life_days=half_life_days,
        sources=sorted(set(sources)),
    )

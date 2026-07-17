"""Unit tests: sentiment aggregation with recency decay and leakage cutoff."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.news.schemas import NewsItem
from src.news.sentiment import LexiconSentiment, team_sentiment

NOW = datetime(2024, 6, 9, 14, 0, tzinfo=timezone.utc)


def _item(body: str, days_ago: float, team="Arsenal", source="bbc") -> NewsItem:
    return NewsItem(team=team, title="", body=body, source=source,
                    published_utc=NOW - timedelta(days=days_ago))


def test_lexicon_polarity() -> None:
    model = LexiconSentiment()
    assert model.score("brilliant win, squad confident") > 0
    assert model.score("crisis deepens, manager sacked amid unrest") < 0
    assert model.score("the match kicks off at three") == 0.0


def test_recency_decay_weights_recent_news_higher() -> None:
    """Fresh negative news must outweigh older positive news at half_life=1d."""
    items = [
        _item("brilliant win boost", days_ago=3.0),   # weight 0.125
        _item("crisis manager sacked", days_ago=0.0), # weight 1.0
    ]
    result = team_sentiment("Arsenal", items, model=LexiconSentiment(),
                            as_of_utc=NOW, half_life_days=1.0)
    assert result.score == pytest.approx((0.125 * 1.0 + 1.0 * -1.0) / 1.125)
    assert result.n_articles == 2


def test_future_articles_excluded() -> None:
    """The leakage cutoff applies to news too."""
    items = [_item("crisis manager sacked", days_ago=-0.5)]  # published after as_of
    result = team_sentiment("Arsenal", items, model=LexiconSentiment(),
                            as_of_utc=NOW)
    assert result.n_articles == 0
    assert result.score == 0.0


def test_other_teams_ignored_and_sources_deduped() -> None:
    items = [
        _item("brilliant win", 0.1),
        _item("brilliant win", 0.2, source="bbc"),
        _item("crisis sacked", 0.1, team="Chelsea"),
    ]
    result = team_sentiment("Arsenal", items, model=LexiconSentiment(),
                            as_of_utc=NOW)
    assert result.n_articles == 2
    assert result.sources == ["bbc"]
    assert result.score == pytest.approx(1.0)

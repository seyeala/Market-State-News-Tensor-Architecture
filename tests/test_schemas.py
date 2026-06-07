from datetime import datetime, timedelta, timezone

import pytest

from market_state_news_tensor.schemas.events import EntityMention, EventObject, RawNewsItem
from market_state_news_tensor.schemas.market import MarketBar, QuoteBar


def dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 9, minute, tzinfo=timezone.utc)


def test_market_bar_requires_available_after_bar_end() -> None:
    with pytest.raises(ValueError, match="available_at"):
        MarketBar(
            symbol="XYZ",
            bar_start=dt(30),
            bar_end=dt(31),
            open=10,
            high=11,
            low=9,
            close=10,
            volume=100,
            available_at=dt(30),
        )


def test_quote_rejects_crossed_market() -> None:
    with pytest.raises(ValueError, match="ask"):
        QuoteBar(symbol="XYZ", timestamp=dt(30), bid=10.1, ask=10.0, available_at=dt(30))


def test_raw_news_requires_text() -> None:
    with pytest.raises(ValueError, match="headline or body"):
        RawNewsItem(
            news_id="n1",
            source="wire",
            event_time=dt(30),
            publish_time=dt(30),
            ingest_time=dt(31),
            available_at=dt(31),
        )


def test_event_object_rejects_ex_post_metadata() -> None:
    with pytest.raises(ValueError, match="ex-post"):
        EventObject(
            event_id="e1",
            source="wire",
            event_time=dt(30),
            publish_time=dt(30),
            ingest_time=dt(31),
            available_at=dt(31),
            entities=(EntityMention(entity_id="XYZ", name="XYZ Corp"),),
            claim="XYZ raises guidance",
            context_type="fundamental",
            event_type="guidance_raise",
            affected_metrics=("growth",),
            direction=0.8,
            magnitude=0.5,
            uncertainty=0.2,
            horizon="1d",
            source_quality=0.9,
            novelty=1.0,
            evidence=("Company said it raised guidance.",),
            metadata={"ar_5m": 0.1},
        )


def test_valid_quote_helpers() -> None:
    quote = QuoteBar(symbol="XYZ", timestamp=dt(30), bid=10, ask=10.2, available_at=dt(30))

    assert quote.mid_price == pytest.approx(10.1)
    assert quote.proportional_spread == pytest.approx(0.2 / 10.1)

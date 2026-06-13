from datetime import datetime, timezone

import pytest

from market_state_news_tensor.events.ontology import validate_event_against_ontology
from market_state_news_tensor.schemas.events import EntityMention, EventObject


def dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 9, minute, tzinfo=timezone.utc)


def event_object(
    *,
    context_type: str = "fundamental",
    event_type: str = "guidance_raise",
    affected_metrics: tuple[str, ...] = ("growth",),
) -> EventObject:
    return EventObject(
        event_id="e1",
        source="wire",
        event_time=dt(30),
        publish_time=dt(30),
        ingest_time=dt(31),
        available_at=dt(31),
        entities=(EntityMention(entity_id="XYZ", name="XYZ Corp"),),
        claim="XYZ raises guidance",
        context_type=context_type,
        event_type=event_type,
        affected_metrics=affected_metrics,
        direction=0.8,
        magnitude=0.5,
        uncertainty=0.2,
        horizon="1d",
        source_quality=0.9,
        novelty=1.0,
        evidence=("Company said it raised guidance.",),
    )


@pytest.mark.parametrize(
    ("context_type", "event_type", "affected_metrics"),
    [
        ("fundamental", "guidance_raise", ("growth", "revenue")),
        ("macro", "central_bank_decision", ("rates", "liquidity")),
        ("legal/regulatory", "regulatory_approval", ("revenue", "regulatory_risk")),
        ("market-belief", "sentiment_shift", ("sentiment", "momentum")),
    ],
)
def test_valid_event_objects_pass_ontology_validation(
    context_type: str, event_type: str, affected_metrics: tuple[str, ...]
) -> None:
    validate_event_against_ontology(
        event_object(
            context_type=context_type,
            event_type=event_type,
            affected_metrics=affected_metrics,
        )
    )


def test_unknown_context_type_fails() -> None:
    with pytest.raises(ValueError, match="unknown context_type"):
        validate_event_against_ontology(event_object(context_type="weather"))


def test_unknown_event_type_fails() -> None:
    with pytest.raises(ValueError, match="unknown event_type"):
        validate_event_against_ontology(event_object(event_type="dividend_cut"))


def test_invalid_affected_metrics_fail() -> None:
    with pytest.raises(ValueError, match="invalid affected_metrics"):
        validate_event_against_ontology(event_object(affected_metrics=("weather",)))

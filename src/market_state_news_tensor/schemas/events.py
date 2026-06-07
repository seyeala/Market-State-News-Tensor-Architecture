"""News and event schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_SCORE_FIELDS = ("magnitude", "uncertainty", "source_quality", "novelty")
EX_POST_FIELD_NAMES = {
    "ar_5m",
    "ar_15m",
    "ar_60m",
    "ar_1d",
    "delta_volume",
    "delta_volatility",
    "reversal",
    "outcome_available_at",
}


def _require_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RawNewsItem:
    news_id: str
    source: str
    event_time: datetime
    publish_time: datetime
    ingest_time: datetime
    available_at: datetime
    headline: str | None = None
    body: str | None = None
    vendor_received_time: datetime | None = None
    url: str | None = None
    language: str = "en"

    def __post_init__(self) -> None:
        if not self.news_id:
            raise ValueError("news_id is required")
        if not self.source:
            raise ValueError("source is required")
        if not (self.headline or self.body):
            raise ValueError("headline or body is required")
        if self.available_at < self.ingest_time:
            raise ValueError("available_at must be at or after ingest_time")


@dataclass(frozen=True)
class EntityMention:
    entity_id: str
    name: str
    relevance: float = 1.0

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id is required")
        if not self.name:
            raise ValueError("name is required")
        _require_probability("relevance", self.relevance)


@dataclass(frozen=True)
class EventObject:
    """Ex-ante structured event; safe for live features after validity checks."""

    event_id: str
    source: str
    event_time: datetime
    publish_time: datetime
    ingest_time: datetime
    available_at: datetime
    entities: tuple[EntityMention, ...]
    claim: str
    context_type: str
    event_type: str
    affected_metrics: tuple[str, ...]
    direction: float
    magnitude: float
    uncertainty: float
    horizon: str
    source_quality: float
    novelty: float
    evidence: tuple[str, ...]
    vendor_received_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        if not self.source:
            raise ValueError("source is required")
        if not self.entities:
            raise ValueError("at least one entity is required")
        if not self.claim:
            raise ValueError("claim is required")
        if not self.context_type:
            raise ValueError("context_type is required")
        if not self.event_type:
            raise ValueError("event_type is required")
        if not self.affected_metrics:
            raise ValueError("affected_metrics is required")
        if not -1.0 <= self.direction <= 1.0:
            raise ValueError("direction must be in [-1, 1]")
        for field_name in _SCORE_FIELDS:
            _require_probability(field_name, getattr(self, field_name))
        if not self.horizon:
            raise ValueError("horizon is required")
        if not self.evidence:
            raise ValueError("evidence is required")
        if self.available_at < self.ingest_time:
            raise ValueError("available_at must be at or after ingest_time")
        leaked = EX_POST_FIELD_NAMES.intersection(self.metadata)
        if leaked:
            raise ValueError(f"ex-post fields are not allowed in EventObject metadata: {sorted(leaked)}")


@dataclass(frozen=True)
class EventOutcome:
    """Ex-post event-study outcome for audit/calibration, not live features."""

    event_id: str
    outcome_available_at: datetime
    ar_5m: float | None = None
    ar_15m: float | None = None
    ar_60m: float | None = None
    ar_1d: float | None = None
    delta_volume: float | None = None
    delta_volatility: float | None = None
    reversal: bool | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")

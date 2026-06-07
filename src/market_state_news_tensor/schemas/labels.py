"""Feature and label schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(frozen=True)
class FeatureRow:
    symbol: str
    decision_time: datetime
    available_at: datetime
    market_features: dict[str, float | None] = field(default_factory=dict)
    context_features: dict[str, float | None] = field(default_factory=dict)
    event_state_features: dict[str, float | None] = field(default_factory=dict)
    belief_features: dict[str, float | None] = field(default_factory=dict)
    disagreement_features: dict[str, float | None] = field(default_factory=dict)
    feature_valid: bool = True

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.available_at > self.decision_time:
            object.__setattr__(self, "feature_valid", False)


@dataclass(frozen=True)
class FixedHorizonLabel:
    symbol: str
    decision_time: datetime
    horizon: timedelta
    return_value: float
    class_label: int | None
    label_start: datetime
    label_end: datetime
    cost: float | None = None
    volatility_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.label_end <= self.label_start:
            raise ValueError("label_end must be after label_start")
        if self.class_label not in (-1, 0, 1, None):
            raise ValueError("class_label must be -1, 0, 1, or None")


@dataclass(frozen=True)
class TripleBarrierLabel:
    symbol: str
    decision_time: datetime
    upper_barrier: float
    lower_barrier: float
    vertical_barrier_time: datetime
    touch_time: datetime
    class_label: int
    label_start: datetime
    label_end: datetime

    def __post_init__(self) -> None:
        if self.upper_barrier <= 0:
            raise ValueError("upper_barrier must be positive")
        if self.lower_barrier <= 0:
            raise ValueError("lower_barrier must be positive")
        if self.class_label not in (-1, 0, 1):
            raise ValueError("class_label must be -1, 0, or 1")
        if self.label_end <= self.label_start:
            raise ValueError("label_end must be after label_start")

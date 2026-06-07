"""Point-in-time validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, TypeVar

from market_state_news_tensor.schemas.events import EX_POST_FIELD_NAMES

T = TypeVar("T")


def is_point_in_time_valid(available_at: datetime, decision_time: datetime) -> bool:
    """Return whether an object was available by the decision timestamp."""

    return available_at <= decision_time


def filter_valid_rows(
    rows: Iterable[T],
    *,
    decision_time_attr: str = "decision_time",
    available_at_attr: str = "available_at",
) -> list[T]:
    """Keep rows whose availability timestamp is no later than decision time."""

    valid: list[T] = []
    for row in rows:
        decision_time = getattr(row, decision_time_attr)
        available_at = getattr(row, available_at_attr)
        if is_point_in_time_valid(available_at, decision_time):
            valid.append(row)
    return valid


def assert_no_ex_post_fields(feature_payload: Mapping[str, Any]) -> None:
    """Reject feature payloads containing ex-post event-outcome fields."""

    leaked = EX_POST_FIELD_NAMES.intersection(feature_payload.keys())
    if leaked:
        raise ValueError(f"ex-post fields are not allowed in live features: {sorted(leaked)}")

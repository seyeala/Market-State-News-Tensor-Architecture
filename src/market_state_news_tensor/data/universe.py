"""Universe filtering helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, TypeVar

from market_state_news_tensor.schemas.universe import UniverseMembership

T = TypeVar("T")


def _symbol(row: Any) -> str:
    if isinstance(row, Mapping):
        return row["symbol"]
    return getattr(row, "symbol")


def _is_active_and_eligible(membership: UniverseMembership, decision_time: datetime) -> bool:
    return (
        membership.start <= decision_time
        and (membership.end is None or decision_time <= membership.end)
        and membership.tradable
        and membership.observable
        and membership.eligible
    )


def filter_universe_asof(
    rows: Iterable[T],
    universe: Iterable[UniverseMembership],
    decision_time: datetime,
) -> list[T]:
    """Return rows whose symbols are eligible universe members at decision time."""

    eligible_symbols = {
        membership.symbol
        for membership in universe
        if _is_active_and_eligible(membership, decision_time)
    }
    return [row for row in rows if _symbol(row) in eligible_symbols]

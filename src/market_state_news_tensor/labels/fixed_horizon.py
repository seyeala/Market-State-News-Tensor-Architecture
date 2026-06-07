"""Fixed-horizon return and trading-class labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from market_state_news_tensor.schemas.labels import FixedHorizonLabel


@dataclass(frozen=True)
class PricePoint:
    symbol: str
    timestamp: datetime
    log_price: float


def make_future_returns(prices: Sequence[PricePoint], horizon: timedelta) -> list[FixedHorizonLabel]:
    """Compute fixed-horizon future log-return labels for aligned price points."""

    by_symbol: dict[str, list[PricePoint]] = {}
    for point in prices:
        by_symbol.setdefault(point.symbol, []).append(point)

    labels: list[FixedHorizonLabel] = []
    for symbol, points in by_symbol.items():
        points = sorted(points, key=lambda point: point.timestamp)
        by_time = {point.timestamp: point for point in points}
        for point in points:
            end_time = point.timestamp + horizon
            future = by_time.get(end_time)
            if future is None:
                continue
            labels.append(
                FixedHorizonLabel(
                    symbol=symbol,
                    decision_time=point.timestamp,
                    horizon=horizon,
                    return_value=future.log_price - point.log_price,
                    class_label=None,
                    label_start=point.timestamp,
                    label_end=end_time,
                )
            )
    return labels


def assign_fixed_horizon_classes(
    labels: Sequence[FixedHorizonLabel],
    *,
    costs: Sequence[float] | float,
    volatility: Sequence[float] | float,
    theta: float,
) -> list[FixedHorizonLabel]:
    """Apply cost/volatility threshold to fixed-horizon return labels."""

    if theta < 0:
        raise ValueError("theta must be non-negative")
    costs_list = [costs] * len(labels) if isinstance(costs, (int, float)) else list(costs)
    vol_list = [volatility] * len(labels) if isinstance(volatility, (int, float)) else list(volatility)
    if len(costs_list) != len(labels) or len(vol_list) != len(labels):
        raise ValueError("costs and volatility must align with labels")

    classed: list[FixedHorizonLabel] = []
    for label, cost, vol in zip(labels, costs_list, vol_list):
        if cost < 0 or vol < 0:
            raise ValueError("cost and volatility must be non-negative")
        threshold = cost + theta * vol
        if label.return_value > threshold:
            class_label = 1
        elif label.return_value < -threshold:
            class_label = -1
        else:
            class_label = 0
        classed.append(
            FixedHorizonLabel(
                symbol=label.symbol,
                decision_time=label.decision_time,
                horizon=label.horizon,
                return_value=label.return_value,
                class_label=class_label,
                label_start=label.label_start,
                label_end=label.label_end,
                cost=cost,
                volatility_threshold=theta * vol,
            )
        )
    return classed

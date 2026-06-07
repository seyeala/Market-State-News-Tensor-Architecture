"""Triple-barrier labels for trade-path outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from market_state_news_tensor.labels.fixed_horizon import PricePoint
from market_state_news_tensor.schemas.labels import TripleBarrierLabel


def find_first_barrier_touch(path_returns: Sequence[float], upper: float, lower: float) -> tuple[int | None, int]:
    """Return the index and class of the first horizontal barrier touch."""

    if upper <= 0 or lower <= 0:
        raise ValueError("barriers must be positive")
    for idx, value in enumerate(path_returns):
        if value > upper:
            return idx, 1
        if value < -lower:
            return idx, -1
    return None, 0


def make_triple_barrier_labels(
    prices: Sequence[PricePoint],
    *,
    max_horizon: timedelta,
    upper_mult: float,
    lower_mult: float,
    costs: Sequence[float] | float,
    volatility: Sequence[float] | float,
) -> list[TripleBarrierLabel]:
    """Create triple-barrier labels from ordered price points."""

    if upper_mult < 0 or lower_mult < 0:
        raise ValueError("barrier multipliers must be non-negative")

    by_symbol: dict[str, list[PricePoint]] = {}
    for point in prices:
        by_symbol.setdefault(point.symbol, []).append(point)

    all_points = [point for symbol_points in by_symbol.values() for point in symbol_points]
    n = len(all_points)
    costs_list = [costs] * n if isinstance(costs, (int, float)) else list(costs)
    vol_list = [volatility] * n if isinstance(volatility, (int, float)) else list(volatility)
    if len(costs_list) != n or len(vol_list) != n:
        raise ValueError("costs and volatility must align with prices")

    keyed_costs = {(point.symbol, point.timestamp): cost for point, cost in zip(all_points, costs_list)}
    keyed_vol = {(point.symbol, point.timestamp): vol for point, vol in zip(all_points, vol_list)}

    labels: list[TripleBarrierLabel] = []
    for symbol, points in by_symbol.items():
        points = sorted(points, key=lambda point: point.timestamp)
        for start_idx, point in enumerate(points):
            vertical_time = point.timestamp + max_horizon
            future_points = [candidate for candidate in points[start_idx + 1 :] if candidate.timestamp <= vertical_time]
            if not future_points or future_points[-1].timestamp < vertical_time:
                continue
            cost = keyed_costs[(point.symbol, point.timestamp)]
            vol = keyed_vol[(point.symbol, point.timestamp)]
            if cost < 0 or vol < 0:
                raise ValueError("cost and volatility must be non-negative")
            upper = cost + upper_mult * vol
            lower = cost + lower_mult * vol
            if upper <= 0 or lower <= 0:
                raise ValueError("computed barriers must be positive")
            path_returns = [future.log_price - point.log_price for future in future_points]
            touch_offset, class_label = find_first_barrier_touch(path_returns, upper, lower)
            if touch_offset is None:
                touch_point = future_points[-1]
            else:
                touch_point = future_points[touch_offset]
            labels.append(
                TripleBarrierLabel(
                    symbol=symbol,
                    decision_time=point.timestamp,
                    upper_barrier=upper,
                    lower_barrier=lower,
                    vertical_barrier_time=vertical_time,
                    touch_time=touch_point.timestamp,
                    class_label=class_label,
                    label_start=point.timestamp,
                    label_end=touch_point.timestamp,
                )
            )
    return labels

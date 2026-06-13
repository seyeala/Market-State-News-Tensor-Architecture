from datetime import datetime, timedelta, timezone

import pytest

from market_state_news_tensor.labels.fixed_horizon import PricePoint, assign_fixed_horizon_classes, make_future_returns
from market_state_news_tensor.labels.triple_barrier import find_first_barrier_touch, make_triple_barrier_labels
from market_state_news_tensor.validation.costs import CostModel


def ts(minute: int) -> datetime:
    return datetime(2026, 1, 1, 9, minute, tzinfo=timezone.utc)


def points(values: list[float]) -> list[PricePoint]:
    return [PricePoint(symbol="XYZ", timestamp=ts(i), log_price=value) for i, value in enumerate(values)]


def test_executable_cost_increases_with_quantity() -> None:
    model = CostModel(spread=0.02, fees=0.001, eta=0.5)

    small = model.executable_cost(quantity=10, volatility=0.01, volume=1000)
    large = model.executable_cost(quantity=100, volatility=0.01, volume=1000)

    assert large > small


def test_executable_cost_increases_with_volatility() -> None:
    model = CostModel(spread=0.02, fees=0.001, eta=0.5)

    low_vol = model.executable_cost(quantity=100, volatility=0.01, volume=1000)
    high_vol = model.executable_cost(quantity=100, volatility=0.03, volume=1000)

    assert high_vol > low_vol


def test_executable_cost_decreases_as_volume_increases() -> None:
    model = CostModel(spread=0.02, fees=0.001, eta=0.5)

    low_volume = model.executable_cost(quantity=100, volatility=0.02, volume=1000)
    high_volume = model.executable_cost(quantity=100, volatility=0.02, volume=10_000)

    assert high_volume < low_volume


def test_fixed_horizon_returns_and_classes() -> None:
    labels = make_future_returns(points([0.0, 0.03, 0.01]), timedelta(minutes=1))
    classed = assign_fixed_horizon_classes(labels, costs=0.005, volatility=0.01, theta=1.0)

    assert [label.return_value for label in labels] == pytest.approx([0.03, -0.02])
    assert [label.class_label for label in classed] == [1, -1]


def test_fixed_horizon_no_trade_class() -> None:
    labels = make_future_returns(points([0.0, 0.005]), timedelta(minutes=1))
    classed = assign_fixed_horizon_classes(labels, costs=0.005, volatility=0.01, theta=1.0)

    assert classed[0].class_label == 0


def test_find_first_barrier_touch() -> None:
    assert find_first_barrier_touch([0.01, 0.03], upper=0.02, lower=0.02) == (1, 1)
    assert find_first_barrier_touch([-0.01, -0.03], upper=0.02, lower=0.02) == (1, -1)
    assert find_first_barrier_touch([0.01, -0.01], upper=0.02, lower=0.02) == (None, 0)


def test_make_triple_barrier_labels() -> None:
    labels = make_triple_barrier_labels(
        points([0.0, 0.01, 0.03, 0.02]),
        max_horizon=timedelta(minutes=2),
        upper_mult=1.0,
        lower_mult=1.0,
        costs=0.001,
        volatility=0.02,
    )

    assert labels[0].class_label == 1
    assert labels[0].touch_time == ts(2)


def test_triple_barrier_lower_first_outcome() -> None:
    labels = make_triple_barrier_labels(
        points([0.0, -0.03, 0.04]),
        max_horizon=timedelta(minutes=2),
        upper_mult=1.0,
        lower_mult=1.0,
        costs=0.0,
        volatility=0.02,
    )

    assert labels[0].class_label == -1
    assert labels[0].touch_time == ts(1)


def test_triple_barrier_no_touch_uses_vertical_barrier_outcome() -> None:
    labels = make_triple_barrier_labels(
        points([0.0, 0.005, 0.01]),
        max_horizon=timedelta(minutes=2),
        upper_mult=1.0,
        lower_mult=1.0,
        costs=0.0,
        volatility=0.02,
    )

    assert labels[0].class_label == 0
    assert labels[0].touch_time == ts(2)
    assert labels[0].label_end == ts(2)


def test_triple_barrier_skips_rows_with_insufficient_future_data() -> None:
    labels = make_triple_barrier_labels(
        points([0.0, 0.01, 0.015]),
        max_horizon=timedelta(minutes=2),
        upper_mult=1.0,
        lower_mult=1.0,
        costs=0.0,
        volatility=0.02,
    )

    assert len(labels) == 1
    assert labels[0].decision_time == ts(0)

    assert make_triple_barrier_labels(
        points([0.0, 0.01]),
        max_horizon=timedelta(minutes=2),
        upper_mult=1.0,
        lower_mult=1.0,
        costs=0.0,
        volatility=0.02,
    ) == []

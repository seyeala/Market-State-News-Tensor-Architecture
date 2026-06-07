from datetime import datetime, timezone

import pytest

from market_state_news_tensor.data.point_in_time import (
    assert_no_ex_post_fields,
    filter_valid_rows,
    is_point_in_time_valid,
)
from market_state_news_tensor.schemas.labels import FeatureRow


def dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 9, minute, tzinfo=timezone.utc)


def test_point_in_time_validity_allows_available_before_decision() -> None:
    assert is_point_in_time_valid(dt(30), dt(31))


def test_point_in_time_validity_rejects_future_availability() -> None:
    assert not is_point_in_time_valid(dt(32), dt(31))


def test_filter_valid_rows_drops_unavailable_rows() -> None:
    rows = [
        FeatureRow(symbol="A", decision_time=dt(31), available_at=dt(30)),
        FeatureRow(symbol="A", decision_time=dt(31), available_at=dt(32)),
    ]

    valid = filter_valid_rows(rows)

    assert len(valid) == 1
    assert valid[0].available_at == dt(30)


def test_ex_post_guard_rejects_outcome_fields() -> None:
    with pytest.raises(ValueError, match="ex-post"):
        assert_no_ex_post_fields({"r_1m": 0.01, "ar_5m": 0.02})

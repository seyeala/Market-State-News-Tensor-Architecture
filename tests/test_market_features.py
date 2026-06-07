from datetime import datetime, timedelta, timezone

import pytest

from market_state_news_tensor.features.market import (
    MARKET_FEATURE_NAMES,
    build_market_feature_rows,
    build_market_tensor,
    compute_log_price,
)
from market_state_news_tensor.schemas.market import MarketBar


def make_bar(idx: int, close: float | None = None) -> MarketBar:
    start = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=idx)
    end = start + timedelta(minutes=1)
    price = float(idx + 100) if close is None else close
    return MarketBar(
        symbol="XYZ",
        bar_start=start,
        bar_end=end,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=1000 + idx,
        available_at=end,
        vwap=price - 0.5,
    )


def test_compute_log_price_uses_mid_when_quotes_available() -> None:
    assert compute_log_price(100, bid=99, ask=101) == pytest.approx(0.0 + __import__("math").log(100))


def test_build_market_feature_rows_uses_past_returns_only() -> None:
    rows = build_market_feature_rows([make_bar(i) for i in range(16)])

    assert rows[0].features["r_1m"] is None
    assert rows[5].features["r_5m"] is not None
    assert rows[15].features["r_15m"] is not None
    assert rows[15].features["sigma_15m"] is not None
    assert rows[14].features["sigma_15m"] is None


def test_build_market_tensor_shape_with_padding() -> None:
    feature_rows = build_market_feature_rows([make_bar(i) for i in range(3)])
    tensor = build_market_tensor(feature_rows, decision_time=feature_rows[-1].timestamp, lookback=5, pad=True)

    assert len(tensor) == 5
    assert len(tensor[0]) == len(MARKET_FEATURE_NAMES)
    assert tensor[0] == [None] * len(MARKET_FEATURE_NAMES)


def test_build_market_tensor_rejects_insufficient_history_without_padding() -> None:
    feature_rows = build_market_feature_rows([make_bar(i) for i in range(3)])

    with pytest.raises(ValueError, match="insufficient"):
        build_market_tensor(feature_rows, decision_time=feature_rows[-1].timestamp, lookback=5)

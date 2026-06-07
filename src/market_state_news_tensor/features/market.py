"""Market feature generation for bar/quote-level data."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from market_state_news_tensor.schemas.market import MarketBar

MARKET_FEATURE_NAMES = (
    "log_price",
    "r_1m",
    "r_5m",
    "r_15m",
    "sigma_15m",
    "sigma_60m",
    "range",
    "log_volume",
    "vwap_dev",
    "spread",
    "tau_sin",
    "tau_cos",
)


def compute_log_price(close: float, bid: float | None = None, ask: float | None = None) -> float:
    """Compute log mid-price when quotes exist, otherwise log close."""

    if bid is not None and ask is not None:
        if bid <= 0 or ask <= 0:
            raise ValueError("bid and ask must be positive")
        if ask < bid:
            raise ValueError("ask must be greater than or equal to bid")
        return math.log((bid + ask) / 2.0)
    if close <= 0:
        raise ValueError("close must be positive")
    return math.log(close)


def _safe_return(log_prices: Sequence[float], idx: int, lag: int) -> float | None:
    if idx - lag < 0:
        return None
    return log_prices[idx] - log_prices[idx - lag]


def _realized_volatility(one_minute_returns: Sequence[float | None], idx: int, window: int) -> float | None:
    if idx - window + 1 < 0:
        return None
    window_values = one_minute_returns[idx - window + 1 : idx + 1]
    if any(value is None for value in window_values):
        return None
    return math.sqrt(sum(float(value) ** 2 for value in window_values))


def _time_of_day(timestamp: datetime) -> tuple[float, float]:
    seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second + timestamp.microsecond / 1_000_000
    angle = 2.0 * math.pi * seconds / 86_400.0
    return math.sin(angle), math.cos(angle)


@dataclass(frozen=True)
class MarketFeatureRow:
    symbol: str
    timestamp: datetime
    available_at: datetime
    features: dict[str, float | None]

    def as_ordered_list(self, feature_names: Sequence[str] = MARKET_FEATURE_NAMES) -> list[float | None]:
        return [self.features.get(name) for name in feature_names]


def build_market_feature_rows(
    bars: Sequence[MarketBar],
    *,
    bids: Sequence[float | None] | None = None,
    asks: Sequence[float | None] | None = None,
) -> list[MarketFeatureRow]:
    """Build ordered one-minute feature rows from bars and optional aligned quotes."""

    sorted_bars = sorted(bars, key=lambda bar: (bar.symbol, bar.bar_end))
    if bids is None:
        bids = [None] * len(sorted_bars)
    if asks is None:
        asks = [None] * len(sorted_bars)
    if len(bids) != len(sorted_bars) or len(asks) != len(sorted_bars):
        raise ValueError("bids and asks must align with bars")

    rows: list[MarketFeatureRow] = []
    by_symbol: dict[str, list[tuple[MarketBar, float | None, float | None]]] = {}
    for bar, bid, ask in zip(sorted_bars, bids, asks):
        by_symbol.setdefault(bar.symbol, []).append((bar, bid, ask))

    for symbol, items in by_symbol.items():
        log_prices = [compute_log_price(bar.close, bid, ask) for bar, bid, ask in items]
        one_minute_returns = [_safe_return(log_prices, idx, 1) for idx in range(len(items))]
        for idx, (bar, bid, ask) in enumerate(items):
            log_price = log_prices[idx]
            spread = None
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
                spread = (ask - bid) / mid
            range_feature = (bar.high - bar.low) / bar.close
            log_volume = math.log1p(bar.volume)
            vwap_dev = None if bar.vwap is None else (bar.close - bar.vwap) / bar.close
            tau_sin, tau_cos = _time_of_day(bar.bar_end)
            features = {
                "log_price": log_price,
                "r_1m": one_minute_returns[idx],
                "r_5m": _safe_return(log_prices, idx, 5),
                "r_15m": _safe_return(log_prices, idx, 15),
                "sigma_15m": _realized_volatility(one_minute_returns, idx, 15),
                "sigma_60m": _realized_volatility(one_minute_returns, idx, 60),
                "range": range_feature,
                "log_volume": log_volume,
                "vwap_dev": vwap_dev,
                "spread": spread,
                "tau_sin": tau_sin,
                "tau_cos": tau_cos,
            }
            rows.append(MarketFeatureRow(symbol=symbol, timestamp=bar.bar_end, available_at=bar.available_at, features=features))
    return rows


def build_market_tensor(
    feature_rows: Sequence[MarketFeatureRow],
    *,
    decision_time: datetime,
    lookback: int = 180,
    feature_names: Sequence[str] = MARKET_FEATURE_NAMES,
    pad: bool = False,
) -> list[list[float | None]]:
    """Build a rolling market tensor ending at or before `decision_time`."""

    eligible = [row for row in feature_rows if row.timestamp <= decision_time and row.available_at <= decision_time]
    eligible.sort(key=lambda row: row.timestamp)
    selected = eligible[-lookback:]
    if len(selected) < lookback and not pad:
        raise ValueError("insufficient history for requested lookback")
    tensor = [row.as_ordered_list(feature_names) for row in selected]
    if pad and len(tensor) < lookback:
        missing = lookback - len(tensor)
        tensor = [[None for _ in feature_names] for _ in range(missing)] + tensor
    return tensor

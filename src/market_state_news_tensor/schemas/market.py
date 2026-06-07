"""Market-data schemas with point-in-time availability metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class MarketBar:
    """OHLCV bar that records when it became usable by a model."""

    symbol: str
    bar_start: datetime
    bar_end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    available_at: datetime
    vwap: float | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.bar_end <= self.bar_start:
            raise ValueError("bar_end must be after bar_start")
        for name in ("open", "high", "low", "close"):
            _require_positive(name, getattr(self, name))
        _require_non_negative("volume", self.volume)
        if self.vwap is not None:
            _require_positive("vwap", self.vwap)
        if self.high < self.low:
            raise ValueError("high must be greater than or equal to low")
        if self.available_at < self.bar_end:
            raise ValueError("available_at must be at or after bar_end")


@dataclass(frozen=True)
class QuoteBar:
    """Best bid/ask quote with helper methods for mid and spread."""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    available_at: datetime
    bid_size: float | None = None
    ask_size: float | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")
        _require_positive("bid", self.bid)
        _require_positive("ask", self.ask)
        if self.ask < self.bid:
            raise ValueError("ask must be greater than or equal to bid")
        if self.bid_size is not None:
            _require_non_negative("bid_size", self.bid_size)
        if self.ask_size is not None:
            _require_non_negative("ask_size", self.ask_size)
        if self.available_at < self.timestamp:
            raise ValueError("available_at must be at or after timestamp")

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def proportional_spread(self) -> float:
        return (self.ask - self.bid) / self.mid_price

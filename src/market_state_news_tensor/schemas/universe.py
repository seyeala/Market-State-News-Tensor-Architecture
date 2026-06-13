"""Tradable universe membership schemas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class UniverseMembership:
    """Point-in-time membership record for a symbol in the trading universe."""

    symbol: str
    start: datetime
    end: datetime | None
    tradable: bool
    observable: bool
    eligible: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.end is not None and self.end < self.start:
            raise ValueError("end must be at or after start")

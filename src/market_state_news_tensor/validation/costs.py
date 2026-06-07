"""Execution-cost helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Simple return-unit transaction-cost model."""

    spread: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    expected_impact: float = 0.0
    eta: float = 0.0

    def __post_init__(self) -> None:
        for name in ("spread", "fees", "slippage", "expected_impact", "eta"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def base_cost(self) -> float:
        return self.spread + self.fees + self.slippage + self.expected_impact

    def executable_cost(self, *, quantity: float, volatility: float, volume: float, spread: float | None = None) -> float:
        if volatility < 0:
            raise ValueError("volatility must be non-negative")
        if volume <= 0:
            raise ValueError("volume must be positive")
        effective_spread = self.spread if spread is None else spread
        if effective_spread < 0:
            raise ValueError("spread must be non-negative")
        impact = self.eta * volatility * math.sqrt(abs(quantity) / volume)
        return effective_spread / 2.0 + self.fees + self.slippage + self.expected_impact + impact

"""Experiment-ledger schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    created_at: datetime
    git_commit: str
    features: dict[str, Any]
    labels: dict[str, Any]
    universe: dict[str, Any]
    split: dict[str, Any]
    hyperparameters: dict[str, Any]
    cost_model: dict[str, Any]
    metrics: dict[str, Any]
    notes: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.git_commit:
            raise ValueError("git_commit is required")

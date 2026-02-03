"""Experiment specification schema for controlled learning-book experiments."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from .llm_strategist import SerializableModel, TriggerCategory


ExperimentLifecycle = Literal["draft", "running", "paused", "completed", "cancelled"]

_VALID_TRANSITIONS: Dict[ExperimentLifecycle, set[ExperimentLifecycle]] = {
    "draft": {"running", "cancelled"},
    "running": {"paused", "completed", "cancelled"},
    "paused": {"running", "completed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}


class ExposureSpec(SerializableModel):
    """Bounds on what the experiment is allowed to trade."""

    target_symbols: List[str] = Field(default_factory=list, description="Symbols the experiment may trade.")
    trigger_categories: List[TriggerCategory] = Field(
        default_factory=list,
        description="Trigger categories allowed for this experiment.",
    )
    max_notional_usd: float = Field(
        default=500.0, ge=0.0,
        description="Per-trade notional cap in USD.",
    )
    max_concurrent_positions: int = Field(
        default=1, ge=0,
        description="Max simultaneous open positions for this experiment.",
    )


class MetricSpec(SerializableModel):
    """Success / failure criteria for the experiment."""

    target_metric: str = Field(
        default="sharpe_ratio",
        description="Primary metric to evaluate (e.g. sharpe_ratio, win_rate, pnl).",
    )
    target_value: float = Field(
        default=0.0,
        description="Threshold value for the primary metric.",
    )
    min_sample_size: int = Field(
        default=10, ge=1,
        description="Minimum number of trades before drawing conclusions.",
    )
    max_loss_usd: float = Field(
        default=100.0, ge=0.0,
        description="Maximum cumulative loss before auto-pausing.",
    )


class ExperimentSpec(SerializableModel):
    """Complete experiment definition that lives alongside a strategy run."""

    experiment_id: str
    name: str
    description: str = ""
    status: ExperimentLifecycle = "draft"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    exposure: ExposureSpec = Field(default_factory=ExposureSpec)
    metrics: MetricSpec = Field(default_factory=MetricSpec)
    hypothesis: str = ""
    variant: str = Field(default="treatment", description="Variant label (control, treatment, etc.).")
    tags: Dict[str, str] = Field(default_factory=dict)

    def can_transition(self, new_status: ExperimentLifecycle) -> bool:
        """Return True if transitioning from current status to *new_status* is valid."""
        allowed = _VALID_TRANSITIONS.get(self.status, set())
        return new_status in allowed

    def transition(self, new_status: ExperimentLifecycle) -> None:
        """Transition to a new status, raising ValueError on invalid transitions."""
        if not self.can_transition(new_status):
            raise ValueError(
                f"Cannot transition experiment from '{self.status}' to '{new_status}'. "
                f"Allowed: {_VALID_TRANSITIONS.get(self.status, set())}"
            )
        self.status = new_status

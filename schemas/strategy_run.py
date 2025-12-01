"""Strategy run schema tying plans, judge feedback, and config together."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import Field

from .judge_feedback import JudgeFeedback
from .llm_strategist import SerializableModel


class StrategyRunConfig(SerializableModel):
    """Static configuration for a strategy run."""

    symbols: List[str] = Field(..., min_length=1)
    timeframes: List[str] = Field(default_factory=list)
    history_window_days: int = Field(default=30, ge=1)
    plan_cadence_hours: int = Field(default=24, ge=1)
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrategyRun(SerializableModel):
    """Registry entry linking StrategyPlan, JudgeFeedback, and configuration."""

    run_id: str
    config: StrategyRunConfig
    current_plan_id: str | None = None
    compiled_plan_id: str | None = None
    plan_active: bool = False
    latest_judge_feedback: JudgeFeedback | None = None
    is_locked: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

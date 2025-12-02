"""Strategy run schema tying plans, judge feedback, and config together."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import Field

from .judge_feedback import JudgeFeedback
from .llm_strategist import RiskConstraint, SerializableModel


class RiskLimitSettings(SerializableModel):
    """User-configurable guardrails that cap gross exposure."""

    max_position_risk_pct: float = Field(2.0, ge=0.0, description="Upper bound for position risk as % of equity per trade.")
    max_symbol_exposure_pct: float = Field(25.0, ge=0.0, description="Max % of equity exposed to a single symbol.")
    max_portfolio_exposure_pct: float = Field(80.0, ge=0.0, description="Max % of equity that can be deployed across all symbols.")
    max_daily_loss_pct: float = Field(3.0, ge=0.0, description="Daily loss cap that triggers a stop for the session.")

    def scaled(self, multiplier: float) -> "RiskLimitSettings":
        """Return a new RiskLimitSettings scaled by multiplier."""

        if multiplier <= 0:
            multiplier = 0.0
        return RiskLimitSettings(
            max_position_risk_pct=self.max_position_risk_pct * multiplier,
            max_symbol_exposure_pct=self.max_symbol_exposure_pct * multiplier,
            max_portfolio_exposure_pct=self.max_portfolio_exposure_pct * multiplier,
            max_daily_loss_pct=self.max_daily_loss_pct * multiplier,
        )

    def to_risk_params(self) -> Dict[str, float]:
        """Return the limits as a dict that can be injected into LLMInput.risk_params."""

        return {
            "max_position_risk_pct": self.max_position_risk_pct,
            "max_symbol_exposure_pct": self.max_symbol_exposure_pct,
            "max_portfolio_exposure_pct": self.max_portfolio_exposure_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
        }

    def as_constraint(self) -> RiskConstraint:
        """Convert settings into a RiskConstraint instance."""

        return RiskConstraint(
            max_position_risk_pct=self.max_position_risk_pct,
            max_symbol_exposure_pct=self.max_symbol_exposure_pct,
            max_portfolio_exposure_pct=self.max_portfolio_exposure_pct,
            max_daily_loss_pct=self.max_daily_loss_pct,
        )


class StrategyRunConfig(SerializableModel):
    """Static configuration for a strategy run."""

    symbols: List[str] = Field(..., min_length=1)
    timeframes: List[str] = Field(default_factory=list)
    history_window_days: int = Field(default=30, ge=1)
    plan_cadence_hours: int = Field(default=24, ge=1)
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_limits: RiskLimitSettings = Field(default_factory=RiskLimitSettings)


class RiskAdjustmentState(SerializableModel):
    """Tracks an active risk scaling instruction from the judge."""

    multiplier: float = Field(1.0, ge=0.0, description="Scalar applied to risk limits or symbol sizing.")
    restore_after_wins: Optional[int] = Field(default=None, ge=1, description="Consecutive winning days required to restore baseline risk.")
    wins_progress: int = Field(default=0, ge=0, description="Number of winning days observed while adjustment is active.")
    instruction: Optional[str] = Field(default=None, description="Raw instruction text for audit/debugging.")

    def record_day(self, winning_day: bool) -> bool:
        """Update win counter and return True if the adjustment should be cleared."""

        if not self.restore_after_wins:
            return False
        if winning_day:
            self.wins_progress += 1
        else:
            self.wins_progress = 0
        return self.wins_progress >= (self.restore_after_wins or 0)


class StrategyRun(SerializableModel):
    """Registry entry linking StrategyPlan, JudgeFeedback, and configuration."""

    run_id: str
    config: StrategyRunConfig
    current_plan_id: str | None = None
    compiled_plan_id: str | None = None
    plan_active: bool = False
    latest_judge_feedback: JudgeFeedback | None = None
    risk_adjustments: Dict[str, RiskAdjustmentState] = Field(default_factory=dict)
    is_locked: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

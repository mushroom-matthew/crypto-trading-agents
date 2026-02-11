"""Strategy run schema tying plans, judge feedback, and config together."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import Field

from .judge_feedback import JudgeFeedback, JudgeAction
from .learning_gate import LearningGateThresholds, LearningKillSwitchConfig
from .llm_strategist import RiskConstraint, SerializableModel


class RiskLimitSettings(SerializableModel):
    """User-configurable guardrails that cap gross exposure."""

    max_position_risk_pct: float = Field(2.0, ge=0.0, description="Upper bound for position risk as % of equity per trade.")
    max_symbol_exposure_pct: float = Field(25.0, ge=0.0, description="Max % of equity exposed to a single symbol.")
    max_portfolio_exposure_pct: float = Field(80.0, ge=0.0, description="Max % of equity that can be deployed across all symbols.")
    max_daily_loss_pct: float = Field(3.0, ge=0.0, description="Daily loss cap that triggers a stop for the session.")
    max_daily_risk_budget_pct: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional cap on cumulative per-trade risk allocated over a day.",
    )

    def scaled(self, multiplier: float) -> "RiskLimitSettings":
        """Return a new RiskLimitSettings scaled by multiplier."""

        if multiplier <= 0:
            multiplier = 0.0
        return RiskLimitSettings(
            max_position_risk_pct=self.max_position_risk_pct * multiplier,
            max_symbol_exposure_pct=self.max_symbol_exposure_pct * multiplier,
            max_portfolio_exposure_pct=self.max_portfolio_exposure_pct * multiplier,
            max_daily_loss_pct=self.max_daily_loss_pct * multiplier,
            max_daily_risk_budget_pct=(self.max_daily_risk_budget_pct * multiplier) if self.max_daily_risk_budget_pct else None,
        )

    def to_risk_params(self) -> Dict[str, float]:
        """Return the limits as a dict that can be injected into LLMInput.risk_params."""

        params = {
            "max_position_risk_pct": self.max_position_risk_pct,
            "max_symbol_exposure_pct": self.max_symbol_exposure_pct,
            "max_portfolio_exposure_pct": self.max_portfolio_exposure_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
        }
        if self.max_daily_risk_budget_pct is not None:
            params["max_daily_risk_budget_pct"] = self.max_daily_risk_budget_pct
        return params

    def as_constraint(self) -> RiskConstraint:
        """Convert settings into a RiskConstraint instance."""

        return RiskConstraint(
            max_position_risk_pct=self.max_position_risk_pct,
            max_symbol_exposure_pct=self.max_symbol_exposure_pct,
            max_portfolio_exposure_pct=self.max_portfolio_exposure_pct,
            max_daily_loss_pct=self.max_daily_loss_pct,
        )


class LearningBookSettings(SerializableModel):
    """Settings governing the learning book: a sandboxed sub-portfolio used for
    exploratory / experimental trades that are tracked separately from the
    profit book."""

    enabled: bool = Field(default=False, description="Whether the learning book is active.")
    daily_risk_budget_pct: float = Field(
        default=1.0, ge=0.0, le=100.0,
        description="Max cumulative risk the learning book can allocate per day, as % of equity.",
    )
    max_position_risk_pct: float = Field(
        default=0.5, ge=0.0, le=100.0,
        description="Max risk per individual learning trade as % of equity.",
    )
    max_portfolio_exposure_pct: float = Field(
        default=5.0, ge=0.0, le=100.0,
        description="Max total notional exposure from learning positions as % of equity.",
    )
    max_trades_per_day: int = Field(
        default=3, ge=0, le=100,
        description="Maximum number of learning trades per day.",
    )
    sizing_mode: str = Field(
        default="notional",
        description="Position sizing mode for learning trades (notional, fixed_fraction, vol_target).",
    )
    notional_usd: float = Field(
        default=100.0, ge=0.0,
        description="Fixed notional in USD per learning trade when sizing_mode='notional'.",
    )
    max_hold_minutes: int = Field(
        default=60, ge=1,
        description="Maximum minutes a learning position may be held before forced exit.",
    )
    allow_short: bool = Field(
        default=False,
        description="Whether learning-book trades may go short.",
    )
    gate_thresholds: LearningGateThresholds = Field(
        default_factory=LearningGateThresholds,
        description="Market-condition thresholds that close the learning gate.",
    )
    kill_switches: LearningKillSwitchConfig = Field(
        default_factory=LearningKillSwitchConfig,
        description="Kill-switch configuration for the learning book.",
    )


class StrategyRunConfig(SerializableModel):
    """Static configuration for a strategy run."""

    symbols: List[str] = Field(..., min_length=1)
    timeframes: List[str] = Field(default_factory=list)
    history_window_days: int = Field(default=30, ge=1)
    plan_cadence_hours: int = Field(default=24, ge=1)
    notes: Optional[str] = None
    debug_trigger_sample_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probability (0.0-1.0) of sampling trigger evaluations for debugging.",
    )
    debug_trigger_max_samples: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum number of trigger evaluation samples to collect.",
    )
    indicator_debug_mode: Optional[str] = Field(
        default=None,
        description="Indicator debug mode: off, full, keys",
    )
    indicator_debug_keys: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_limits: RiskLimitSettings = Field(default_factory=RiskLimitSettings)
    learning_book: LearningBookSettings = Field(default_factory=LearningBookSettings)
    experiment_spec: Dict[str, Any] | None = Field(
        default=None,
        description="Serialized ExperimentSpec for this strategy run.",
    )


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
    latest_judge_action: JudgeAction | None = None
    risk_adjustments: Dict[str, RiskAdjustmentState] = Field(default_factory=dict)
    is_locked: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

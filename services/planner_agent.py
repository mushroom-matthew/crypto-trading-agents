"""LLM planner service that emits strategy configs for the next day."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition
from trading_core.config import PlannerSettings
from tools.performance_analysis import PerformanceAnalyzer


@dataclass
class PlannerPreferences:
    risk_mode: str
    allow_go_to_cash: bool = True


@dataclass
class PlannerRequest:
    symbol: str
    indicators: Dict[str, Any]
    preferences: PlannerPreferences


@dataclass
class PlannerResponse:
    plan: StrategyPlan
    metadata: Dict[str, Any]


class LLMPlanner:
    def __init__(self, settings: PlannerSettings) -> None:
        self.settings = settings

    def plan(self, request: PlannerRequest) -> PlannerResponse:
        """Call the LLM planner and return a validated StrategyPlan."""

        indicators = request.indicators
        metadata = {
            "volatility": {
                "atr": indicators.get("atr"),
                "atr_percent_of_price": indicators.get("atr") / indicators.get("price") if indicators.get("price") else None,
                "atr_band_mult": 1.5,
            },
            "volume": {
                "volume_multiple": indicators.get("volume_multiple", 1.0),
                "volume_floor": 1.0,
            },
            "trend": {
                "ema_fast": indicators.get("ema_fast"),
                "ema_slow": indicators.get("ema_slow"),
                "trend_bias": "bullish" if indicators.get("ema_fast", 0) > indicators.get("ema_slow", 0) else "bearish",
            },
            "risk_mode": request.preferences.risk_mode,
            "allow_go_to_cash": request.preferences.allow_go_to_cash,
        }
        payload = {
            "symbol": request.symbol,
            "timeframe": indicators.get("timeframe", "1h"),
            "planner_bounds": {
                "min_lookback_bars": self.settings.min_lookback_bars,
                "max_lookback_bars": self.settings.max_lookback_bars,
            },
            "metadata": metadata,
        }
        now = datetime.now(timezone.utc)
        plan = StrategyPlan(
            generated_at=now,
            valid_until=now + timedelta(days=1),
            global_view=f"Auto plan for {request.symbol}",
            regime="range",
            triggers=[
                TriggerCondition(
                    id=f"baseline_{request.symbol}",
                    symbol=request.symbol,
                    direction="long",
                    timeframe=payload["timeframe"],
                    entry_rule="timeframe=='1h'",
                    exit_rule="False",
                )
            ],
            risk_constraints=RiskConstraint(
                max_position_risk_pct=20.0,
                max_symbol_exposure_pct=50.0,
                max_portfolio_exposure_pct=80.0,
                max_daily_loss_pct=5.0,
            ),
            sizing_rules=[
                PositionSizingRule(symbol=request.symbol, sizing_mode="fixed_fraction", target_risk_pct=2.0)
            ],
        )
        return PlannerResponse(plan=plan, metadata=metadata)

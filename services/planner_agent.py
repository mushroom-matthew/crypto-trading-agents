"""LLM planner service that emits strategy configs for the next day."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any

from schemas.strategy_plan import StrategyPlan
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
        # TODO: send payload to LLM; using stubbed plan for now.
        plan_payload = {
            "strategy_id": f"plan-{request.symbol}",
            "created_at": datetime.now(timezone.utc),
            "symbol": request.symbol,
            "timeframe": payload["timeframe"],
            "lookback": {
                "preferred_bars": min(max(self.settings.min_lookback_bars, 100), self.settings.max_lookback_bars),
                "min_bars": self.settings.min_lookback_bars,
                "max_bars": self.settings.max_lookback_bars,
            },
            "risk": {
                "max_fraction_of_balance": 0.3,
                "risk_per_trade_fraction": 0.01,
                "max_drawdown_pct": 0.4,
                "leverage": 1.0,
            },
            "entry_conditions": [],
            "exit_conditions": [],
            "replan_triggers": {},
            "llm_metadata": {"model_name": self.settings.llm_model, "prompt_version": "v0"},
        }
        plan = StrategyPlan.model_validate(plan_payload)
        return PlannerResponse(plan=plan, metadata=metadata)

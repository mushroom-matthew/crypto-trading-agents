"""Backtester service bridging strategist plans to the deterministic engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from agents.strategies.llm_client import LLMClient
from backtesting.llm_strategist_runner import LLMStrategistBacktester


def _default_risk_params() -> Dict[str, float]:
    return {
        "max_position_risk_pct": 2.0,
        "max_symbol_exposure_pct": 25.0,
        "max_portfolio_exposure_pct": 80.0,
        "max_daily_loss_pct": 3.0,
    }


def _format_market_data(symbol: str, timeframe: str, candles: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    df = candles.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    else:
        raise ValueError("candles dataframe must include timestamp or time column")
    return {symbol: {timeframe: df}}


@dataclass
class BacktesterService:
    llm_client: LLMClient = field(default_factory=LLMClient)
    cache_dir: Path = field(default_factory=lambda: Path(".cache/strategy_plans"))
    llm_calls_per_day: int = 1
    risk_params: Dict[str, float] = field(default_factory=_default_risk_params)

    def run(
        self,
        candles: pd.DataFrame,
        symbol: str,
        timeframe: str,
        risk_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        market_data = _format_market_data(symbol, timeframe, candles)
        backtester = LLMStrategistBacktester(
            pairs=[symbol],
            start=None,
            end=None,
            initial_cash=1000.0,
            fee_rate=0.001,
            llm_client=self.llm_client,
            cache_dir=self.cache_dir,
            llm_calls_per_day=self.llm_calls_per_day,
            risk_params=risk_params or self.risk_params,
            market_data=market_data,
        )
        result = backtester.run(run_id=f"service-{symbol}")
        return {
            "fills": result.fills.to_dict(orient="records") if not result.fills.empty else [],
            "equity_curve": result.equity_curve.tolist(),
            "final_cash": result.final_cash,
            "final_positions": result.final_positions,
            "plan_log": result.plan_log,
            "llm_costs": result.llm_costs,
            "daily_reports": result.daily_reports,
        }

"""Temporal activity wrapper for running LLM-assisted backtests."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, Field
from temporalio import activity

from agents.strategies.llm_client import LLMClient
from backtesting.llm_strategist_runner import LLMStrategistBacktester

logger = logging.getLogger(__name__)


def _default_risk_params() -> Dict[str, float | None]:
    return {
        "max_position_risk_pct": 2.0,
        "max_symbol_exposure_pct": 25.0,
        "max_portfolio_exposure_pct": 80.0,
        "max_daily_loss_pct": 3.0,
        "max_daily_risk_budget_pct": None,
    }


class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str
    candles: List[Dict[str, Any]]
    initial_cash: float = Field(default=1000.0, ge=0)
    fee_rate: float = 0.001
    llm_model: str = "gpt-4o-mini"
    llm_calls_per_day: int = 1
    llm_cache_dir: str = ".cache/strategy_plans"
    risk_params: Dict[str, float | None] = Field(default_factory=_default_risk_params)
    flatten_daily: bool = False


class BacktestResponse(BaseModel):
    symbol: str
    timeframe: str
    num_trades: int
    final_cash: float
    final_positions: Dict[str, float]
    equity_curve: List[float]
    plan_log: List[Dict[str, Any]]
    llm_costs: Dict[str, float]
    daily_reports: List[Dict[str, Any]]


def _format_market_data(symbol: str, timeframe: str, candles: List[Dict[str, Any]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    df = pd.DataFrame(candles)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    else:
        raise ValueError("candles must include timestamp or time column")
    return {symbol: {timeframe: df}}


@activity.defn
def run_backtest_activity(request_data: Dict[str, Any]) -> Dict[str, Any]:
    request = BacktestRequest.model_validate(request_data)
    logger.info("Starting strategist backtest", extra={"symbol": request.symbol, "bars": len(request.candles)})
    market_data = _format_market_data(request.symbol, request.timeframe, request.candles)
    llm_client = LLMClient(model=request.llm_model)
    backtester = LLMStrategistBacktester(
        pairs=[request.symbol],
        start=None,
        end=None,
        initial_cash=request.initial_cash,
        fee_rate=request.fee_rate,
        llm_client=llm_client,
        cache_dir=Path(request.llm_cache_dir),
        llm_calls_per_day=request.llm_calls_per_day,
        risk_params=request.risk_params,
        market_data=market_data,
        flatten_positions_daily=request.flatten_daily,
    )
    result = backtester.run(run_id=f"activity-{request.symbol}-{int(time.time())}")
    response = BacktestResponse(
        symbol=request.symbol,
        timeframe=request.timeframe,
        num_trades=len(result.fills) if not result.fills.empty else 0,
        final_cash=result.final_cash,
        final_positions=result.final_positions,
        equity_curve=result.equity_curve.tolist(),
        plan_log=result.plan_log,
        llm_costs=result.llm_costs,
        daily_reports=result.daily_reports,
    )
    logger.info(
        "Strategist backtest complete",
        extra={"symbol": request.symbol, "trades": response.num_trades, "final_cash": response.final_cash},
    )
    return response.model_dump()

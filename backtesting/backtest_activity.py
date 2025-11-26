"""Temporal activity wrapper for running LLM-assisted backtests."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, Field

from temporalio import activity

from agents.langfuse_utils import create_openai_client
from backtesting.llm_backtest_engine import (
    BacktestPortfolio,
    run_backtest,
)

logger = logging.getLogger(__name__)


class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str
    candles: List[Dict[str, Any]]
    initial_cash: float = Field(default=1000.0, ge=0)
    config_bounds: Dict[str, int] = Field(
        default_factory=lambda: {"min_lookback_bars": 50, "max_lookback_bars": 300}
    )
    llm_model: str = "gpt-4o-mini"


class BacktestResponse(BaseModel):
    symbol: str
    timeframe: str
    num_trades: int
    final_cash: float
    final_positions: Dict[str, float]
    equity_curve: List[float]


def _get_llm_client() -> Any:
    return create_openai_client()


@activity.defn
def run_backtest_activity(request_data: Dict[str, Any]) -> Dict[str, Any]:
    request = BacktestRequest.model_validate(request_data)
    logger.info("Starting backtest", extra={"symbol": request.symbol, "bars": len(request.candles)})
    df = pd.DataFrame(request.candles)
    llm_client = _get_llm_client()
    portfolio = BacktestPortfolio(cash=request.initial_cash)
    result = run_backtest(
        market_data=df,
        initial_portfolio=portfolio,
        llm_client=llm_client,
        symbol=request.symbol,
        timeframe=request.timeframe,
        config_bounds=request.config_bounds,
    )
    response = BacktestResponse(
        symbol=request.symbol,
        timeframe=request.timeframe,
        num_trades=len(result.trades),
        final_cash=result.final_portfolio.cash,
        final_positions=result.final_portfolio.positions,
        equity_curve=result.equity_curve,
    )
    logger.info(
        "Backtest complete",
        extra={"symbol": request.symbol, "trades": response.num_trades, "final_cash": response.final_cash},
    )
    return response.model_dump()


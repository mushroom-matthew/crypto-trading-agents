"""Backtester service bridging new strategy to walk-forward engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from backtesting.llm_backtest_engine import run_backtest, BacktestPortfolio


@dataclass
class BacktesterService:
    llm_client: Any

    def run(self, candles: pd.DataFrame, symbol: str, timeframe: str, config_bounds: Dict[str, int]) -> Dict[str, Any]:
        portfolio = BacktestPortfolio(cash=1000)
        result = run_backtest(candles, portfolio, self.llm_client, symbol, timeframe, config_bounds)
        return {
            "trades": [trade.__dict__ for trade in result.trades],
            "equity_curve": result.equity_curve,
            "final_cash": result.final_portfolio.cash,
        }

"""Backtesting utilities and entrypoints."""

from .llm_backtest_engine import (
    BacktestAction,
    BacktestPortfolio,
    BacktestResult,
    call_llm_strategy_planner,
    run_backtest as run_gated_backtest,
)
from .simulator import BacktestResult as StrategyBacktestResult, PortfolioBacktestResult, run_backtest, run_portfolio_backtest

__all__ = [
    "BacktestAction",
    "BacktestPortfolio",
    "BacktestResult",
    "call_llm_strategy_planner",
    "run_backtest",
    "run_gated_backtest",
    "StrategyBacktestResult",
    "PortfolioBacktestResult",
    "run_portfolio_backtest",
]

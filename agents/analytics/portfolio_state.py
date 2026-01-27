"""Portfolio state derivation utilities shared by the strategist."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping

import pandas as pd

from schemas.llm_strategist import PortfolioState
from tools.performance_analysis import PerformanceAnalyzer


@dataclass
class PortfolioHistory:
    """Rolling history inputs used to compute derived statistics."""

    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame | None = None


def _window_series(series: pd.Series, start: datetime) -> pd.Series:
    return series[series.index >= start]


def _window_change(series: pd.Series, start: datetime) -> float:
    window = _window_series(series, start)
    if window.empty:
        return 0.0
    return float(window.iloc[-1] - window.iloc[0])


def _win_rate_and_profit_factor(trades: pd.DataFrame | None, start: datetime) -> tuple[float, float]:
    """Compute win rate and profit factor from trade log.

    Win/loss thresholds match trading_core.trade_quality.compute_trade_metrics:
    - Win: pnl > 0.01 (small threshold to filter noise)
    - Loss: pnl < -0.01
    - Breakeven: abs(pnl) <= 0.01

    This ensures consistency between backtest and live metrics.
    """
    if trades is None or trades.empty:
        return 0.0, 0.0
    df = trades.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[df["timestamp"] >= start]
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[df.index >= start]
    if df.empty or "pnl" not in df.columns:
        return 0.0, 0.0
    pnl = df["pnl"].astype(float)
    # Use 0.01 threshold to match trade_quality.py
    wins = pnl[pnl > 0.01]
    losses = pnl[pnl < -0.01]
    total = len(pnl[abs(pnl) > 0.01])  # Exclude breakeven trades from denominator
    win_rate = len(wins) / total if total else 0.0
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    return win_rate, profit_factor


def compute_portfolio_state(
    history: PortfolioHistory,
    as_of: datetime,
    positions: Mapping[str, float],
) -> PortfolioState:
    """Return a PortfolioState computed from historical equity and trade logs."""

    equity_df = history.equity_curve.copy()
    if "timestamp" not in equity_df.columns:
        equity_df = equity_df.reset_index().rename(columns={"index": "timestamp"})
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
    equity_df = equity_df[equity_df["timestamp"] <= as_of]
    if equity_df.empty:
        raise ValueError("equity history missing data for as_of timestamp")
    equity_df = equity_df.set_index("timestamp").sort_index()
    if "equity" not in equity_df.columns:
        raise ValueError("equity_curve must include 'equity' column")
    cash_series = equity_df["cash"] if "cash" in equity_df.columns else equity_df["equity"]

    analyzer = PerformanceAnalyzer()
    equity_series = equity_df["equity"]
    returns_window = _window_series(equity_series.pct_change().dropna(), as_of - timedelta(days=30))
    sharpe = analyzer.calculate_sharpe_ratio(returns_window.tolist()) if not returns_window.empty else 0.0
    dd_window = _window_series(equity_series, as_of - timedelta(days=90))
    max_drawdown = analyzer.calculate_max_drawdown(dd_window.tolist()) if not dd_window.empty else 0.0
    win_rate, profit_factor = _win_rate_and_profit_factor(history.trade_log, as_of - timedelta(days=30))

    return PortfolioState(
        timestamp=equity_series.index[-1].to_pydatetime(),
        equity=float(equity_series.iloc[-1]),
        cash=float(cash_series.iloc[-1]),
        positions=dict(positions),
        realized_pnl_7d=_window_change(equity_series, as_of - timedelta(days=7)),
        realized_pnl_30d=_window_change(equity_series, as_of - timedelta(days=30)),
        sharpe_30d=sharpe,
        max_drawdown_90d=max_drawdown,
        win_rate_30d=win_rate,
        profit_factor_30d=profit_factor,
    )

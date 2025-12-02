"""Core simulation loop for strategy backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from trading_core.judge_agent import PortfolioState
from trading_core.signal_agent import Intent
from tools.performance_analysis import PerformanceAnalyzer

from .dataset import load_ohlcv
from .strategies import ExecutionAgentStrategy, StrategyWrapperConfig


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    summary: Dict[str, Any]


@dataclass
class PortfolioBacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    summary: Dict[str, Any]
    per_pair: Dict[str, BacktestResult]


def _apply_intents(
    intents: List[Intent],
    portfolio: PortfolioState,
    price: float,
    fee_rate: float,
) -> List[Dict[str, Any]]:
    fills: List[Dict[str, Any]] = []
    for intent in intents:
        symbol = intent.symbol
        if intent.action == "BUY":
            allocation = portfolio.cash * max(0.0, min(intent.size_hint, 1.0))
            if allocation <= 0:
                continue
            qty = allocation / price
            fee = allocation * fee_rate
            total_cost = allocation + fee
            if total_cost > portfolio.cash:
                continue
            portfolio.cash -= total_cost
            portfolio.positions[symbol] = portfolio.positions.get(symbol, 0.0) + qty
            fills.append({"symbol": symbol, "side": "BUY", "qty": qty, "price": price, "fee": fee})
        elif intent.action in {"SELL", "CLOSE"}:
            qty_available = portfolio.positions.get(symbol, 0.0)
            if qty_available <= 0:
                continue
            fraction = 1.0 if intent.action == "CLOSE" else max(0.0, min(intent.size_hint, 1.0))
            qty_to_sell = qty_available * fraction
            proceeds = qty_to_sell * price
            fee = proceeds * fee_rate
            portfolio.cash += proceeds - fee
            new_qty = qty_available - qty_to_sell
            if new_qty <= 1e-8:
                portfolio.positions.pop(symbol, None)
            else:
                portfolio.positions[symbol] = new_qty
            fills.append({"symbol": symbol, "side": "SELL", "qty": qty_to_sell, "price": price, "fee": fee})
    return fills


def _flatten_positions(
    portfolio: PortfolioState,
    last_prices: Dict[str, float],
    timestamp: pd.Timestamp | None,
    fee_rate: float,
    trades: List[Dict[str, Any]],
) -> None:
    if timestamp is None:
        return
    updated = False
    for symbol, qty in list(portfolio.positions.items()):
        if abs(qty) <= 1e-9:
            continue
        price = last_prices.get(symbol)
        if price is None:
            continue
        side = "SELL" if qty > 0 else "BUY"
        notional = abs(qty) * price
        fee = notional * fee_rate
        if side == "SELL":
            portfolio.cash += notional - fee
        else:
            portfolio.cash -= notional + fee
        trades.append({"symbol": symbol, "side": side, "qty": abs(qty), "price": price, "fee": fee, "time": timestamp})
        portfolio.positions.pop(symbol, None)
        updated = True
    if updated:
        portfolio.equity = portfolio.cash


def _compute_features(df: pd.DataFrame, lookback: int = 50, atr_window: int = 14) -> pd.DataFrame:
    rolling_high = df["close"].rolling(window=lookback).max()
    rolling_low = df["close"].rolling(window=lookback).min()
    recent_max = df["close"].cummax()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    )
    atr = tr.max(axis=1).rolling(window=atr_window).mean()
    vol_ma = df["volume"].rolling(window=20).mean()
    volume_multiple = df["volume"] / vol_ma
    features = df.copy()
    features["rolling_high"] = rolling_high
    features["rolling_low"] = rolling_low
    features["recent_max"] = recent_max
    features["atr"] = atr
    features["volume_multiple"] = volume_multiple.fillna(1.0)
    return features.dropna()


def _compute_metrics_summary(
    equity_series: pd.Series,
    trades: pd.DataFrame,
    initial_cash: float,
) -> Dict[str, Any]:
    final_equity = float(equity_series.iloc[-1]) if not equity_series.empty else initial_cash
    equity_return_pct = (final_equity / initial_cash - 1) * 100 if initial_cash else 0.0
    gross_trade_return_pct = equity_return_pct
    analyzer = PerformanceAnalyzer()
    returns = equity_series.pct_change().dropna().tolist()
    sharpe = analyzer.calculate_sharpe_ratio(returns) if returns else 0.0
    max_drawdown = analyzer.calculate_max_drawdown(equity_series.tolist()) if not equity_series.empty else 0.0

    if not trades.empty:
        transactions = [
            {
                "timestamp": int(row["time"].timestamp()),
                "symbol": row["symbol"],
                "side": row["side"],
                "quantity": row["qty"],
                "fill_price": row["price"],
            }
            for _, row in trades.iterrows()
        ]
        win_rate, avg_win, avg_loss, profit_factor = analyzer.calculate_win_metrics(transactions)
    else:
        win_rate = avg_win = avg_loss = 0.0
        profit_factor = 0.0

    return {
        "final_equity": final_equity,
        "equity_return_pct": equity_return_pct,
        "gross_trade_return_pct": gross_trade_return_pct,
        "return_pct": equity_return_pct,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
        "win_rate": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def run_backtest(
    pair: str,
    start: datetime,
    end: datetime,
    initial_cash: float,
    fee_rate: float,
    strategy_config: StrategyWrapperConfig,
    flatten_positions_daily: bool = False,
) -> BacktestResult:
    raw_data = load_ohlcv(pair, start, end, timeframe='1h')
    features = _compute_features(raw_data)
    strategy = ExecutionAgentStrategy(strategy_config)

    portfolio = PortfolioState(cash=initial_cash, positions={}, equity=initial_cash, max_equity=initial_cash)
    equity_points: List[float] = []
    equity_index: List[pd.Timestamp] = []
    trades: List[Dict[str, Any]] = []
    last_day: datetime.date | None = None
    last_timestamp: pd.Timestamp | None = None
    last_prices: Dict[str, float] = {}

    for ts, row in features.iterrows():
        day = ts.date()
        if flatten_positions_daily and last_day and day != last_day:
            _flatten_positions(portfolio, last_prices, last_timestamp, fee_rate, trades)
        price = float(row["close"])
        feature_vector = {
            "symbol": pair,
            "time": ts,
            "price": price,
            "rolling_high": float(row["rolling_high"]),
            "rolling_low": float(row["rolling_low"]),
            "recent_max": float(row["recent_max"]),
            "atr": float(row["atr"]),
            "volume_multiple": float(row["volume_multiple"]),
        }
        intents = strategy.decide(feature_vector, portfolio)
        fills = _apply_intents(intents, portfolio, price, fee_rate)
        trades.extend({"time": ts, **fill} for fill in fills)

        equity = portfolio.cash
        for qty in portfolio.positions.values():
            equity += qty * price
        portfolio.equity = equity
        portfolio.max_equity = max(portfolio.max_equity, equity)
        equity_index.append(ts)
        equity_points.append(equity)
        last_day = day
        last_timestamp = ts
        last_prices[pair] = price

    if flatten_positions_daily:
        _flatten_positions(portfolio, last_prices, last_timestamp, fee_rate, trades)

    equity_series = pd.Series(equity_points, index=equity_index)
    trades_df = pd.DataFrame(trades)
    summary = _compute_metrics_summary(equity_series, trades_df, initial_cash)
    return BacktestResult(equity_curve=equity_series, trades=trades_df, summary=summary)


def run_portfolio_backtest(
    pairs: Sequence[str],
    start: datetime,
    end: datetime,
    initial_cash: float,
    fee_rate: float,
    strategy_config: StrategyWrapperConfig,
    weights: Optional[Sequence[float]] = None,
    flatten_positions_daily: bool = False,
) -> PortfolioBacktestResult:
    if not pairs:
        raise ValueError("pairs must not be empty")
    if weights and len(weights) != len(pairs):
        raise ValueError("weights length must match pairs")
    if not weights:
        weights = [1 / len(pairs)] * len(pairs)

    aggregated_equity: Optional[pd.Series] = None
    aggregated_trades: List[pd.DataFrame] = []
    per_pair: Dict[str, BacktestResult] = {}

    for pair, weight in zip(pairs, weights):
        cash_allocation = initial_cash * weight
        result = run_backtest(
            pair=pair,
            start=start,
            end=end,
            initial_cash=cash_allocation,
            fee_rate=fee_rate,
            strategy_config=strategy_config,
            flatten_positions_daily=flatten_positions_daily,
        )
        per_pair[pair] = result

        equity = result.equity_curve
        if aggregated_equity is None:
            aggregated_equity = equity
        else:
            aggregated_equity = aggregated_equity.add(equity, fill_value=0.0)

        if not result.trades.empty:
            trades_copy = result.trades.copy()
            trades_copy["symbol"] = pair
            aggregated_trades.append(trades_copy)

    aggregated_equity = aggregated_equity if aggregated_equity is not None else pd.Series([initial_cash], index=[start])
    combined_trades = pd.concat(aggregated_trades).sort_values("time") if aggregated_trades else pd.DataFrame(columns=["time", "symbol", "side", "qty", "price", "fee"])
    summary = _compute_metrics_summary(aggregated_equity, combined_trades, initial_cash)
    return PortfolioBacktestResult(
        equity_curve=aggregated_equity,
        trades=combined_trades,
        summary=summary,
        per_pair=per_pair,
    )

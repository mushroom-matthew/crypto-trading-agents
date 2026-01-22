"""Core simulation loop for strategy backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from trading_core.judge_agent import PortfolioState
from trading_core.signal_agent import Intent
from tools.performance_analysis import PerformanceAnalyzer

from .dataset import load_ohlcv
from .strategies import ExecutionAgentStrategy, StrategyWrapperConfig

# Type for progress callback: (current, total, timestamp) -> None
ProgressCallback = Optional[Callable[[int, int, Optional[str]], None]]


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

def _sanitize_stop_distance(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        dist = float(value)
    except (TypeError, ValueError):
        return None
    return dist if dist > 0 else None


def _risk_usage(
    qty: float,
    price: float,
    cap: float | None,
    stop_distance: float | None,
) -> tuple[float, float]:
    notional = qty * price
    if stop_distance is not None and stop_distance > 0:
        actual_risk = qty * stop_distance
    else:
        actual_risk = notional
    risk_used = min(actual_risk, cap) if cap is not None else actual_risk
    return risk_used, actual_risk


def _apply_intents(
    intents: List[Intent],
    portfolio: PortfolioState,
    cost_basis: Dict[str, float],
    price: float,
    fee_rate: float,
    stop_distance: float | None = None,
    risk_limits: "RiskLimitSettings | None" = None,
) -> List[Dict[str, Any]]:
    fills: List[Dict[str, Any]] = []
    def _per_trade_cap(equity: float) -> float | None:
        if risk_limits is None:
            return None
        pct = getattr(risk_limits, "max_position_risk_pct", None) or 0.0
        if pct <= 0:
            return None
        return equity * (pct / 100.0)

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
            cost_basis[symbol] = cost_basis.get(symbol, 0.0) + total_cost
            cap = _per_trade_cap(portfolio.equity)
            risk_used, actual_risk = _risk_usage(qty, price, cap, stop_distance)
            fills.append({
                "symbol": symbol,
                "side": "BUY",
                "qty": qty,
                "price": price,
                "fee": fee,
                "pnl": 0.0,
                "risk_used_abs": risk_used,
                "actual_risk_at_stop": actual_risk,
                "trigger_id": "baseline_strategy",
                "timeframe": "1h",
            })
        elif intent.action in {"SELL", "CLOSE"}:
            qty_available = portfolio.positions.get(symbol, 0.0)
            if qty_available <= 0:
                continue
            fraction = 1.0 if intent.action == "CLOSE" else max(0.0, min(intent.size_hint, 1.0))
            qty_to_sell = qty_available * fraction
            proceeds = qty_to_sell * price
            fee = proceeds * fee_rate
            basis_total = cost_basis.get(symbol, 0.0)
            basis_portion = basis_total * fraction
            pnl = proceeds - fee - basis_portion
            portfolio.cash += proceeds - fee
            new_qty = qty_available - qty_to_sell
            if new_qty <= 1e-8:
                portfolio.positions.pop(symbol, None)
                cost_basis.pop(symbol, None)
            else:
                portfolio.positions[symbol] = new_qty
                cost_basis[symbol] = max(0.0, basis_total - basis_portion)
            cap = _per_trade_cap(portfolio.equity)
            risk_used, actual_risk = _risk_usage(qty_to_sell, price, cap, stop_distance)
            fills.append({
                "symbol": symbol,
                "side": "SELL",
                "qty": qty_to_sell,
                "price": price,
                "fee": fee,
                "pnl": pnl,
                "risk_used_abs": risk_used,
                "actual_risk_at_stop": actual_risk,
                "trigger_id": "baseline_strategy",
                "timeframe": "1h",
            })
    return fills


def _apply_intents_multi(
    intents: List[Intent],
    portfolio: PortfolioState,
    cost_basis: Dict[str, float],
    prices: Dict[str, float],
    fee_rate: float,
    stop_distances: Dict[str, float] | None = None,
    risk_limits: "RiskLimitSettings | None" = None,
) -> List[Dict[str, Any]]:
    fills: List[Dict[str, Any]] = []

    def _per_trade_cap(equity: float) -> float | None:
        if risk_limits is None:
            return None
        pct = getattr(risk_limits, "max_position_risk_pct", None) or 0.0
        if pct <= 0:
            return None
        return equity * (pct / 100.0)

    for intent in intents:
        symbol = intent.symbol
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue
        stop_distance = stop_distances.get(symbol) if stop_distances else None
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
            cost_basis[symbol] = cost_basis.get(symbol, 0.0) + total_cost
            cap = _per_trade_cap(portfolio.equity)
            risk_used, actual_risk = _risk_usage(qty, price, cap, stop_distance)
            fills.append({
                "symbol": symbol,
                "side": "BUY",
                "qty": qty,
                "price": price,
                "fee": fee,
                "pnl": 0.0,
                "risk_used_abs": risk_used,
                "actual_risk_at_stop": actual_risk,
                "trigger_id": "baseline_strategy",
                "timeframe": "1h",
            })
        elif intent.action in {"SELL", "CLOSE"}:
            qty_available = portfolio.positions.get(symbol, 0.0)
            if qty_available <= 0:
                continue
            fraction = 1.0 if intent.action == "CLOSE" else max(0.0, min(intent.size_hint, 1.0))
            qty_to_sell = qty_available * fraction
            proceeds = qty_to_sell * price
            fee = proceeds * fee_rate
            basis_total = cost_basis.get(symbol, 0.0)
            basis_portion = basis_total * fraction
            pnl = proceeds - fee - basis_portion
            portfolio.cash += proceeds - fee
            new_qty = qty_available - qty_to_sell
            if new_qty <= 1e-8:
                portfolio.positions.pop(symbol, None)
                cost_basis.pop(symbol, None)
            else:
                portfolio.positions[symbol] = new_qty
                cost_basis[symbol] = max(0.0, basis_total - basis_portion)
            cap = _per_trade_cap(portfolio.equity)
            risk_used, actual_risk = _risk_usage(qty_to_sell, price, cap, stop_distance)
            fills.append({
                "symbol": symbol,
                "side": "SELL",
                "qty": qty_to_sell,
                "price": price,
                "fee": fee,
                "pnl": pnl,
                "risk_used_abs": risk_used,
                "actual_risk_at_stop": actual_risk,
                "trigger_id": "baseline_strategy",
                "timeframe": "1h",
            })
    return fills


def _flatten_positions(
    portfolio: PortfolioState,
    cost_basis: Dict[str, float],
    last_prices: Dict[str, float],
    timestamp: pd.Timestamp | None,
    fee_rate: float,
    trades: List[Dict[str, Any]],
    stop_distances: Dict[str, float] | None = None,
    risk_limits: "RiskLimitSettings | None" = None,
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
        basis_total = cost_basis.get(symbol, 0.0)
        pnl = notional - fee - basis_total if side == "SELL" else -(notional + fee - basis_total)
        cap = None
        if risk_limits is not None:
            pct = getattr(risk_limits, "max_position_risk_pct", None) or 0.0
            if pct > 0:
                cap = portfolio.equity * (pct / 100.0)
        stop_distance = stop_distances.get(symbol) if stop_distances else None
        risk_used, actual_risk = _risk_usage(abs(qty), price, cap, stop_distance)
        if side == "SELL":
            portfolio.cash += notional - fee
        else:
            portfolio.cash -= notional + fee
        trades.append({
            "symbol": symbol,
            "side": side,
            "qty": abs(qty),
            "price": price,
            "fee": fee,
            "pnl": pnl,
            "risk_used_abs": risk_used,
            "actual_risk_at_stop": actual_risk,
            "trigger_id": "baseline_strategy",
            "timeframe": "1h",
            "time": timestamp,
        })
        portfolio.positions.pop(symbol, None)
        cost_basis.pop(symbol, None)
        updated = True
    if updated:
        portfolio.equity = portfolio.cash


def _flatten_positions_multi(
    portfolio: PortfolioState,
    cost_basis: Dict[str, float],
    last_prices: Dict[str, float],
    timestamp: pd.Timestamp | None,
    fee_rate: float,
    trades: List[Dict[str, Any]],
    stop_distances: Dict[str, float] | None = None,
    risk_limits: "RiskLimitSettings | None" = None,
) -> None:
    _flatten_positions(
        portfolio,
        cost_basis,
        last_prices,
        timestamp,
        fee_rate,
        trades,
        stop_distances=stop_distances,
        risk_limits=risk_limits,
    )


def _seed_portfolio_with_allocations(
    *,
    pairs: Sequence[str],
    initial_cash: float,
    initial_allocations: Optional[Dict[str, float]],
    initial_prices: Dict[str, float],
) -> tuple[PortfolioState, Dict[str, float]]:
    cash = initial_cash
    positions: Dict[str, float] = {}
    cost_basis: Dict[str, float] = {}

    if initial_allocations:
        cash = float(initial_allocations.get("cash", cash))
        for symbol, notional in initial_allocations.items():
            if symbol == "cash":
                continue
            if symbol not in pairs:
                continue
            price = initial_prices.get(symbol)
            if notional <= 0 or not price:
                continue
            qty = notional / price
            positions[symbol] = qty
            cost_basis[symbol] = notional

    equity = cash
    for symbol, qty in positions.items():
        price = initial_prices.get(symbol)
        if price:
            equity += qty * price

    portfolio = PortfolioState(cash=cash, positions=positions, equity=equity, max_equity=equity)
    return portfolio, cost_basis

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
                "side": str(row.get("side", "")).upper(),
                "quantity": float(row.get("qty", 0.0) or 0.0),
                "fill_price": float(row.get("price", 0.0) or 0.0),
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
        "total_trades": len(trades) if not trades.empty else 0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def _trade_window(
    start: datetime,
    end: datetime,
    trade_start: datetime | None,
    trade_end: datetime | None,
) -> tuple[datetime, datetime]:
    return trade_start or start, trade_end or end


def _initial_price_for_trade(raw_data: pd.DataFrame, trade_start: datetime) -> float:
    subset = raw_data[raw_data.index >= trade_start]
    if not subset.empty:
        return float(subset["close"].iloc[0])
    return float(raw_data["close"].iloc[0])


def run_backtest(
    pair: str,
    start: datetime,
    end: datetime,
    initial_cash: float,
    fee_rate: float,
    strategy_config: StrategyWrapperConfig,
    initial_allocations: Optional[Dict[str, float]] = None,
    flatten_positions_daily: bool = False,
    risk_limits: "RiskLimitSettings | None" = None,
    progress_callback: ProgressCallback = None,
    trade_start: datetime | None = None,
    trade_end: datetime | None = None,
) -> BacktestResult:
    raw_data = load_ohlcv(pair, start, end, timeframe='1h')
    if raw_data.empty:
        raise ValueError(f"No OHLCV data available for {pair}")
    features = _compute_features(raw_data)
    strategy = ExecutionAgentStrategy(strategy_config)

    trade_start, trade_end = _trade_window(start, end, trade_start, trade_end)
    initial_price = _initial_price_for_trade(raw_data, trade_start)
    initial_prices = {pair: initial_price}
    portfolio, cost_basis = _seed_portfolio_with_allocations(
        pairs=[pair],
        initial_cash=initial_cash,
        initial_allocations=initial_allocations,
        initial_prices=initial_prices,
    )
    starting_equity = portfolio.equity
    equity_points: List[float] = []
    equity_index: List[pd.Timestamp] = []
    trades: List[Dict[str, Any]] = []
    last_day: datetime.date | None = None
    last_timestamp: pd.Timestamp | None = None
    last_prices: Dict[str, float] = {pair: initial_price}
    last_stop_distances: Dict[str, float] = {}

    features_window = features[(features.index >= trade_start) & (features.index <= trade_end)]
    if features_window.empty:
        raise ValueError("No feature data available for supplied pairs")
    total_rows = len(features_window)
    for idx, (ts, row) in enumerate(features_window.iterrows()):
        # Call progress callback every 100 candles
        if progress_callback and idx % 100 == 0:
            progress_callback(idx, total_rows, ts.isoformat())
        day = ts.date()
        if flatten_positions_daily and last_day and day != last_day:
            _flatten_positions(
                portfolio,
                cost_basis,
                last_prices,
                last_timestamp,
                fee_rate,
                trades,
                stop_distances=last_stop_distances,
                risk_limits=risk_limits,
            )
        price = float(row["close"])
        stop_distance = _sanitize_stop_distance(row.get("atr"))
        if stop_distance is not None:
            last_stop_distances[pair] = stop_distance
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
        fills = _apply_intents(
            intents,
            portfolio,
            cost_basis,
            price,
            fee_rate,
            stop_distance=stop_distance,
            risk_limits=risk_limits,
        )
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
        _flatten_positions(
            portfolio,
            cost_basis,
            last_prices,
            last_timestamp,
            fee_rate,
            trades,
            stop_distances=last_stop_distances,
        )

    equity_series = pd.Series(equity_points, index=equity_index)
    trades_df = pd.DataFrame(trades)
    summary = _compute_metrics_summary(equity_series, trades_df, starting_equity)
    return BacktestResult(equity_curve=equity_series, trades=trades_df, summary=summary)


def run_multi_asset_backtest(
    pairs: Sequence[str],
    start: datetime,
    end: datetime,
    initial_cash: float,
    fee_rate: float,
    strategy_config: StrategyWrapperConfig,
    initial_allocations: Optional[Dict[str, float]] = None,
    flatten_positions_daily: bool = False,
    risk_limits: "RiskLimitSettings | None" = None,
    progress_callback: ProgressCallback = None,
    trade_start: datetime | None = None,
    trade_end: datetime | None = None,
) -> PortfolioBacktestResult:
    if not pairs:
        raise ValueError("pairs must not be empty")

    trade_start, trade_end = _trade_window(start, end, trade_start, trade_end)
    raw_data_map: Dict[str, pd.DataFrame] = {}
    features_map: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        raw = load_ohlcv(pair, start, end, timeframe='1h')
        if raw.empty:
            raise ValueError(f"No OHLCV data available for {pair}")
        raw_data_map[pair] = raw
        features_map[pair] = _compute_features(raw)

    initial_prices = {pair: _initial_price_for_trade(raw_data_map[pair], trade_start) for pair in pairs}
    portfolio, cost_basis = _seed_portfolio_with_allocations(
        pairs=pairs,
        initial_cash=initial_cash,
        initial_allocations=initial_allocations,
        initial_prices=initial_prices,
    )
    strategy = ExecutionAgentStrategy(strategy_config)

    timestamps = sorted({ts for df in features_map.values() for ts in df.index if trade_start <= ts <= trade_end})
    if not timestamps:
        raise ValueError("No feature data available for supplied pairs")

    starting_equity = portfolio.equity
    equity_points: List[float] = []
    equity_index: List[pd.Timestamp] = []
    trades: List[Dict[str, Any]] = []
    last_day: datetime.date | None = None
    last_timestamp: pd.Timestamp | None = None
    last_prices: Dict[str, float] = dict(initial_prices)
    last_stop_distances: Dict[str, float] = {}

    total_rows = len(timestamps)
    for idx, ts in enumerate(timestamps):
        if progress_callback and idx % 100 == 0:
            progress_callback(idx, total_rows, ts.isoformat())
        day = ts.date()
        if flatten_positions_daily and last_day and day != last_day:
            _flatten_positions_multi(
                portfolio,
                cost_basis,
                last_prices,
                last_timestamp,
                fee_rate,
                trades,
                stop_distances=last_stop_distances,
                risk_limits=risk_limits,
            )

        intents: List[Intent] = []
        for pair in pairs:
            features = features_map[pair]
            if ts not in features.index:
                continue
            row = features.loc[ts]
            price = float(row["close"])
            last_prices[pair] = price
            stop_distance = _sanitize_stop_distance(row.get("atr"))
            if stop_distance is not None:
                last_stop_distances[pair] = stop_distance
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
            intents.extend(strategy.decide(feature_vector, portfolio))

        fills = _apply_intents_multi(
            intents,
            portfolio,
            cost_basis,
            last_prices,
            fee_rate,
            stop_distances=last_stop_distances,
            risk_limits=risk_limits,
        )
        trades.extend({"time": ts, **fill} for fill in fills)

        equity = portfolio.cash
        for symbol, qty in portfolio.positions.items():
            price = last_prices.get(symbol)
            if price is None:
                continue
            equity += qty * price
        portfolio.equity = equity
        portfolio.max_equity = max(portfolio.max_equity, equity)

        equity_index.append(ts)
        equity_points.append(equity)
        last_day = day
        last_timestamp = ts

    if flatten_positions_daily:
        _flatten_positions_multi(
            portfolio,
            cost_basis,
            last_prices,
            last_timestamp,
            fee_rate,
            trades,
            stop_distances=last_stop_distances,
            risk_limits=risk_limits,
        )

    equity_series = pd.Series(equity_points, index=equity_index)
    trades_df = pd.DataFrame(trades)
    summary = _compute_metrics_summary(equity_series, trades_df, starting_equity)
    return PortfolioBacktestResult(
        equity_curve=equity_series,
        trades=trades_df,
        summary=summary,
        per_pair={},
    )


def run_portfolio_backtest(
    pairs: Sequence[str],
    start: datetime,
    end: datetime,
    initial_cash: float,
    fee_rate: float,
    strategy_config: StrategyWrapperConfig,
    initial_allocations: Optional[Dict[str, float]] = None,
    weights: Optional[Sequence[float]] = None,
    flatten_positions_daily: bool = False,
    risk_limits: "RiskLimitSettings | None" = None,
    progress_callback: ProgressCallback = None,
    trade_start: datetime | None = None,
    trade_end: datetime | None = None,
) -> PortfolioBacktestResult:
    if not pairs:
        raise ValueError("pairs must not be empty")
    if initial_allocations:
        return run_multi_asset_backtest(
            pairs=pairs,
            start=start,
            end=end,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            strategy_config=strategy_config,
            initial_allocations=initial_allocations,
            flatten_positions_daily=flatten_positions_daily,
            risk_limits=risk_limits,
            progress_callback=progress_callback,
            trade_start=trade_start,
            trade_end=trade_end,
        )
    if weights and len(weights) != len(pairs):
        raise ValueError("weights length must match pairs")
    if not weights:
        weights = [1 / len(pairs)] * len(pairs)

    aggregated_equity: Optional[pd.Series] = None
    aggregated_trades: List[pd.DataFrame] = []
    per_pair: Dict[str, BacktestResult] = {}

    total_pairs = len(pairs)
    for pair_idx, (pair, weight) in enumerate(zip(pairs, weights)):
        # Create progress wrapper for this pair
        def pair_progress_callback(current: int, total: int, timestamp: str = None):
            if progress_callback:
                # Scale progress across all pairs
                overall_current = (pair_idx * total) + current
                overall_total = total_pairs * total
                progress_callback(overall_current, overall_total, timestamp)
        cash_allocation = initial_cash * weight
        result = run_backtest(
            pair=pair,
            start=start,
            end=end,
            initial_cash=cash_allocation,
            fee_rate=fee_rate,
            strategy_config=strategy_config,
            flatten_positions_daily=flatten_positions_daily,
            risk_limits=risk_limits,
            progress_callback=pair_progress_callback,
            trade_start=trade_start,
            trade_end=trade_end,
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

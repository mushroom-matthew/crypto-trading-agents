"""Walk-forward LLM-assisted backtest engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from indicators import technical
from schemas.strategy_plan import StrategyPlan


@dataclass
class BacktestPortfolio:
    """Simple portfolio tracker for backtests."""

    cash: float
    positions: Dict[str, float] = field(default_factory=dict)
    avg_entry_price: Dict[str, float] = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)

    def update_equity(self, mark_prices: Dict[str, float]) -> None:
        equity = self.cash
        for symbol, qty in self.positions.items():
            price = mark_prices.get(symbol)
            if price is not None:
                equity += qty * price
        self.equity_curve.append(equity)


@dataclass
class BacktestAction:
    symbol: str
    side: str  # "buy", "sell", "hold"
    size: float
    reason: str


@dataclass
class BacktestResult:
    trades: List[BacktestAction]
    equity_curve: List[float]
    final_portfolio: BacktestPortfolio


def build_planner_snapshot(
    past_data: pd.DataFrame,
    indicators: Dict[str, List[float]],
    portfolio: BacktestPortfolio,
    bar_index: int,
    symbol: str,
    timeframe: str,
    config_bounds: Dict[str, int],
) -> Dict[str, Any]:
    """Assemble deterministic snapshot for LLM planner."""

    recent_candles = past_data.tail(config_bounds["max_lookback_bars"]).reset_index(drop=True)
    portfolio_state = {
        "cash_balance": portfolio.cash,
        "positions": [
            {
                "symbol": sym,
                "size": qty,
                "avg_entry_price": portfolio.avg_entry_price.get(sym),
            }
            for sym, qty in portfolio.positions.items()
        ],
        "historical_equity_curve": [
            {"index": i, "equity": equity}
            for i, equity in enumerate(portfolio.equity_curve)
        ],
    }
    timestamp = (
        past_data.index[-1].isoformat()
        if not past_data.empty and hasattr(past_data.index[-1], "isoformat")
        else datetime.now(timezone.utc).isoformat()
    )
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "timeframe": timeframe,
        "config_bounds": config_bounds,
        "recent_candles": recent_candles.to_dict(orient="records"),
        "indicator_history": indicators,
        "portfolio_state": portfolio_state,
        "constraints": {
            "max_position_pct": 0.25,
            "max_leverage": 3.0,
            "max_daily_loss_pct": 0.05,
        },
        "backtest_meta": {
            "bar_index": bar_index,
        },
    }


def call_llm_strategy_planner(llm_client: Any, snapshot: Dict[str, Any]) -> StrategyPlan:
    """Call the external LLM planner and parse StrategyPlan."""

    completion = llm_client.responses.create(
        model=snapshot.get("llm_model", "gpt-4o-mini"),
        input=json.dumps(snapshot),
        temperature=0.1,
        max_output_tokens=800,
    )
    content = completion.output[0].content[0].text
    data = json.loads(content)
    return StrategyPlan.model_validate(data)


def compute_indicators(past_data: pd.DataFrame) -> Dict[str, List[float]]:
    """Compute walk-forward indicators from the provided data."""

    closes = past_data["close"].tolist()
    highs = past_data["high"].tolist()
    lows = past_data["low"].tolist()
    volumes = past_data["volume"].tolist()

    def walk_forward_series(fn, min_period: int, *series) -> List[float]:
        values: List[float] = []
        for idx in range(len(closes)):
            if idx + 1 < min_period:
                values.append(float("nan"))
                continue
            try:
                slice_args = [seq[: idx + 1] for seq in series]
                values.append(fn(*slice_args))
            except Exception:
                values.append(float("nan"))
        return values

    ema20 = walk_forward_series(lambda seq: technical.ema(seq, period=20), 20, closes)
    ema50 = walk_forward_series(lambda seq: technical.ema(seq, period=50), 50, closes)
    rsi14 = walk_forward_series(lambda seq: technical.rsi(seq, period=14), 15, closes)
    atr14 = walk_forward_series(
        lambda h, l, c: technical.atr(h, l, c, period=14),
        15,
        highs,
        lows,
        closes,
    )
    vol_ma20 = walk_forward_series(lambda seq: technical.volume_moving_average(seq, period=20), 20, volumes)

    return {
        "ema_20": ema20,
        "ema_50": ema50,
        "rsi_14": rsi14,
        "atr_14": atr14,
        "volume_ma_20": vol_ma20,
    }


def check_replan_triggers(
    triggers: Dict[str, Any],
    past_data: pd.DataFrame,
    indicators: Dict[str, List[float]],
    portfolio: BacktestPortfolio,
) -> bool:
    """Evaluate trigger configs to determine if re-plan is necessary."""

    latest_idx = len(past_data) - 1
    if latest_idx < 1:
        return False
    ema_fast = indicators.get("ema_20", [])
    ema_slow = indicators.get("ema_50", [])
    if (
        triggers["ema_cross_overrides"]["enabled"]
        and not any(map(pd.isna, ema_fast[-2:]))
        and not any(map(pd.isna, ema_slow[-2:]))
        and technical.ema_crossed(ema_fast[-2:], ema_slow[-2:])
    ):
        return True
    rsi_series = indicators.get("rsi_14", [])
    if triggers["rsi_regime_shift"]["enabled"] and len(rsi_series) >= 2:
        bounds = triggers["rsi_regime_shift"]
        prev_rsi, curr_rsi = rsi_series[latest_idx - 1], rsi_series[latest_idx]
        if (prev_rsi < bounds["overbought"] <= curr_rsi) or (prev_rsi > bounds["oversold"] >= curr_rsi):
            return True
    atr_series = indicators.get("atr_14", [])
    if triggers["atr_volatility_shift"]["enabled"] and len(atr_series) >= 2:
        window = atr_series[-triggers["atr_volatility_shift"]["lookback_bars"] :]
        window = [val for val in window if not pd.isna(val)]
        if window:
            avg_atr = sum(window) / len(window)
            latest_atr = window[-1]
            if latest_atr >= avg_atr * triggers["atr_volatility_shift"]["high_vol_threshold"]:
                return True
            if latest_atr <= avg_atr * triggers["atr_volatility_shift"]["low_vol_threshold"]:
                return True
    return False


def execution_engine(
    plan: StrategyPlan,
    indicators: Dict[str, List[float]],
    portfolio: BacktestPortfolio,
    current_bar: pd.Series,
) -> Optional[BacktestAction]:
    """Very simple placeholder deterministic execution logic."""

    price = current_bar["close"]
    ema20 = indicators["ema_20"][-1]
    ema50 = indicators["ema_50"][-1]
    if pd.isna(ema20) or pd.isna(ema50):
        return None
    if ema20 > ema50 and portfolio.cash > 0:
        size = (portfolio.cash * plan.risk_management.max_position_pct) / price
        return BacktestAction(symbol=plan.symbol, side="buy", size=size, reason="ema_cross_up")
    if ema20 < ema50 and portfolio.positions.get(plan.symbol):
        size = portfolio.positions[plan.symbol]
        return BacktestAction(symbol=plan.symbol, side="sell", size=size, reason="ema_cross_down")
    return None


def simulator_step(portfolio: BacktestPortfolio, action: Optional[BacktestAction], current_bar: pd.Series) -> None:
    if not action:
        return
    price = current_bar["close"]
    symbol = action.symbol
    if action.side == "buy":
        cost = action.size * price
        if cost > portfolio.cash:
            return
        portfolio.cash -= cost
        position = portfolio.positions.get(symbol, 0.0)
        portfolio.positions[symbol] = position + action.size
        portfolio.avg_entry_price[symbol] = price
    elif action.side == "sell":
        position = portfolio.positions.get(symbol, 0.0)
        if position < action.size:
            action.size = position
        proceeds = action.size * price
        portfolio.cash += proceeds
        new_position = position - action.size
        if new_position <= 0:
            portfolio.positions.pop(symbol, None)
            portfolio.avg_entry_price.pop(symbol, None)
        else:
            portfolio.positions[symbol] = new_position


def run_backtest(
    market_data: pd.DataFrame,
    initial_portfolio: BacktestPortfolio,
    llm_client: Any,
    symbol: str,
    timeframe: str,
    config_bounds: Dict[str, int],
) -> BacktestResult:
    """Run LLM-assisted backtest over provided market data."""

    portfolio = initial_portfolio
    trades: List[BacktestAction] = []
    current_plan: Optional[StrategyPlan] = None
    indicators_cache: Dict[str, List[float]] = {}

    for idx in range(len(market_data)):
        past_data = market_data.iloc[: idx + 1]
        indicators = compute_indicators(past_data)
        indicators_cache = indicators
        needs_replan = (
            current_plan is None
            or check_replan_triggers(current_plan.replan_triggers.model_dump(), past_data, indicators, portfolio)
        )
        if needs_replan:
            snapshot = build_planner_snapshot(
                past_data=past_data,
                indicators=indicators,
                portfolio=portfolio,
                bar_index=idx,
                symbol=symbol,
                timeframe=timeframe,
                config_bounds=config_bounds,
            )
            current_plan = call_llm_strategy_planner(llm_client, snapshot)

        lookback = current_plan.clamp_lookback(idx + 1)
        window_data = market_data.iloc[max(0, idx + 1 - lookback) : idx + 1]
        window_indicators = compute_indicators(window_data)

        action = execution_engine(current_plan, window_indicators, portfolio, market_data.iloc[idx])
        simulator_step(portfolio, action, market_data.iloc[idx])
        portfolio.update_equity({symbol: market_data.iloc[idx]["close"]})
        if action:
            trades.append(action)

    return BacktestResult(trades=trades, equity_curve=portfolio.equity_curve, final_portfolio=portfolio)

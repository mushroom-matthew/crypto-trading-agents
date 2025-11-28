"""Activities invoked by Temporal workflows for planning/live trading."""

from __future__ import annotations

from typing import Any, Dict

from temporalio import activity

from services.market_data_worker import fetch_ohlcv_history, fetch_recent_prices
from services.indicator_engine import summarize_indicators
from services.planner_agent import LLMPlanner, PlannerRequest, PlannerPreferences
from services.signal_agent_service import build_market_snapshots, SignalAgentService
from services.judge_agent_service import JudgeAgentService, PortfolioState
from services.execution_agent_service import ExecutionAgentService
from services.strategy_config_store import save_plan, load_plan
from trading_core.config import DEFAULT_STRATEGY_CONFIG


@activity.defn
def fetch_market_data_activity(params: Dict[str, Any]) -> Dict[str, Any]:
    candles = fetch_ohlcv_history(params["symbol"], params["timeframe"], params.get("lookback_days", 7))
    return {"candles": [candle.__dict__ for candle in candles]}


@activity.defn
def summarize_indicators_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    candles = payload["candles"]
    candle_objs = [type("Candle", (), candle) for candle in candles]
    summary = summarize_indicators(payload["symbol"], candle_objs)
    return summary.__dict__


@activity.defn
def plan_strategy_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    planner = LLMPlanner(DEFAULT_STRATEGY_CONFIG.planner)
    request = PlannerRequest(
        symbol=payload["symbol"],
        indicators=payload["summary"],
        preferences=PlannerPreferences(risk_mode="aggressive"),
    )
    response = planner.plan(request)
    return {"plan": response.plan.model_dump(), "metadata": response.metadata}


@activity.defn
def store_strategy_config_activity(payload: Dict[str, Any]) -> None:
    save_plan(payload["symbol"], payload)


@activity.defn
def load_strategy_config_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    symbols = payload.get("symbols")
    result: Dict[str, Any] = {}
    if symbols:
        plan_map = {}
        for symbol in symbols:
            entry = load_plan(symbol)
            if entry:
                plan_map[symbol] = entry
        result["plans"] = plan_map
    else:
        plan = load_plan(payload["symbol"])
        result["plans"] = {payload["symbol"]: plan} if plan else {}
    return result


@activity.defn
def build_snapshots_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    summaries: Dict[str, Dict[str, Any]] = {}
    timeframe = payload.get("timeframe", "1h")
    for symbol in payload["symbols"]:
        candles = fetch_ohlcv_history(symbol, timeframe, lookback_days=payload.get("lookback_days", 2))
        summary = summarize_indicators(symbol, candles)
        summaries[symbol] = {
            "price": candles[-1].close,
            "rolling_high": summary.rolling_high,
            "rolling_low": summary.rolling_low,
            "recent_max": summary.rolling_high,
            "atr": summary.atr,
            "volume_multiple": summary.volume_multiple,
        }
    plan_map = payload.get("plan_map")
    snapshots = build_market_snapshots(summaries, plan_map)
    return {symbol: snapshot.__dict__ for symbol, snapshot in snapshots.items()}


@activity.defn
def generate_signals_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    service = SignalAgentService(DEFAULT_STRATEGY_CONFIG)
    snapshots = build_market_snapshots(payload["snapshots"], payload.get("plan_map"))
    intents = service.generate(snapshots)
    return {"intents": intents}


@activity.defn
def judge_intents_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    service = JudgeAgentService(DEFAULT_STRATEGY_CONFIG)
    portfolio = PortfolioState(**payload["portfolio_state"])
    intents = [type("Intent", (), intent) for intent in payload["intents"]]
    judgements = service.evaluate(portfolio, intents)
    return {"judgements": judgements}


@activity.defn
def execute_orders_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    executor = ExecutionAgentService()
    receipts = executor.execute(payload.get("approved_intents", []))
    return {"receipts": receipts}

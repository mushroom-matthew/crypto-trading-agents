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
    return response.plan.model_dump()


@activity.defn
def store_strategy_config_activity(payload: Dict[str, Any]) -> None:
    save_plan(payload["symbol"], payload["plan"])


@activity.defn
def load_strategy_config_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    plan = load_plan(payload["symbol"])
    return {"plan": plan}


@activity.defn
def build_snapshots_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    prices = fetch_recent_prices(payload["symbols"])
    summaries = {
        symbol: {
            "price": prices[symbol],
            "rolling_high": payload.get("rolling_high", prices[symbol] * 1.05),
            "rolling_low": payload.get("rolling_low", prices[symbol] * 0.95),
            "recent_max": payload.get("recent_max", prices[symbol]),
            "atr": 1.0,
            "atr_band": 2.0,
            "volume_multiple": 1.0,
            "volume_floor": 0.5,
        }
        for symbol in payload["symbols"]
    }
    snapshots = build_market_snapshots(summaries)
    return {symbol: snapshot.__dict__ for symbol, snapshot in snapshots.items()}


@activity.defn
def generate_signals_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    service = SignalAgentService(DEFAULT_STRATEGY_CONFIG)
    snapshots = build_market_snapshots(payload["snapshots"])
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

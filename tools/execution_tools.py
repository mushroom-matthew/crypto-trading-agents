"""Tools exposing the deterministic execution engine."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from schemas.compiled_plan import CompiledPlan
from schemas.judge_feedback import JudgeFeedback, JudgeConstraints
from schemas.llm_strategist import StrategyPlan
from services.strategy_run_registry import registry
from trading_core.execution_engine import ExecutionEngine, TradeEvent


engine = ExecutionEngine()


def _ensure_list(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, list):
        return payload
    raise TypeError("Expected list payload for trigger events")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _determine_constraints(run, judge_payload: Any | None) -> JudgeConstraints:
    if judge_payload:
        feedback = JudgeFeedback.model_validate(judge_payload)
        return feedback.constraints
    if getattr(run, "latest_judge_action", None):
        action = run.latest_judge_action
        if action and action.status == "applied" and action.evals_remaining > 0:
            return action.constraints
    if run.latest_judge_feedback:
        return run.latest_judge_feedback.constraints
    return JudgeFeedback().constraints


def _serialize_event(event: TradeEvent) -> Dict[str, Any]:
    return {
        "timestamp": event.timestamp.isoformat(),
        "trigger_id": event.trigger_id,
        "symbol": event.symbol,
        "action": event.action,
        "reason": event.reason,
        "detail": event.detail,
    }


def simulate_day_tool(
    run_id: str,
    strategy_plan_payload: Any,
    compiled_plan_payload: Any,
    trigger_events_payload: Any,
    judge_feedback_payload: Any | None = None,
) -> Dict[str, Any]:
    """Simulate one day of trigger evaluations and enforce constraints."""

    run = registry.get_strategy_run(run_id)
    strategy_plan = StrategyPlan.model_validate(strategy_plan_payload)
    compiled_plan = CompiledPlan.model_validate(compiled_plan_payload)
    if strategy_plan.run_id != run.run_id or compiled_plan.run_id != run.run_id:
        raise ValueError("StrategyPlan/CompiledPlan run_id mismatch")
    constraints = _determine_constraints(run, judge_feedback_payload)
    events = _ensure_list(trigger_events_payload)

    engine.reset_run(run_id)
    trigger_map = {trigger.trigger_id: trigger for trigger in compiled_plan.triggers}
    processed: List[Dict[str, Any]] = []
    skip_counts: Dict[str, int] = {}
    executed = 0
    last_day: str | None = None

    for raw_event in sorted(events, key=lambda e: e["timestamp"]):
        trigger_id = raw_event["trigger_id"]
        trigger = trigger_map.get(trigger_id)
        if not trigger:
            continue
        timestamp = _parse_timestamp(raw_event["timestamp"])
        last_day = timestamp.date().isoformat()
        event = engine.evaluate_trigger(run, compiled_plan, strategy_plan, constraints, trigger, timestamp)
        if not event:
            continue
        processed.append(_serialize_event(event))
        if event.action == "executed":
            executed += 1
        elif event.reason:
            skip_counts[event.reason] = skip_counts.get(event.reason, 0) + 1

    trades_attempted = executed + sum(skip_counts.values())
    return {
        "run_id": run_id,
        "plan_id": strategy_plan.plan_id,
        "executed": executed,
        "trades_attempted": trades_attempted,
        "skipped": skip_counts,
        "events": processed,
        "plan_trades": engine.plan_trades(run_id, strategy_plan.plan_id, last_day),
    }


def run_live_step_tool(
    run_id: str,
    strategy_plan_payload: Any,
    compiled_plan_payload: Any,
    trigger_events_payload: Any,
    judge_feedback_payload: Any | None = None,
) -> Dict[str, Any]:
    """Process a batch of trigger events without resetting day state."""

    run = registry.get_strategy_run(run_id)
    strategy_plan = StrategyPlan.model_validate(strategy_plan_payload)
    compiled_plan = CompiledPlan.model_validate(compiled_plan_payload)
    constraints = _determine_constraints(run, judge_feedback_payload)
    events = _ensure_list(trigger_events_payload)

    trigger_map = {trigger.trigger_id: trigger for trigger in compiled_plan.triggers}
    processed: List[Dict[str, Any]] = []
    executed = 0
    skip_counts: Dict[str, int] = {}
    last_day: str | None = None

    for raw_event in events:
        trigger = trigger_map.get(raw_event["trigger_id"])
        if not trigger:
            continue
        timestamp = _parse_timestamp(raw_event["timestamp"])
        last_day = timestamp.date().isoformat()
        event = engine.evaluate_trigger(run, compiled_plan, strategy_plan, constraints, trigger, timestamp)
        if not event:
            continue
        processed.append(_serialize_event(event))
        if event.action == "executed":
            executed += 1
        elif event.reason:
            skip_counts[event.reason] = skip_counts.get(event.reason, 0) + 1

    trades_attempted = executed + sum(skip_counts.values())
    return {
        "run_id": run_id,
        "plan_id": strategy_plan.plan_id,
        "executed": executed,
        "trades_attempted": trades_attempted,
        "skipped": skip_counts,
        "events": processed,
        "plan_trades": engine.plan_trades(run_id, strategy_plan.plan_id, last_day),
    }


def reset_run_state(run_id: str) -> None:
    """Clear counters for a run (useful when a new day starts)."""

    engine.reset_run(run_id)


__all__ = ["simulate_day_tool", "run_live_step_tool", "engine", "reset_run_state"]

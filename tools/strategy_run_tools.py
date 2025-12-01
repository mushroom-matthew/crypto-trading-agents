"""Tool-style helpers for managing StrategyRun lifecycle."""

from __future__ import annotations

import json
from typing import Any, Dict

from schemas.compiled_plan import CompiledPlan
from schemas.llm_strategist import LLMInput, StrategyPlan
from schemas.strategy_run import StrategyRun, StrategyRunConfig
from services.strategy_run_registry import registry
from services.strategist_plan_service import plan_service
from trading_core.trigger_compiler import compile_plan as compile_strategy_plan, validate_plan as validate_strategy_plan, TriggerCompilationError


def _ensure_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    raise TypeError("Payload must be a dict or JSON string")


def create_strategy_run_tool(config_payload: Any) -> Dict[str, Any]:
    """Create a StrategyRun from a config JSON/dict."""

    config_dict = _ensure_dict(config_payload)
    config = StrategyRunConfig.model_validate(config_dict)
    run = registry.create_strategy_run(config)
    return run.model_dump()


def get_strategy_run_tool(run_id: str) -> Dict[str, Any]:
    """Fetch a StrategyRun by ID."""

    run = registry.get_strategy_run(run_id)
    return run.model_dump()


def update_strategy_run_tool(run_payload: Any) -> Dict[str, Any]:
    """Update a StrategyRun by providing its serialized payload."""

    run_dict = _ensure_dict(run_payload)
    run = StrategyRun.model_validate(run_dict)
    updated = registry.update_strategy_run(run)
    return updated.model_dump()


def lock_run_tool(run_id: str) -> Dict[str, Any]:
    """Lock an existing StrategyRun to prevent further edits."""

    run = registry.lock_run(run_id)
    return run.model_dump()


def generate_plan_for_run_tool(run_id: str, llm_input_payload: Any, prompt_template: str | None = None) -> Dict[str, Any]:
    """Generate a StrategyPlan for a given run_id using the strategist plan service."""

    llm_input_dict = _ensure_dict(llm_input_payload)
    llm_input = LLMInput.model_validate(llm_input_dict)
    plan = plan_service.generate_plan_for_run(run_id, llm_input, prompt_template=prompt_template)
    return plan.model_dump()


def compile_plan_tool(plan_payload: Any) -> Dict[str, Any]:
    """Compile a StrategyPlan payload into a CompiledPlan."""

    plan_dict = _ensure_dict(plan_payload)
    plan = StrategyPlan.model_validate(plan_dict)
    compiled = compile_strategy_plan(plan)
    run = registry.get_strategy_run(plan.run_id)
    run.compiled_plan_id = plan.plan_id
    run.plan_active = True
    run.current_plan_id = plan.plan_id
    registry.update_strategy_run(run)
    return compiled.model_dump()


def validate_plan_tool(plan_payload: Any) -> Dict[str, Any]:
    """Validate a StrategyPlan without mutating state."""

    plan_dict = _ensure_dict(plan_payload)
    plan = StrategyPlan.model_validate(plan_dict)
    errors = validate_strategy_plan(plan)
    return {"valid": not errors, "errors": errors}


__all__ = [
    "create_strategy_run_tool",
    "get_strategy_run_tool",
    "update_strategy_run_tool",
    "lock_run_tool",
    "generate_plan_for_run_tool",
    "compile_plan_tool",
    "validate_plan_tool",
]

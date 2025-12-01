"""MCP server exposing strategy run lifecycle tools."""

from __future__ import annotations

from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from services.tool_call_logger import logger as tool_logger
from tools import execution_tools, strategy_run_tools

app = FastMCP("strategy-run-tools")


def _record(tool_name: str, run_id: str | None, args: Any, kwargs: Dict[str, Any], result: Any) -> None:
    tool_logger.record_call(tool_name, run_id, args, kwargs, result)


@app.tool()
async def create_strategy_run(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new StrategyRun."""

    result = strategy_run_tools.create_strategy_run_tool(config)
    _record("create_strategy_run", result.get("run_id"), (config,), {}, result)
    return result


@app.tool()
async def get_strategy_run(run_id: str) -> Dict[str, Any]:
    """Fetch a StrategyRun."""

    result = strategy_run_tools.get_strategy_run_tool(run_id)
    _record("get_strategy_run", run_id, (run_id,), {}, result)
    return result


@app.tool()
async def update_strategy_run(run_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update a StrategyRun."""

    result = strategy_run_tools.update_strategy_run_tool(run_payload)
    _record("update_strategy_run", run_payload.get("run_id"), (run_payload,), {}, result)
    return result


@app.tool()
async def lock_strategy_run(run_id: str) -> Dict[str, Any]:
    """Lock a StrategyRun."""

    result = strategy_run_tools.lock_run_tool(run_id)
    _record("lock_strategy_run", run_id, (run_id,), {}, result)
    return result


@app.tool()
async def generate_plan_for_run(run_id: str, llm_input: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a StrategyPlan for a run."""

    result = strategy_run_tools.generate_plan_for_run_tool(run_id, llm_input)
    _record("generate_plan_for_run", run_id, (run_id, llm_input), {}, result)
    return result


@app.tool()
async def compile_plan(plan_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compile a StrategyPlan."""

    result = strategy_run_tools.compile_plan_tool(plan_payload)
    _record("compile_plan", plan_payload.get("run_id"), (plan_payload,), {}, result)
    return result


@app.tool()
async def validate_plan(plan_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a StrategyPlan."""

    result = strategy_run_tools.validate_plan_tool(plan_payload)
    _record("validate_plan", plan_payload.get("run_id"), (plan_payload,), {}, result)
    return result


@app.tool()
async def simulate_day(
    run_id: str,
    strategy_plan_payload: Dict[str, Any],
    compiled_plan_payload: Dict[str, Any],
    trigger_events: Any,
    judge_feedback: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Simulate a day of trigger events."""

    result = execution_tools.simulate_day_tool(
        run_id,
        strategy_plan_payload,
        compiled_plan_payload,
        trigger_events,
        judge_feedback_payload=judge_feedback,
    )
    _record(
        "simulate_day",
        run_id,
        (run_id, strategy_plan_payload, compiled_plan_payload, trigger_events, judge_feedback),
        {},
        result,
    )
    return result


@app.tool()
async def run_live_step(
    run_id: str,
    strategy_plan_payload: Dict[str, Any],
    compiled_plan_payload: Dict[str, Any],
    trigger_events: Any,
    judge_feedback: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Process trigger events in live mode."""

    result = execution_tools.run_live_step_tool(
        run_id,
        strategy_plan_payload,
        compiled_plan_payload,
        trigger_events,
        judge_feedback_payload=judge_feedback,
    )
    _record(
        "run_live_step",
        run_id,
        (run_id, strategy_plan_payload, compiled_plan_payload, trigger_events, judge_feedback),
        {},
        result,
    )
    return result


__all__ = ["app"]

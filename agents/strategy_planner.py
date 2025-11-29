"""LLM-powered strategy planner that returns StrategySpec payloads."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI
from temporalio.client import Client, RPCError, RPCStatusCode

from agents.logging_utils import setup_logging
from agents.prompt_manager import PromptManager
from agents.temporal_utils import connect_temporal
from agents.langfuse_utils import create_openai_client, init_langfuse
from agents.workflows.strategy_spec_workflow import StrategySpecWorkflow
from tools.strategy_spec import StrategySpec

logger = setup_logging(__name__)

init_langfuse()
openai_client: OpenAI = create_openai_client()
prompt_manager = PromptManager()


def _strategy_spec_prompt() -> str:
    base_prompt = prompt_manager.templates["execution_agent_standard"].render()
    schema_instructions = """
You are a deterministic trading strategy planner. Respond ONLY with JSON that can be parsed into
the following schema (StrategySpec):

StrategySpec:
  strategy_id: string (unique)
  market: string (symbol, e.g. ETH-USD)
  timeframe: string (e.g. 15m)
  mode: one of ["trend", "mean_revert", "breakout"]
  entry_conditions: array of EntryCondition
  exit_conditions: array of ExitCondition
  risk: RiskSpec
  expiry_ts: ISO timestamp or null
  allow_auto_execute: boolean

EntryCondition:
  type: one of ["breakout", "crossover", "pullback"]
  level: number or null
  direction: "above" | "below" or null
  indicator: "ema" | "sma" | "rsi" | "price" or null
  lookback: integer or null
  confirmation_candles: integer or null
  min_volume_multiple: number or null
  side: "buy" | "sell" | "close" or null

ExitCondition:
  type: one of ["take_profit", "stop_loss", "timed_exit", "trailing_stop"]
  take_profit_pct: number or null
  stop_loss_pct: number or null
  max_bars_in_trade: integer or null
  trail_pct: number or null

RiskSpec:
  max_fraction_of_balance: number
  risk_per_trade_fraction: number
  max_drawdown_pct: number
  leverage: number

Output valid JSON only. No additional commentary.
"""
    return f"{base_prompt}\n\n{schema_instructions.strip()}"


async def ensure_strategy_workflow(client: Client) -> None:
    handle = client.get_workflow_handle("strategy-spec-store")
    try:
        await handle.describe()
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            await client.start_workflow(
                StrategySpecWorkflow.run,
                id="strategy-spec-store",
                task_queue=os.environ.get("TASK_QUEUE", "mcp-tools"),
            )
            logger.info("StrategySpecWorkflow started")
        else:
            raise


async def store_strategy(client: Client, spec: StrategySpec) -> None:
    handle = client.get_workflow_handle("strategy-spec-store")
    await handle.signal("update_strategy_spec", spec.model_dump())


async def plan_strategy_spec(
    market: str,
    timeframe: str,
    risk_profile: str,
    mode: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a StrategySpec using the LLM once and store it."""
    temporal = await connect_temporal()
    await ensure_strategy_workflow(temporal)
    payload = {
        "market": market,
        "timeframe": timeframe,
        "risk_profile": risk_profile,
        "mode": mode,
        "notes": notes,
    }
    messages = [
        {"role": "system", "content": _strategy_spec_prompt()},
        {"role": "user", "content": json.dumps(payload)},
    ]
    completion = openai_client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        input=messages,
        max_output_tokens=512,
        temperature=0.2,
    )
    content = completion.output[0].content[0].text
    data = json.loads(content)
    spec = StrategySpec.model_validate(data)
    await store_strategy(temporal, spec)
    logger.info(
        "Strategy planned",
        extra={"strategy_id": spec.strategy_id, "market": market, "timeframe": timeframe},
    )
    return spec.model_dump()

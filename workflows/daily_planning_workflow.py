"""Temporal workflow for daily LLM strategy planning."""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, Any

from temporalio import workflow

from workflows.activities import (
    fetch_market_data_activity,
    summarize_indicators_activity,
    plan_strategy_activity,
    store_strategy_config_activity,
)


@workflow.defn
class DailyPlanningWorkflow:
    @workflow.run
    async def run(self, strategy_payload: Dict[str, Any]) -> Dict[str, Any]:
        candles = await workflow.execute_activity(
            fetch_market_data_activity,
            strategy_payload,
            schedule_to_close_timeout=timedelta(minutes=1),
        )
        summary = await workflow.execute_activity(
            summarize_indicators_activity,
            {"symbol": strategy_payload["symbol"], "candles": candles["candles"]},
            schedule_to_close_timeout=timedelta(minutes=1),
        )
        plan = await workflow.execute_activity(
            plan_strategy_activity,
            {"symbol": strategy_payload["symbol"], "summary": summary},
            schedule_to_close_timeout=timedelta(minutes=2),
        )
        await workflow.execute_activity(
            store_strategy_config_activity,
            {"symbol": strategy_payload["symbol"], "plan": plan},
            schedule_to_close_timeout=timedelta(minutes=1),
        )
        return {"status": "planned", "plan": plan}

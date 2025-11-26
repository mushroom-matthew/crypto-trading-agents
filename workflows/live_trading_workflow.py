"""Temporal workflow for executing live trading loops."""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, Any

from temporalio import workflow

from workflows.activities import (
    load_strategy_config_activity,
    build_snapshots_activity,
    generate_signals_activity,
    judge_intents_activity,
    execute_orders_activity,
)


@workflow.defn
class LiveTradingWorkflow:
    @workflow.run
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        await workflow.execute_activity(
            load_strategy_config_activity,
            params,
            schedule_to_close_timeout=timedelta(minutes=1),
        )
        snapshots = await workflow.execute_activity(
            build_snapshots_activity,
            {"symbols": params["symbols"]},
            schedule_to_close_timeout=timedelta(minutes=1),
        )
        intents_payload = await workflow.execute_activity(
            generate_signals_activity,
            {"snapshots": snapshots},
            schedule_to_close_timeout=timedelta(minutes=1),
        )
        judgements = await workflow.execute_activity(
            judge_intents_activity,
            {
                "portfolio_state": params["portfolio_state"],
                "intents": intents_payload["intents"],
            },
            schedule_to_close_timeout=timedelta(minutes=1),
        )
        receipts = await workflow.execute_activity(
            execute_orders_activity,
            {"approved_intents": judgements["judgements"]},
            schedule_to_close_timeout=timedelta(minutes=1),
        )
        return {
            "intents": intents_payload["intents"],
            "judgements": judgements["judgements"],
            "receipts": receipts["receipts"],
        }

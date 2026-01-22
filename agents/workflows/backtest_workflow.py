"""Workflow entrypoint for LLM-assisted backtests."""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, Any

from temporalio import workflow

from backtesting.backtest_activity import run_backtest_activity


@workflow.defn
class LLMBacktestWorkflow:
    @workflow.run
    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return await workflow.execute_activity(
            run_backtest_activity,
            request,
            schedule_to_close_timeout=timedelta(minutes=10),
        )

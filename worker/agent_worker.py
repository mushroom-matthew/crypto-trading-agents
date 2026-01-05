"""Agent-mode Temporal worker with explicit module allowlist."""

from __future__ import annotations

import importlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Sequence

from temporalio.client import Client
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from agents.runtime_mode import RuntimeMode

ALLOWED_MODULES = [
    # Activities
    "agents.activities.ledger",
    "backtesting.activities",  # Backtest execution activities
    # Agent workflows
    "agents.workflows.backtest_workflow",
    "agents.workflows.broker_agent_workflow",
    "agents.workflows.ensemble_workflow",
    "agents.workflows.execution_agent_workflow",
    "agents.workflows.execution_ledger_workflow",
    "agents.workflows.judge_agent_workflow",
    "agents.workflows.strategy_spec_workflow",
    # Durable tools
    "tools.backtest_execution",  # Backtest workflow
    "tools.ensemble_nudge",
    "tools.execution",
    "tools.feature_engineering",
    "tools.market_data",
    "tools.strategy_signal",
]


def _import_modules() -> list[Any]:
    modules: list[Any] = []
    for name in ALLOWED_MODULES:
        try:
            modules.append(importlib.import_module(name))
        except Exception as exc:  # pragma: no cover - import errors should be surfaced
            print(f"[agent_worker] Failed to import {name}: {exc}", file=sys.stderr)
            raise
    _assert_no_legacy_modules()
    return modules


def _assert_no_legacy_modules() -> None:
    """Guard against accidental legacy imports in agent mode."""
    for name in sys.modules:
        if name == "workflows" or name.startswith("workflows.") or name.startswith("legacy."):
            raise RuntimeError(
                f"[agent_worker] Detected legacy module import {name!r} while TRADING_STACK=agent"
            )


def _collect_definitions(modules: Iterable[Any]) -> tuple[Sequence[type], Sequence[Any]]:
    """Collect unique workflow and activity definitions from modules."""
    workflows: set[type] = set()
    activities: set[Any] = set()
    for module in modules:
        for obj in module.__dict__.values():
            if hasattr(obj, "__temporal_workflow_definition"):
                workflows.add(obj)
            elif hasattr(obj, "__temporal_activity_definition"):
                activities.add(obj)
    return list(workflows), list(activities)


async def run_worker(runtime: RuntimeMode) -> None:
    if runtime.stack != "agent":
        raise RuntimeError(f"[agent_worker] Invalid stack for agent worker: {runtime.stack}")

    modules = _import_modules()
    workflows, activities = _collect_definitions(modules)
    print(f"[agent_worker] Loaded {len(workflows)} workflows and {len(activities)} activities")

    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    client = await Client.connect(address, namespace=namespace)
    task_queue = os.environ.get("TASK_QUEUE", "mcp-tools")

    with ThreadPoolExecutor() as activity_executor:
        worker = Worker(
            client,
            task_queue=task_queue,
            workflows=workflows,
            activities=activities,
            activity_executor=activity_executor,
            workflow_runner=UnsandboxedWorkflowRunner(),
        )
        await worker.run()

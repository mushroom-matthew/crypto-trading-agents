"""Legacy/live Temporal worker with opt-in import discovery."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Sequence
import os

from temporalio.client import Client
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from agents.runtime_mode import RuntimeMode

BASE_PACKAGES = ["tools", "workflows", "services", "agents"]


def _discover_modules() -> Iterable[Any]:
    """Discover modules under the legacy/live surface."""
    for pkg_name in BASE_PACKAGES:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as exc:  # pragma: no cover - import errors should be surfaced
            print(f"[legacy_worker] Failed to import {pkg_name}: {exc}", file=sys.stderr)
            continue
        yield pkg
        if hasattr(pkg, "__path__"):
            for _, name, _ in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
                try:
                    module = importlib.import_module(name)
                    yield module
                except Exception as exc:  # pragma: no cover
                    print(f"[legacy_worker] Failed to import {name}: {exc}", file=sys.stderr)
                    continue


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
    if runtime.stack != "legacy_live":
        raise RuntimeError(f"[legacy_worker] Invalid stack for legacy worker: {runtime.stack}")

    modules = list(_discover_modules())
    workflows, activities = _collect_definitions(modules)
    print(f"[legacy_worker] Loaded {len(workflows)} workflows and {len(activities)} activities")

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

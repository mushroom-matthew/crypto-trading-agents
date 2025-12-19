"""Runtime router for Temporal workers based on TRADING_STACK."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _add_project_root_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_project_root_to_path()

from agents.runtime_mode import get_runtime_mode  # noqa: E402
from worker.agent_worker import run_worker as run_agent_worker  # noqa: E402
from worker.legacy_live_worker import run_worker as run_legacy_worker  # noqa: E402


async def main() -> None:
    runtime = get_runtime_mode()
    print(f"[worker/run] Starting worker with {runtime.banner}")
    if runtime.stack == "agent":
        await run_agent_worker(runtime)
    elif runtime.stack == "legacy_live":
        await run_legacy_worker(runtime)
    else:  # pragma: no cover - guarded by get_runtime_mode()
        raise RuntimeError(f"Unsupported TRADING_STACK={runtime.stack}")


if __name__ == "__main__":
    asyncio.run(main())

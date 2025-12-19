#!/usr/bin/env python
"""Backward-compatible worker entrypoint routing to ``worker/run.py``."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys


def _add_project_root_to_path() -> None:
    """Add repository root directory to ``sys.path``."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


async def main() -> None:
    _add_project_root_to_path()
    from worker.run import main as run_main

    await run_main()


if __name__ == "__main__":
    asyncio.run(main())

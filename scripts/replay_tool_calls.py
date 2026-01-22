"""Replay recorded tool calls to verify deterministic outputs."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict

from mcp_server import strategy_tools_server


def _load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def replay(records_path: Path) -> None:
    records = _load_records(records_path)
    success = 0
    for record in records:
        tool_name = record["tool_name"]
        func = getattr(strategy_tools_server, tool_name, None)
        if not func:
            print(f"Skipping {tool_name}: not found on server")
            continue
        args = record["args"]
        kwargs = record["kwargs"]
        result = await func(*args, **kwargs)
        if result != record["result"]:
            raise AssertionError(f"Mismatch for {tool_name}: expected {record['result']} got {result}")
        success += 1
    print(f"Replayed {success} calls successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay tool call logs for determinism.")
    parser.add_argument("log_path", type=Path, help="Path to tool_calls.jsonl")
    args = parser.parse_args()
    import asyncio

    asyncio.run(replay(args.log_path))


if __name__ == "__main__":
    main()

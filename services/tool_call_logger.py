"""Lightweight tool-call recorder for deterministic replay."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class ToolCallRecord:
    timestamp: str
    tool_name: str
    run_id: str | None
    args: Any
    kwargs: Dict[str, Any]
    result: Any
    git_sha: str | None


class ToolCallLogger:
    """Appends structured tool call logs for later replay."""

    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or Path("logs/tool_calls")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.git_sha = self._get_git_sha()

    def _get_git_sha(self) -> str | None:
        version = os.environ.get("GIT_SHA")
        if version:
            return version
        try:
            import subprocess

            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            return sha
        except Exception:
            return None

    def _default(self, obj):  # type: ignore[override]
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return str(obj)

    def log(self, record: ToolCallRecord) -> None:
        path = self.log_dir / "tool_calls.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), default=self._default) + "\n")

    def record_call(self, tool_name: str, run_id: str | None, args: Any, kwargs: Dict[str, Any], result: Any) -> None:
        record = ToolCallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_name=tool_name,
            run_id=run_id,
            args=args,
            kwargs=kwargs,
            result=result,
            git_sha=self.git_sha,
        )
        self.log(record)


logger = ToolCallLogger()

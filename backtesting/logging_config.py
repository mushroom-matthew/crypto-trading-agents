"""Logging helpers for backtesting CLI and engines."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


class _JSONFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_backtest_logging(
    level: str = "INFO",
    log_file: str | None = None,
    json_logs: bool = False,
) -> None:
    """Configure logging for backtests.

    Parameters
    ----------
    level:
        Log level (INFO, DEBUG, etc.)
    log_file:
        Optional path to append logs to. If omitted, only stdout is used.
    json_logs:
        Emit JSON-per-line logs instead of textual format.
    """

    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    formatter: logging.Formatter = _JSONFormatter() if json_logs else logging.Formatter(fmt=fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers: list[logging.Handler] = [stream_handler]

    if log_file:
        log_path = Path(log_file)
        if log_path.is_dir():
            log_path = log_path / f"backtest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root.setLevel(resolved_level)
    for handler in handlers:
        root.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("temporalio").setLevel(logging.WARNING)


__all__ = ["setup_backtest_logging"]

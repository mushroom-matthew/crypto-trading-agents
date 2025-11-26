"""Configuration helpers for execution agent gating."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ExecutionGatingConfig:
    """User-configurable thresholds for execution-agent LLM calls."""

    min_price_move_pct: float = 0.5
    max_staleness_seconds: int = 1800
    max_calls_per_hour_per_symbol: int = 60


def load_execution_gating_config() -> ExecutionGatingConfig:
    """Load execution gating config from environment with defaults."""
    return ExecutionGatingConfig(
        min_price_move_pct=float(
            os.getenv("EXECUTION_MIN_PRICE_MOVE_PCT", 0.5)
        ),
        max_staleness_seconds=int(
            os.getenv("EXECUTION_MAX_STALENESS_SECONDS", 1800)
        ),
        max_calls_per_hour_per_symbol=int(
            os.getenv("EXECUTION_MAX_CALLS_PER_HOUR_PER_SYMBOL", 60)
        ),
    )

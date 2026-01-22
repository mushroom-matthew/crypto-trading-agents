"""Runtime mode configuration and safety latches."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

Stack = Literal["agent", "legacy_live"]
Mode = Literal["paper", "live"]

DEFAULT_STACK: Stack = "agent"
DEFAULT_MODE: Mode = "paper"


def _read_env(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value.strip().lower()


@dataclass(frozen=True)
class RuntimeMode:
    """Resolved runtime mode with safety constraints."""

    stack: Stack
    mode: Mode
    live_trading_ack: bool
    ui_unlock: bool

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def banner(self) -> str:
        suffix = "LIVE" if self.is_live else "PAPER"
        return f"[{self.stack}:{suffix}]"

    def ensure_valid(self) -> None:
        if self.stack not in {"agent", "legacy_live"}:
            raise ValueError(f"Invalid TRADING_STACK={self.stack!r}")
        if self.mode not in {"paper", "live"}:
            raise ValueError(f"Invalid TRADING_MODE={self.mode!r}")
        if self.is_live and not self.live_trading_ack:
            raise RuntimeError(
                "Live trading requested but LIVE_TRADING_ACK=true not set; refusing to start."
            )


@lru_cache(maxsize=1)
def get_runtime_mode() -> RuntimeMode:
    """Resolve runtime mode from environment and enforce latches."""

    stack: Stack = _read_env("TRADING_STACK", DEFAULT_STACK)  # type: ignore[assignment]
    mode: Mode = _read_env("TRADING_MODE", DEFAULT_MODE)  # type: ignore[assignment]
    live_trading_ack = _read_env("LIVE_TRADING_ACK", "false") in {"1", "true", "yes"}
    ui_unlock = _read_env("LIVE_TRADING_UI_UNLOCK", "false") in {"1", "true", "yes"}

    runtime = RuntimeMode(
        stack=stack,
        mode=mode,
        live_trading_ack=live_trading_ack,
        ui_unlock=ui_unlock,
    )
    runtime.ensure_valid()
    return runtime

"""Learning gate schemas: thresholds, kill switches, and gate status."""

from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from .llm_strategist import SerializableModel


class LearningGateThresholds(SerializableModel):
    """Market-condition thresholds that close the learning gate."""

    volatility_spike_multiple: float = Field(
        default=3.0, ge=1.0,
        description="Close gate when realized vol exceeds this multiple of median vol.",
    )
    liquidity_thin_volume_multiple: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Close gate when volume drops below this fraction of median volume.",
    )
    spread_wide_pct: float = Field(
        default=1.0, ge=0.0,
        description="Close gate when bid-ask spread exceeds this % of mid price.",
    )


class LearningKillSwitchConfig(SerializableModel):
    """Kill switches that disable the learning book for the rest of the day."""

    daily_loss_limit_pct: float = Field(
        default=1.0, ge=0.0,
        description="Cumulative learning loss as % of equity that triggers kill switch.",
    )
    consecutive_loss_limit: int = Field(
        default=3, ge=1,
        description="Number of consecutive learning losses that triggers kill switch.",
    )


class LearningGateStatus(SerializableModel):
    """Evaluation result from the learning gate."""

    open: bool = Field(default=True, description="Whether the learning gate is open (True = learning allowed).")
    reasons: List[str] = Field(default_factory=list, description="Reasons the gate is closed, if any.")

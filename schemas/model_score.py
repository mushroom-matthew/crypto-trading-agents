"""Model integration contract.

The model returns a ModelScorePacket. The risk engine consumes it for:
  1. Hard gate:  p_false_breakout > 0.40 → block entry
  2. Sizing:     size_multiplier = clamp(0.5 + 1.0 * (quality - 0.5), 0.5, 1.25)

When model_quality_score is None, the engine behaves as if no model exists:
  - size_multiplier = 1.0 (no effect)
  - hard gate = False (not blocked)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ModelScorePacket(BaseModel):
    """Output contract from ModelScorer.score(). Consumed by risk engine."""

    model_config = {"extra": "forbid"}

    model_quality_score: Optional[float] = Field(
        default=None,
        description="Overall setup quality (0–1). None if model not available.",
    )
    p_cont_1R: Optional[float] = Field(
        default=None,
        description="P(price hits 1R before stop within TTL).",
    )
    p_false_breakout: Optional[float] = Field(
        default=None,
        description=(
            "P(price returns inside compression range and hits stop within TTL). "
            "Hard gate threshold: > 0.40 blocks entry."
        ),
    )
    p_atr_expand: Optional[float] = Field(
        default=None,
        description="P(ATR expands > 1.5x over next 20 bars).",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Model artifact version (e.g., 'lgbm-v1.0.2').",
    )
    calibration_bucket: Optional[str] = Field(
        default=None,
        description=(
            "Calibration group for this prediction (e.g., 'crypto_largecap'). "
            "Used to stratify reliability analysis."
        ),
    )

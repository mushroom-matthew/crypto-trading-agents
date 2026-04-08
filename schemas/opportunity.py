"""Opportunity scanner schemas.

Runbook 74: OpportunityCard Scorer and Scanner Service.

Every symbol in the trading universe is scored every 5-15 min.
The top-N cards form the scanner ranking consumed by the AI planner (R76)
and displayed in the scanner UI panel (R75).

Score formula:
  opportunity_score = 0.28*vol_edge + 0.24*structure_edge + 0.18*trend_edge
                    + 0.20*liquidity_score - 0.07*spread_penalty - 0.03*instability_penalty
  opportunity_score_norm = clamp((opportunity_score + 0.10) / 1.00, 0, 1)
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class OpportunityCard(BaseModel):
    """Per-symbol opportunity score with component breakdown.

    Consumers:
    - ops_api/routers/scanner.py: GET /scanner/opportunities
    - services/session_planner.py: feeds SessionIntent (R76)
    - PaperTradingWorkflow._generate_plan(): injected as OPPORTUNITY_CONTEXT block
    """

    model_config = {"extra": "forbid"}

    symbol: str
    opportunity_score: float = Field(
        description="Raw weighted score. Theoretical range [-0.10, 0.90]."
    )
    opportunity_score_norm: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized to [0, 1]. Consumers should use this for ranking.",
    )

    # Component scores (all normalized to [0, 1] before weighting)
    vol_edge: float = Field(
        ge=0.0,
        le=1.0,
        description="ATR expansion vs 20-bar mean. >1.0 ATR ratio clamped to 1. "
        "High = favourable volatility for entries.",
    )
    structure_edge: float = Field(
        ge=0.0,
        le=1.0,
        description="Level count and clarity from StructureSnapshot. "
        "len(levels)/10 clamped to 1, bonus for levels near price.",
    )
    trend_edge: float = Field(
        ge=0.0,
        le=1.0,
        description="EMA alignment + ADX/50 clamped to 1. "
        "High = directional momentum present.",
    )
    liquidity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="volume_ratio from indicator snapshot (volume / 20-bar mean volume), "
        "clamped to 1.",
    )
    spread_penalty: float = Field(
        ge=0.0,
        le=1.0,
        description="Bid/ask spread as fraction of mid price, clamped to 1. "
        "0 = tight spread (good), 1 = very wide (bad).",
    )
    instability_penalty: float = Field(
        ge=0.0,
        le=1.0,
        description="consecutive_price_failures / 3 clamped to 1. "
        "0 = stable feed, 1 = frequent failures.",
    )

    expected_hold_horizon: Literal["scalp", "intraday", "swing"] = Field(
        description="Derived from vol/trend/structure context. "
        "scalp: ATR contracting; swing: trend + structure aligned; else intraday."
    )

    scored_at: datetime
    indicator_as_of: datetime = Field(
        description="as_of timestamp of the IndicatorSnapshot used for scoring. "
        "Used for freshness checks by consumers."
    )

    component_explanation: Dict[str, str] = Field(
        default_factory=dict,
        description="Human-readable explanation per component. "
        "Keys: vol_edge, structure_edge, trend_edge, liquidity_score, "
        "spread_penalty, instability_penalty, expected_hold_horizon.",
    )

    # Optional: populated when structure snapshot is available
    nearest_support: Optional[float] = Field(
        default=None, description="Nearest support level price from structure snapshot."
    )
    nearest_resistance: Optional[float] = Field(
        default=None,
        description="Nearest resistance level price from structure snapshot.",
    )
    structure_levels_count: int = Field(
        default=0,
        description="Total number of structure levels in snapshot.",
    )


class OpportunityRanking(BaseModel):
    """Ranked list of opportunity cards from the most recent scanner run."""

    model_config = {"extra": "forbid"}

    ranked_at: datetime
    cards: List[OpportunityCard] = Field(
        description="Sorted by opportunity_score_norm descending. "
        "Caller should assume this list is pre-sorted."
    )
    universe_size: int = Field(description="Total number of symbols scored in this run.")
    scan_duration_ms: int = Field(description="Wall-clock time for the full scan in ms.")
    top_n: int = Field(
        default=10, description="How many cards are included (may be less than universe_size)."
    )

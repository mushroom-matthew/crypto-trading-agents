"""
SignalEvent schema.

DISCLAIMER: Signals are research observations, not personalized investment advice.
They carry no sizing. Subscribers apply their own risk rules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SignalEvent(BaseModel):
    """One emitted signal from the strategy engine.

    A signal is a research observation: the engine detected a setup and recorded
    its projections at that moment. Sizing is NOT included — each execution adapter
    applies its own risk policy.

    Fields are append-only after emission. The Signal Ledger adds outcome fields
    once the signal is resolved.
    """

    model_config = {"extra": "forbid"}

    # --- Identity ---
    signal_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this signal (uuid4).",
    )
    engine_version: str = Field(
        description=(
            "Semver of the strategy engine at emission time (e.g., '1.4.2'). "
            "Protects track-record integrity across strategy changes. "
            "Track-record analysis must be stratified by engine_version."
        ),
    )
    ts: datetime = Field(
        description="UTC timestamp when the trigger evaluation fired.",
    )
    valid_until: datetime = Field(
        description=(
            "UTC timestamp after which the signal is considered expired "
            "if unfilled. The reconciler marks expired signals with "
            "outcome='expired' at this time."
        ),
    )
    timeframe: str = Field(
        description="Candle timeframe on which the trigger fired (e.g., '1h', '15m').",
    )

    # --- Instrument ---
    symbol: str = Field(
        description="Trading pair symbol (e.g., 'BTC-USD').",
    )
    direction: Literal["long", "short"] = Field(
        description="Intended direction of the trade.",
    )

    # --- Trigger provenance ---
    trigger_id: str = Field(
        description="ID of the TriggerCondition that generated this signal.",
    )
    strategy_type: str = Field(
        description=(
            "Strategy template in use (e.g., 'compression_breakout', "
            "'mean_reversion', 'trend_continuation')."
        ),
    )
    regime_snapshot_hash: str = Field(
        description=(
            "SHA-256 hex digest of the IndicatorSnapshot dict at signal time. "
            "Proves the signal was generated from conditions that preceded the "
            "subsequent price move, not retrofitted."
        ),
    )

    # --- Price levels ---
    entry_price: float = Field(
        description="Signal entry price (close of the trigger bar).",
    )
    stop_price_abs: float = Field(
        description=(
            "Absolute stop price resolved at signal time. "
            "For longs: price below which the setup is invalidated. "
            "Source: TradeLeg.stop_price_abs (Runbook 42)."
        ),
    )
    target_price_abs: float = Field(
        description=(
            "Absolute profit target price resolved at signal time. "
            "Source: TradeLeg.target_price_abs (Runbook 42)."
        ),
    )
    stop_anchor_type: Optional[str] = Field(
        default=None,
        description=(
            "How the stop was computed: 'pct', 'atr', 'htf_daily_low', "
            "'donchian_lower', 'candle_low', 'manual', etc."
        ),
    )
    target_anchor_type: Optional[str] = Field(
        default=None,
        description=(
            "How the target was computed: 'measured_move', 'htf_daily_high', "
            "'r_multiple_2', 'r_multiple_3', 'fib_618_above', etc."
        ),
    )

    # --- Risk projections ---
    risk_r_multiple: float = Field(
        description=(
            "Projected R to first target: "
            "(target_price_abs - entry_price) / (entry_price - stop_price_abs). "
            "For longs. Flip sign for shorts. "
            "Negative values indicate an invalid signal (target is on the wrong side of entry)."
        ),
    )
    expected_hold_bars: int = Field(
        description=(
            "LLM-estimated number of bars to hold before target or stop is hit. "
            "Used with timeframe to compute valid_until."
        ),
    )

    # --- Qualitative context ---
    thesis: str = Field(
        description="One or two sentence explanation of the setup (from LLM strategist).",
    )
    screener_rank: Optional[int] = Field(
        default=None,
        description=(
            "Rank of this symbol in the universe screener output that triggered "
            "the strategy cycle (1 = top candidate). Null if screener not active."
        ),
    )
    confidence: Optional[str] = Field(
        default=None,
        description=(
            "LLM confidence in the signal: 'high', 'medium', or 'low'. "
            "Null if not provided by the strategist."
        ),
    )

    # --- Setup event linkage (added by Runbook 44) ---
    setup_event_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of the SetupEvent (break_attempt state) that preceded this signal. "
            "Null if the trigger fired outside a tracked setup lifecycle."
        ),
    )
    feature_schema_version: str = Field(
        default="1.2.0",
        description="IndicatorSnapshot schema version at emission time.",
    )
    strategy_template_version: Optional[str] = Field(
        default=None,
        description="Active strategy template name (e.g., 'compression_breakout_v1').",
    )
    model_score: Optional[dict] = Field(
        default=None,
        description=(
            "Serialized ModelScorePacket at signal emission time. "
            "Null if NullModelScorer active."
        ),
    )

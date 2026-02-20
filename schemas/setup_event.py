"""
SetupEvent schema — frozen research record at a setup state transition.

DISCLAIMER: Setup events are research observations, not personalized investment
advice. They carry no sizing. Subscribers apply their own risk rules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


SetupState = Literal[
    "compression_candidate",
    "break_attempt",
    "confirmed",
    "failed",
    "ttl_expired",
]


class SessionContext(BaseModel):
    """Lightweight session embedding for domain/time conditioning."""

    model_config = {"extra": "forbid"}

    session_type: str = Field(
        description=(
            "Session identifier: 'crypto_utc_day', 'crypto_asia', "
            "'crypto_london', 'crypto_us', 'crypto_close'."
        ),
    )
    time_in_session_sin: float = Field(
        description="Sine of fractional time-in-session (cyclic encoding).",
    )
    time_in_session_cos: float = Field(
        description="Cosine of fractional time-in-session (cyclic encoding).",
    )
    minutes_to_session_close: Optional[float] = Field(
        default=None,
        description="Minutes until session ends. None for continuous markets.",
    )
    is_weekend: bool = Field(
        description="True if bar is in a weekend period (crypto only).",
    )
    asset_class: str = Field(
        default="crypto",
        description="Asset class domain token: 'crypto', 'equity', 'fx'.",
    )


class SetupEvent(BaseModel):
    """Frozen record of a setup state transition.

    One row per state transition. Multiple rows share the same
    `setup_chain_id` if they are the same setup lifecycle
    (compression → break → confirmed/failed).

    DISCLAIMER: Research observation only. Not personalized investment advice.
    """

    model_config = {"extra": "forbid"}

    # --- Identity ---
    setup_event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this individual state transition record.",
    )
    setup_chain_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description=(
            "Shared ID for all events in the same setup lifecycle "
            "(compression → break → outcome). Set at CompressionCandidate; "
            "inherited by subsequent transitions."
        ),
    )
    state: SetupState = Field(
        description="State machine state at the time of this event.",
    )

    # --- Provenance / version ---
    engine_semver: str = Field(
        description="Engine semver at emission time (from ENGINE_SEMVER constant).",
    )
    feature_schema_version: str = Field(
        description="IndicatorSnapshot schema version (from FEATURE_SCHEMA_VERSION).",
    )
    strategy_template_version: Optional[str] = Field(
        default=None,
        description=(
            "Active strategy template name (e.g., 'compression_breakout_v1', "
            "'htf_cascade_v1'). Null if no template active."
        ),
    )

    # --- Time / instrument ---
    ts: datetime = Field(
        description="UTC bar timestamp when this state transition was detected.",
    )
    symbol: str
    timeframe: str

    # --- Session context (for model domain conditioning) ---
    session: SessionContext = Field(
        description="Session and domain tokens for model conditioning.",
    )

    # --- Frozen feature snapshot ---
    feature_snapshot: Dict[str, Any] = Field(
        description=(
            "Full IndicatorSnapshot serialized to dict at the moment of this "
            "transition. Keys include all indicator fields including candlestick "
            "and htf_* fields. Values are float | bool | str | None. "
            "Do NOT update this field after creation — immutability is what "
            "makes supervised labels meaningful."
        ),
    )
    feature_snapshot_hash: str = Field(
        description=(
            "SHA-256 hex digest of feature_snapshot JSON (sorted keys). "
            "Proves the features were recorded at decision time, not retrofitted."
        ),
    )

    # --- Range context (locked at compression detection) ---
    compression_range_high: Optional[float] = Field(
        default=None,
        description=(
            "donchian_upper_short at time of CompressionCandidate detection. "
            "Used as BreakAttempt criterion."
        ),
    )
    compression_range_low: Optional[float] = Field(
        default=None,
        description="donchian_lower_short at time of CompressionCandidate detection.",
    )
    compression_atr_at_detection: Optional[float] = Field(
        default=None,
        description="atr_14 at time of CompressionCandidate detection (for R sizing).",
    )

    # --- Model score (populated if model available at this state) ---
    model_quality_score: Optional[float] = Field(
        default=None,
        description="Model setup quality score (0–1). None if model not available.",
    )
    p_cont_1R: Optional[float] = Field(
        default=None,
        description="Model-estimated probability of hitting 1R before stop.",
    )
    p_false_breakout: Optional[float] = Field(
        default=None,
        description=(
            "Model-estimated probability of false breakout. "
            "If > 0.40 at BreakAttempt, entry is BLOCKED."
        ),
    )
    p_atr_expand: Optional[float] = Field(
        default=None,
        description="Model-estimated probability of ATR expanding in next 20 bars.",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Model artifact version that produced the score.",
    )

    # --- Outcome (filled by SetupOutcomeReconciler after TTL) ---
    outcome: Optional[Literal["hit_1r", "hit_stop", "ttl_expired"]] = Field(
        default=None,
        description=(
            "Outcome after TTL: hit the 1R target, hit the stop, or expired. "
            "Null until reconciled. This is the primary training label."
        ),
    )
    outcome_ts: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the outcome was resolved.",
    )
    bars_to_outcome: Optional[int] = Field(
        default=None,
        description="Number of bars from BreakAttempt to outcome.",
    )
    mfe_pct: Optional[float] = Field(
        default=None,
        description="Maximum favorable excursion as % of entry price.",
    )
    mae_pct: Optional[float] = Field(
        default=None,
        description="Maximum adverse excursion as % of entry price.",
    )
    r_achieved: Optional[float] = Field(
        default=None,
        description=(
            "R-multiple achieved at outcome. "
            "Positive = favorable, negative = loss."
        ),
    )
    ttl_bars: int = Field(
        default=48,
        description=(
            "Number of bars from BreakAttempt detection before outcome is "
            "forced to 'ttl_expired'. Default: 48 bars (48h on 1h timeframe)."
        ),
    )

    # --- Link to execution ---
    signal_event_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of the SignalEvent that fired from this setup, if any. "
            "Null if setup expired without a trigger firing."
        ),
    )

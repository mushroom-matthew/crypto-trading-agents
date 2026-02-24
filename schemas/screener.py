"""Schemas for universe screener scoring and instrument recommendation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from schemas.llm_strategist import SerializableModel


TrendState = Literal["uptrend", "downtrend", "range", "unclear"]
VolState = Literal["low", "normal", "high", "extreme"]
Confidence = Literal["high", "medium", "low"]


class SymbolAnomalyScore(SerializableModel):
    """Per-symbol anomaly score from the screener."""

    symbol: str
    as_of: datetime
    volume_z: float
    atr_expansion: float
    range_expansion_z: float
    bb_bandwidth_pct_rank: float
    close: float
    trend_state: TrendState
    vol_state: VolState
    dist_to_prior_high_pct: float
    dist_to_prior_low_pct: float
    composite_score: float
    score_components: dict[str, Any] = Field(default_factory=dict)


class ScreenerResult(SerializableModel):
    """Output of a single screening pass."""

    run_id: str
    as_of: datetime
    universe_size: int
    top_candidates: list[SymbolAnomalyScore] = Field(default_factory=list)
    screener_config: dict[str, Any] = Field(default_factory=dict)


class InstrumentRecommendation(SerializableModel):
    """Instrument recommendation built from screener candidates."""

    selected_symbol: str
    thesis: str
    strategy_type: str
    template_id: str | None = None
    regime_view: str
    key_levels: dict[str, float] | None = None
    expected_hold_timeframe: str
    confidence: Confidence
    disqualified_symbols: list[str] = Field(default_factory=list)
    disqualification_reasons: dict[str, str] = Field(default_factory=dict)


class InstrumentRecommendationItem(SerializableModel):
    """One recommendation candidate inside a grouped shortlist."""

    symbol: str
    hypothesis: str
    template_id: str | None = None
    expected_hold_timeframe: str
    thesis: str
    confidence: Confidence
    composite_score: float
    key_levels: dict[str, float] | None = None
    rank_global: int = Field(ge=1)
    rank_in_group: int = Field(ge=1)
    score_components: dict[str, Any] = Field(default_factory=dict)


class InstrumentRecommendationGroup(SerializableModel):
    """A hypothesis + timeframe bucket for UI presentation."""

    hypothesis: str
    timeframe: str
    template_id: str | None = None
    label: str
    rationale: str
    recommendations: list[InstrumentRecommendationItem] = Field(default_factory=list)


class InstrumentRecommendationBatch(SerializableModel):
    """Grouped shortlist of screener recommendations for session-start UX."""

    run_id: str
    as_of: datetime
    source: str = "deterministic_screener_grouping"
    supported_hypotheses: list[str] = Field(default_factory=list)
    max_per_group: int = Field(default=10, ge=1)
    total_candidates_considered: int = Field(default=0, ge=0)
    groups: list[InstrumentRecommendationGroup] = Field(default_factory=list)
    annotation_meta: dict[str, Any] | None = None


class ScreenerSessionPreflight(SerializableModel):
    """Payload for paper/live session-start screens."""

    mode: Literal["paper", "live"]
    as_of: datetime
    screener_run_id: str
    shortlist: InstrumentRecommendationBatch
    suggested_default_symbol: str | None = None
    suggested_default_template_id: str | None = None
    notes: list[str] = Field(default_factory=list)

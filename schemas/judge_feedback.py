"""Schemas for judge feedback and constraints."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import Field

from .llm_strategist import SerializableModel


class DisplayConstraints(SerializableModel):
    """Human-readable guidance surfaced to the strategist."""

    must_fix: List[str] = Field(default_factory=list)
    vetoes: List[str] = Field(default_factory=list)
    boost: List[str] = Field(default_factory=list)
    regime_correction: Optional[str] = None
    sizing_adjustments: Dict[str, str] = Field(default_factory=dict)


class JudgeConstraints(SerializableModel):
    """Machine-readable knobs that the executor must enforce."""

    max_trades_per_day: Optional[int] = Field(default=None, ge=0)
    min_trades_per_day: Optional[int] = Field(default=None, ge=0)
    symbol_risk_multipliers: Dict[str, float] = Field(default_factory=dict)
    disabled_trigger_ids: List[str] = Field(default_factory=list)
    disabled_categories: List[str] = Field(default_factory=list)
    risk_mode: Literal["normal", "conservative", "emergency"] = "normal"


class JudgeFeedback(SerializableModel):
    """Structured evaluation payload produced by the judge."""

    score: Optional[float] = None
    notes: Optional[str] = None
    constraints: JudgeConstraints = Field(default_factory=JudgeConstraints)
    strategist_constraints: DisplayConstraints = Field(default_factory=DisplayConstraints)

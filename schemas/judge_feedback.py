"""Schemas for judge feedback and constraints."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator

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
    max_triggers_per_symbol_per_day: Optional[int] = Field(default=None, ge=0)
    symbol_risk_multipliers: Dict[str, float] = Field(default_factory=dict)
    disabled_trigger_ids: List[str] = Field(default_factory=list)
    disabled_categories: List[str] = Field(default_factory=list)
    risk_mode: Literal["normal", "conservative", "emergency"] = "normal"

    @field_validator("disabled_categories")
    @classmethod
    def validate_disabled_categories(cls, values: List[str]) -> List[str]:
        entry_categories = {
            "trend_continuation",
            "reversal",
            "volatility_breakout",
            "mean_reversion",
        }
        all_categories = entry_categories | {"emergency_exit", "other"}
        cleaned: List[str] = []
        seen: set[str] = set()
        for value in values or []:
            if value in all_categories and value not in seen:
                cleaned.append(value)
                seen.add(value)
        disabled_entry = entry_categories & set(cleaned)
        if disabled_entry == entry_categories:
            for value in list(reversed(cleaned)):
                if value in entry_categories:
                    cleaned.remove(value)
                    break
        return cleaned


class JudgeFeedback(SerializableModel):
    """Structured evaluation payload produced by the judge."""

    score: Optional[float] = None
    notes: Optional[str] = None
    constraints: JudgeConstraints = Field(default_factory=JudgeConstraints)
    strategist_constraints: DisplayConstraints = Field(default_factory=DisplayConstraints)

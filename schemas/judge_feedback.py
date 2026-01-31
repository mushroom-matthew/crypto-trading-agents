"""Schemas for judge feedback and constraints."""

from __future__ import annotations

from typing import ClassVar, Dict, FrozenSet, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

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

    ENTRY_CATEGORIES: ClassVar[FrozenSet[str]] = frozenset({
        "trend_continuation",
        "reversal",
        "volatility_breakout",
        "mean_reversion",
    })
    MIN_ENABLED_ENTRY_CATEGORIES: ClassVar[int] = 2
    MIN_ENABLED_ENTRY_TRIGGERS: ClassVar[int] = 2

    @field_validator("disabled_categories")
    @classmethod
    def validate_disabled_categories(cls, values: List[str]) -> List[str]:
        entry_categories = cls.ENTRY_CATEGORIES
        all_categories = entry_categories | {"emergency_exit", "other"}
        cleaned: List[str] = []
        seen: set[str] = set()
        for value in values or []:
            if value in all_categories and value not in seen:
                cleaned.append(value)
                seen.add(value)
        # Enforce N-2 rule: at least MIN_ENABLED_ENTRY_CATEGORIES must remain enabled
        disabled_entry = [v for v in cleaned if v in entry_categories]
        max_disabled = len(entry_categories) - cls.MIN_ENABLED_ENTRY_CATEGORIES
        while len(disabled_entry) > max_disabled:
            to_remove = disabled_entry.pop()
            cleaned.remove(to_remove)
        return cleaned


class JudgeFeedback(SerializableModel):
    """Structured evaluation payload produced by the judge."""

    score: Optional[float] = None
    notes: Optional[str] = None
    constraints: JudgeConstraints = Field(default_factory=JudgeConstraints)
    strategist_constraints: DisplayConstraints = Field(default_factory=DisplayConstraints)


def apply_trigger_floor(
    constraints: JudgeConstraints,
    triggers: list,
    *,
    min_enabled: int | None = None,
) -> JudgeConstraints:
    """Trim ``disabled_trigger_ids`` so at least *min_enabled* entry triggers remain.

    Entry triggers are those whose ``category`` is in
    ``JudgeConstraints.ENTRY_CATEGORIES``.  If disabling the requested IDs
    would leave fewer than *min_enabled* entry triggers active, the most
    recently added disabled IDs are dropped until the floor is met.

    Args:
        constraints: The constraints to adjust (not mutated).
        triggers: Sequence of trigger objects with ``.id`` and ``.category``
            (e.g. ``TriggerCondition`` or ``TriggerSummary``).
        min_enabled: Override for
            ``JudgeConstraints.MIN_ENABLED_ENTRY_TRIGGERS``.

    Returns:
        A new ``JudgeConstraints`` with ``disabled_trigger_ids`` trimmed.
    """
    if min_enabled is None:
        min_enabled = JudgeConstraints.MIN_ENABLED_ENTRY_TRIGGERS

    entry_categories = JudgeConstraints.ENTRY_CATEGORIES
    entry_trigger_ids = {
        t.id for t in triggers
        if getattr(t, "category", None) in entry_categories
    }
    total_entry = len(entry_trigger_ids)

    disabled_ids = list(constraints.disabled_trigger_ids)
    disabled_entry_ids = [tid for tid in disabled_ids if tid in entry_trigger_ids]
    enabled_entry = total_entry - len(disabled_entry_ids)

    if enabled_entry < min_enabled:
        # Re-enable from the end (most recently disabled) until floor met
        while disabled_entry_ids and enabled_entry < min_enabled:
            to_reenable = disabled_entry_ids.pop()
            disabled_ids.remove(to_reenable)
            enabled_entry += 1

    if disabled_ids != list(constraints.disabled_trigger_ids):
        return constraints.model_copy(update={"disabled_trigger_ids": disabled_ids})
    return constraints

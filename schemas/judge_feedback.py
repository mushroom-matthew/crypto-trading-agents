"""Schemas for judge feedback and constraints."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, FrozenSet, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .llm_strategist import SerializableModel


# Attribution type definitions (Runbook 20: Judge Attribution Rubric)
AttributionLayer = Literal["plan", "trigger", "policy", "execution", "safety"]
AttributionConfidence = Literal["low", "medium", "high"]
RecommendedAction = Literal["hold", "policy_adjust", "replan", "investigate_execution", "stand_down"]


class AttributionEvidence(SerializableModel):
    """Evidence backing a Judge attribution decision."""

    metrics: List[str] = Field(
        default_factory=list,
        description="Metric names/values that informed the attribution",
    )
    trade_sets: List[str] = Field(
        default_factory=list,
        description="Trade set IDs examined",
    )
    events: List[str] = Field(
        default_factory=list,
        description="Event IDs (emergency exits, overrides, etc.) considered",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional context for the attribution decision",
    )


class JudgeAttribution(SerializableModel):
    """Attribution output from Judge evaluation.

    Every evaluation must produce exactly one primary attribution with
    evidence references. The recommended_action is gated by the attribution:
    - replan: only valid for 'plan' or 'trigger' attribution
    - policy_adjust: only valid for 'policy' attribution
    - stand_down: valid for any attribution
    - investigate_execution: typically for 'execution' attribution
    - hold: valid for any attribution (no change needed)
    """

    primary_attribution: AttributionLayer = Field(
        description="Single primary layer responsible for outcome",
    )
    secondary_factors: List[AttributionLayer] = Field(
        default_factory=list,
        description="Contributing factors (not primary cause)",
    )
    confidence: AttributionConfidence = Field(
        default="medium",
        description="Confidence in the attribution decision",
    )
    recommended_action: RecommendedAction = Field(
        default="hold",
        description="Recommended action based on attribution",
    )
    evidence: AttributionEvidence = Field(
        default_factory=AttributionEvidence,
        description="Evidence backing the attribution",
    )
    canonical_verdict: Optional[str] = Field(
        default=None,
        description="Human-readable verdict explaining the attribution",
    )

    # Action gating rules
    REPLAN_ALLOWED_LAYERS: ClassVar[FrozenSet[AttributionLayer]] = frozenset({"plan", "trigger"})
    POLICY_ADJUST_ALLOWED_LAYERS: ClassVar[FrozenSet[AttributionLayer]] = frozenset({"policy"})

    @model_validator(mode="after")
    def validate_action_gating(self) -> "JudgeAttribution":
        """Enforce action gating rules based on attribution layer."""
        action = self.recommended_action
        layer = self.primary_attribution

        if action == "replan" and layer not in self.REPLAN_ALLOWED_LAYERS:
            raise ValueError(
                f"replan action requires attribution to 'plan' or 'trigger', got '{layer}'"
            )

        if action == "policy_adjust" and layer not in self.POLICY_ADJUST_ALLOWED_LAYERS:
            raise ValueError(
                f"policy_adjust action requires attribution to 'policy', got '{layer}'"
            )

        return self

    @model_validator(mode="after")
    def validate_evidence_required(self) -> "JudgeAttribution":
        """Ensure evidence is provided (at least one field non-empty)."""
        ev = self.evidence
        if not ev.metrics and not ev.trade_sets and not ev.events and not ev.notes:
            raise ValueError("Attribution must include evidence (metrics, trade_sets, events, or notes)")
        return self


class DisplayConstraints(SerializableModel):
    """Human-readable guidance surfaced to the strategist."""

    must_fix: List[str] = Field(default_factory=list)
    vetoes: List[str] = Field(default_factory=list)
    boost: List[str] = Field(default_factory=list)
    regime_correction: Optional[str] = None
    sizing_adjustments: Dict[str, str] = Field(default_factory=dict)
    recommended_stance: Optional[Literal["active", "defensive", "wait"]] = None


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
    attribution: Optional[JudgeAttribution] = Field(
        default=None,
        description="Attribution analysis when evaluation determines outcome causes",
    )


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

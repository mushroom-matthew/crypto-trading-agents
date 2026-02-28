"""Schemas for judge feedback and constraints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, FrozenSet, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .llm_strategist import SerializableModel


# Attribution type definitions (Runbook 20: Judge Attribution Rubric)
AttributionLayer = Literal["plan", "trigger", "policy", "execution", "safety"]
AttributionConfidence = Literal["low", "medium", "high"]
RecommendedAction = Literal["hold", "policy_adjust", "replan", "investigate_execution", "stand_down"]
JudgeActionStatus = Literal["pending", "applied", "skipped", "expired"]
JudgeActionScope = Literal["intraday", "daily", "manual"]

# Extended action type including research-loop actions (Runbook 48)
JudgeActionType = Literal[
    "hold",
    "replan",
    "policy_adjust",
    "stand_down",
    "suggest_experiment",   # NEW: propose a research ExperimentSpec
    "update_playbook",      # NEW: surface a playbook edit suggestion for human review
]


class ExperimentSuggestion(SerializableModel):
    """Payload for a judge suggest_experiment action."""

    playbook_id: str
    hypothesis: str
    target_symbols: List[str]
    trigger_categories: List[str]
    min_sample_size: int = 20
    max_loss_usd: float = 50.0
    rationale: str  # Why the judge is suggesting this experiment


class PlaybookEditSuggestion(SerializableModel):
    """Payload for a judge update_playbook action."""

    playbook_id: str
    section: Literal["Notes", "Patterns", "Validation Evidence"]
    suggested_text: str
    evidence_summary: str  # What evidence supports this edit
    requires_human_review: bool = True  # Always True; human approves before applying


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


class JudgeAction(SerializableModel):
    """Structured judge action derived from feedback and enforced deterministically."""

    action_id: str
    source_eval_id: Optional[str] = Field(
        default=None,
        description="Identifier for the evaluation window (day key, plan id, etc.).",
    )
    recommended_action: RecommendedAction = Field(
        description="Action to take based on attribution and constraints.",
    )
    constraints: JudgeConstraints = Field(default_factory=JudgeConstraints)
    strategist_constraints: DisplayConstraints = Field(default_factory=DisplayConstraints)
    stance_override: Optional[Literal["active", "defensive", "wait"]] = None
    ttl_evals: int = Field(default=1, ge=1, description="Number of judge evaluations to retain this action.")
    evals_remaining: int = Field(default=1, ge=0, description="Remaining evaluations before expiry.")
    status: JudgeActionStatus = Field(default="pending")
    reason: Optional[str] = Field(default=None, description="Reason for applying/skipping the action.")
    scope: Optional[JudgeActionScope] = Field(default=None, description="Intraday or daily action scope.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    applied_at: Optional[datetime] = None
    # Research-loop action payloads (Runbook 48)
    experiment_suggestion: Optional[ExperimentSuggestion] = None
    playbook_edit_suggestion: Optional[PlaybookEditSuggestion] = None


# ---------------------------------------------------------------------------
# R53 — Judge Plan Validation Schemas (memory-backed, policy-boundary check)
# ---------------------------------------------------------------------------


#: Valid decisions produced by the plan-level judge validation gate.
JudgeValidationDecision = Literal["approve", "revise", "reject", "stand_down"]

#: Finding classes that distinguish the nature of a non-approve verdict.
JudgeValidationFindingClass = Literal[
    "structural_violation",    # hard reject — deterministic invariant or playbook contract
    "statistical_suspicion",   # soft revise — evidence weak/mixed, confidence too high
    "memory_contradiction",    # explain-or-revise — loser patterns contradict proposal
    "none",                    # approve path — no material finding
]

#: Confidence calibration result.
JudgeConfidenceCalibration = Literal["supported", "weakly_supported", "unsupported"]


class JudgeValidationVerdict(SerializableModel):
    """Typed output of the judge plan validation gate (Runbook 53).

    Produced once per policy-boundary evaluation.  Tick-level execution
    remains deterministic and is never routed through this object.

    Fields
    ------
    decision
        ``approve`` / ``revise`` / ``reject`` / ``stand_down``.
    finding_class
        Explicit classification of the primary finding.
    reasons
        Human-readable list of reasons for the decision.
    judge_confidence_score
        Judge's own confidence in the verdict (0.0–1.0).
    memory_evidence_refs
        Bundle IDs / snapshot hashes consulted.
    cited_episode_ids
        Specific episode IDs cited as evidence (win or loss).
    failure_pattern_matches
        Failure mode labels from FAILURE_MODE_TAXONOMY that match the plan.
    cluster_support_summary
        One-sentence summary of cluster win/loss support (or None).
    confidence_calibration
        Whether proposal confidence is supported by evidence.
    divergence_from_nearest_losers
        Required non-None when approving despite contradictory loser evidence.
    requested_revisions
        Ordered list of specific revisions requested (non-empty when revise).
    revision_count
        How many revision attempts have been made so far this policy event.
    """

    decision: JudgeValidationDecision
    finding_class: JudgeValidationFindingClass = "none"
    reasons: List[str] = Field(default_factory=list)
    judge_confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    memory_evidence_refs: List[str] = Field(default_factory=list)
    cited_episode_ids: List[str] = Field(default_factory=list)
    failure_pattern_matches: List[str] = Field(default_factory=list)
    cluster_support_summary: Optional[str] = None
    confidence_calibration: JudgeConfidenceCalibration = "weakly_supported"
    divergence_from_nearest_losers: Optional[str] = None
    requested_revisions: List[str] = Field(default_factory=list)
    revision_count: int = Field(default=0, ge=0)


class JudgePlanRevisionRequest(SerializableModel):
    """Structured revision request passed back to the strategist when verdict=revise.

    Contains the exact failing criteria, cited failure patterns, and expectation
    mismatches so the strategist can address them without re-running a full replan.
    """

    verdict: JudgeValidationVerdict
    revision_number: int = Field(ge=1)
    max_revisions: int = Field(ge=1)
    failing_criteria: List[str] = Field(default_factory=list)
    cited_failure_patterns: List[str] = Field(default_factory=list)
    expectation_mismatches: List[str] = Field(default_factory=list)
    revision_guidance: Optional[str] = None


class RevisionLoopResult(SerializableModel):
    """Result of a full revision loop run for one policy event.

    Includes the final plan (may be None if stand_down), the final verdict,
    and metadata about how many revisions were attempted.
    """

    final_verdict: JudgeValidationVerdict
    revision_attempts: int = Field(ge=0)
    revision_budget_exhausted: bool = False
    stand_down_reason: Optional[str] = None
    # plan_id of the final plan accepted (or None if stand_down)
    accepted_plan_id: Optional[str] = None


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

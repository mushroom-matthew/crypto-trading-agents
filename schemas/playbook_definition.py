"""Typed playbook definitions with regime tags (Runbook 52).

PlaybookDefinition is a structured representation of the information encoded
in vector_store/playbooks/*.md files, enriched with regime-eligibility rules,
entry/risk/invalidation rule sets, horizon expectations, historical performance
stats, and policy-stability constraints.

These types are intentionally decoupled from any LLM or workflow runtime:
they are pure data containers that can be loaded from .md frontmatter,
serialized to JSON, and attached as structured context to LLM prompts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Literals / enums
# ---------------------------------------------------------------------------

PolicyClass = Literal[
    "trend_following",
    "mean_reversion",
    "breakout",
    "volatility_expansion",
    "volatility_compression",
]

ActivationRefinementMode = Literal[
    "price_touch",
    "close_confirmed",
    "liquidity_sweep",
    "next_bar_open",
]

ThesisState = Literal[
    "thesis_armed",
    "position_open",
    "hold_lock",
    "invalidated",
    "cooldown",
    "waiting",
]

ActivationExpiredReason = Literal[
    "timeout",
    "structure_break",
    "shock",
    "regime_cancel",
    "safety_cancel",
]

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class RegimeEligibility(BaseModel):
    model_config = {"extra": "forbid"}

    eligible_regimes: List[str] = Field(default_factory=list)
    disallowed_regimes: List[str] = Field(default_factory=list)
    min_confidence: Optional[float] = None  # 0-1


class EntryRuleSet(BaseModel):
    model_config = {"extra": "forbid"}

    thesis_conditions: List[str] = Field(default_factory=list)
    activation_triggers: List[str] = Field(default_factory=list)
    activation_timeout_bars: Optional[int] = None  # execution-TF bars
    activation_refinement_mode: ActivationRefinementMode = "price_touch"
    # Telemetry fields — populated at runtime
    activation_expired_reason: Optional[ActivationExpiredReason] = None
    armed_duration_bars: Optional[int] = None  # runtime only


class RiskRuleSet(BaseModel):
    model_config = {"extra": "forbid"}

    stop_methods: List[str] = Field(default_factory=list)
    target_methods: List[str] = Field(default_factory=list)
    minimum_structural_r_multiple: Optional[float] = None  # expectancy gate
    require_structural_target: bool = False
    structural_target_sources: List[str] = Field(default_factory=list)
    fallback_behavior: Optional[str] = None  # what to do if no anchor found


class InvalidationRuleSet(BaseModel):
    model_config = {"extra": "forbid"}

    pre_entry_conditions: List[str] = Field(default_factory=list)   # cancel thesis_armed
    post_entry_conditions: List[str] = Field(default_factory=list)  # risk-off after entry


class HorizonExpectations(BaseModel):
    model_config = {"extra": "forbid"}

    expected_hold_bars_p50: Optional[int] = None
    expected_hold_bars_p90: Optional[int] = None
    setup_maturation_bars: Optional[int] = None
    expiry_bars: Optional[int] = None  # max hold / TTL


class PlaybookPerformanceStats(BaseModel):
    model_config = {"extra": "forbid"}

    n: int = 0
    win_rate: Optional[float] = None        # 0-1
    avg_r: Optional[float] = None
    expectancy: Optional[float] = None      # win_rate * avg_r - (1-win_rate) * avg_loss_r
    p50_hold_bars: Optional[int] = None
    p90_hold_bars: Optional[int] = None
    hold_bars_mean: Optional[float] = None
    hold_bars_std: Optional[float] = None
    hold_time_z_score: Optional[float] = None  # calibration check
    mae_p50: Optional[float] = None
    mfe_p50: Optional[float] = None
    last_updated: Optional[datetime] = None
    evidence_source: Optional[str] = None  # "signal_ledger" | "manual" | "research_budget" | "episode_memory"
    regime: Optional[str] = None           # regime this stats slice applies to


class PolicyStabilityConstraints(BaseModel):
    model_config = {"extra": "forbid"}

    min_hold_bars: Optional[int] = None              # before policy re-evaluation allowed
    policy_mutation_cooldown_bars: Optional[int] = None
    allowed_mutations: List[str] = Field(default_factory=list)    # e.g. ["stop_tighten_only"]
    forbidden_mutations: List[str] = Field(default_factory=list)  # e.g. ["switch_playbook_family"]
    cross_policy_class_mutation_allowed: bool = False


class RefinementModeMapping(BaseModel):
    """Maps activation_refinement_mode to deterministic trigger identifier + timeframe."""

    model_config = {"extra": "forbid"}

    mode: ActivationRefinementMode
    trigger_identifiers: List[str]
    evaluation_timeframe: Optional[str] = None  # e.g. "execution_tf", "micro_tf"
    confirmation_rule: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level PlaybookDefinition
# ---------------------------------------------------------------------------


class PlaybookDefinition(BaseModel):
    model_config = {"extra": "forbid"}

    # Identity
    playbook_id: str
    version: str = "1.0.0"
    template_id: Optional[str] = None  # links to vector_store/strategies/*.md
    policy_class: Optional[PolicyClass] = None

    # Regime
    regime_eligibility: RegimeEligibility = Field(default_factory=RegimeEligibility)

    # Entry
    entry_rules: EntryRuleSet = Field(default_factory=EntryRuleSet)

    # Risk
    risk_rules: RiskRuleSet = Field(default_factory=RiskRuleSet)

    # Invalidation
    invalidation_rules: InvalidationRuleSet = Field(default_factory=InvalidationRuleSet)

    # Horizon
    horizon_expectations: HorizonExpectations = Field(default_factory=HorizonExpectations)

    # Historical stats (by regime; one entry per regime slice)
    performance_stats: List[PlaybookPerformanceStats] = Field(default_factory=list)

    # Policy stability
    stability_constraints: PolicyStabilityConstraints = Field(default_factory=PolicyStabilityConstraints)

    # Refinement mode mappings
    refinement_mode_mappings: List[RefinementModeMapping] = Field(default_factory=list)

    # Supplementary
    description: Optional[str] = None
    identifiers: List[str] = Field(default_factory=list)  # indicator identifiers this playbook uses
    tags: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Default refinement mode mappings (canonical reference)
# ---------------------------------------------------------------------------

REFINEMENT_MODE_DEFAULTS: Dict[str, RefinementModeMapping] = {
    "price_touch": RefinementModeMapping(
        mode="price_touch",
        trigger_identifiers=["break_level_touch"],
        evaluation_timeframe="execution_tf",
        confirmation_rule="touch_or_cross",
    ),
    "close_confirmed": RefinementModeMapping(
        mode="close_confirmed",
        trigger_identifiers=["break_level_close_confirmed"],
        evaluation_timeframe="micro_tf",
        confirmation_rule="close_beyond_level",
    ),
    "liquidity_sweep": RefinementModeMapping(
        mode="liquidity_sweep",
        trigger_identifiers=["sweep_low_reclaim", "sweep_high_reject"],
        evaluation_timeframe="micro_tf",
        confirmation_rule="sweep_plus_reclaim",
    ),
    "next_bar_open": RefinementModeMapping(
        mode="next_bar_open",
        trigger_identifiers=["next_bar_open_entry"],
        evaluation_timeframe="execution_tf",
        confirmation_rule="none",
    ),
}

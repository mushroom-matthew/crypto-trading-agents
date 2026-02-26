"""Tests for schemas/playbook_definition.py (Runbook 52).

Validates that all Pydantic models enforce their contracts:
- Required fields present
- Literal constraints enforced
- Default values correct
- extra="forbid" active on all models
- REFINEMENT_MODE_DEFAULTS completeness
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.playbook_definition import (
    REFINEMENT_MODE_DEFAULTS,
    ActivationExpiredReason,
    ActivationRefinementMode,
    EntryRuleSet,
    HorizonExpectations,
    InvalidationRuleSet,
    PlaybookDefinition,
    PlaybookPerformanceStats,
    PolicyClass,
    PolicyStabilityConstraints,
    RegimeEligibility,
    RefinementModeMapping,
    RiskRuleSet,
    ThesisState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_playbook(**overrides) -> PlaybookDefinition:
    defaults = dict(playbook_id="test_playbook")
    defaults.update(overrides)
    return PlaybookDefinition(**defaults)


# ---------------------------------------------------------------------------
# PlaybookDefinition — identity and required fields
# ---------------------------------------------------------------------------


class TestPlaybookDefinitionIdentity:
    def test_requires_playbook_id(self):
        """PlaybookDefinition must fail if playbook_id is missing."""
        with pytest.raises(ValidationError):
            PlaybookDefinition()  # no playbook_id

    def test_minimal_with_playbook_id_only(self):
        """PlaybookDefinition can be created with just playbook_id."""
        pb = PlaybookDefinition(playbook_id="my_playbook")
        assert pb.playbook_id == "my_playbook"

    def test_default_version(self):
        pb = _make_playbook()
        assert pb.version == "1.0.0"

    def test_template_id_defaults_none(self):
        pb = _make_playbook()
        assert pb.template_id is None

    def test_policy_class_defaults_none(self):
        pb = _make_playbook()
        assert pb.policy_class is None

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            PlaybookDefinition(playbook_id="x", unknown_field="oops")

    def test_description_defaults_none(self):
        pb = _make_playbook()
        assert pb.description is None

    def test_identifiers_defaults_empty_list(self):
        pb = _make_playbook()
        assert pb.identifiers == []

    def test_tags_defaults_empty_list(self):
        pb = _make_playbook()
        assert pb.tags == []

    def test_performance_stats_defaults_empty(self):
        pb = _make_playbook()
        assert pb.performance_stats == []

    def test_refinement_mode_mappings_defaults_empty(self):
        pb = _make_playbook()
        assert pb.refinement_mode_mappings == []


# ---------------------------------------------------------------------------
# RegimeEligibility
# ---------------------------------------------------------------------------


class TestRegimeEligibility:
    def test_eligible_regimes_defaults_empty(self):
        re = RegimeEligibility()
        assert re.eligible_regimes == []

    def test_disallowed_regimes_defaults_empty(self):
        re = RegimeEligibility()
        assert re.disallowed_regimes == []

    def test_min_confidence_defaults_none(self):
        re = RegimeEligibility()
        assert re.min_confidence is None

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            RegimeEligibility(unknown="x")

    def test_eligible_regimes_populated(self):
        re = RegimeEligibility(eligible_regimes=["range", "volatile"])
        assert "range" in re.eligible_regimes
        assert "volatile" in re.eligible_regimes


# ---------------------------------------------------------------------------
# EntryRuleSet
# ---------------------------------------------------------------------------


class TestEntryRuleSet:
    def test_activation_refinement_mode_defaults_price_touch(self):
        ers = EntryRuleSet()
        assert ers.activation_refinement_mode == "price_touch"

    def test_thesis_conditions_defaults_empty(self):
        ers = EntryRuleSet()
        assert ers.thesis_conditions == []

    def test_activation_triggers_defaults_empty(self):
        ers = EntryRuleSet()
        assert ers.activation_triggers == []

    def test_activation_timeout_bars_defaults_none(self):
        ers = EntryRuleSet()
        assert ers.activation_timeout_bars is None

    def test_activation_expired_reason_defaults_none(self):
        ers = EntryRuleSet()
        assert ers.activation_expired_reason is None

    def test_armed_duration_bars_defaults_none(self):
        ers = EntryRuleSet()
        assert ers.armed_duration_bars is None

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            EntryRuleSet(bad_field=True)


# ---------------------------------------------------------------------------
# RiskRuleSet
# ---------------------------------------------------------------------------


class TestRiskRuleSet:
    def test_require_structural_target_defaults_false(self):
        rrs = RiskRuleSet()
        assert rrs.require_structural_target is False

    def test_stop_methods_defaults_empty(self):
        rrs = RiskRuleSet()
        assert rrs.stop_methods == []

    def test_target_methods_defaults_empty(self):
        rrs = RiskRuleSet()
        assert rrs.target_methods == []

    def test_minimum_structural_r_multiple_defaults_none(self):
        rrs = RiskRuleSet()
        assert rrs.minimum_structural_r_multiple is None

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            RiskRuleSet(bad_key=1)


# ---------------------------------------------------------------------------
# PolicyStabilityConstraints
# ---------------------------------------------------------------------------


class TestPolicyStabilityConstraints:
    def test_cross_policy_class_mutation_allowed_defaults_false(self):
        psc = PolicyStabilityConstraints()
        assert psc.cross_policy_class_mutation_allowed is False

    def test_allowed_mutations_defaults_empty(self):
        psc = PolicyStabilityConstraints()
        assert psc.allowed_mutations == []

    def test_forbidden_mutations_defaults_empty(self):
        psc = PolicyStabilityConstraints()
        assert psc.forbidden_mutations == []

    def test_min_hold_bars_defaults_none(self):
        psc = PolicyStabilityConstraints()
        assert psc.min_hold_bars is None

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            PolicyStabilityConstraints(bad=True)


# ---------------------------------------------------------------------------
# PlaybookPerformanceStats
# ---------------------------------------------------------------------------


class TestPlaybookPerformanceStats:
    def test_n_defaults_zero(self):
        stats = PlaybookPerformanceStats()
        assert stats.n == 0

    def test_win_rate_defaults_none(self):
        stats = PlaybookPerformanceStats()
        assert stats.win_rate is None

    def test_avg_r_defaults_none(self):
        stats = PlaybookPerformanceStats()
        assert stats.avg_r is None

    def test_evidence_source_defaults_none(self):
        stats = PlaybookPerformanceStats()
        assert stats.evidence_source is None

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            PlaybookPerformanceStats(n=0, unexpected="x")


# ---------------------------------------------------------------------------
# HorizonExpectations
# ---------------------------------------------------------------------------


class TestHorizonExpectations:
    def test_all_fields_default_none(self):
        he = HorizonExpectations()
        assert he.expected_hold_bars_p50 is None
        assert he.expected_hold_bars_p90 is None
        assert he.setup_maturation_bars is None
        assert he.expiry_bars is None

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            HorizonExpectations(bad=1)


# ---------------------------------------------------------------------------
# REFINEMENT_MODE_DEFAULTS
# ---------------------------------------------------------------------------


class TestRefinementModeDefaults:
    def test_has_all_four_modes(self):
        assert "price_touch" in REFINEMENT_MODE_DEFAULTS
        assert "close_confirmed" in REFINEMENT_MODE_DEFAULTS
        assert "liquidity_sweep" in REFINEMENT_MODE_DEFAULTS
        assert "next_bar_open" in REFINEMENT_MODE_DEFAULTS

    def test_each_mode_has_non_empty_trigger_identifiers(self):
        for mode_key, mapping in REFINEMENT_MODE_DEFAULTS.items():
            assert len(mapping.trigger_identifiers) > 0, (
                f"Mode '{mode_key}' has empty trigger_identifiers"
            )

    def test_price_touch_trigger_identifier(self):
        m = REFINEMENT_MODE_DEFAULTS["price_touch"]
        assert "break_level_touch" in m.trigger_identifiers

    def test_liquidity_sweep_has_two_identifiers(self):
        m = REFINEMENT_MODE_DEFAULTS["liquidity_sweep"]
        assert len(m.trigger_identifiers) == 2

    def test_refinement_mode_mapping_extra_rejected(self):
        with pytest.raises(ValidationError):
            RefinementModeMapping(
                mode="price_touch",
                trigger_identifiers=["x"],
                bad_field="oops",
            )


# ---------------------------------------------------------------------------
# Literal type checks
# ---------------------------------------------------------------------------


class TestLiterals:
    def test_thesis_state_valid_values(self):
        valid: list[ThesisState] = [
            "thesis_armed", "position_open", "hold_lock",
            "invalidated", "cooldown", "waiting",
        ]
        for v in valid:
            # Ensure value is in the Literal args (structural check)
            assert v in ThesisState.__args__  # type: ignore[attr-defined]

    def test_activation_expired_reason_valid_values(self):
        valid: list[ActivationExpiredReason] = [
            "timeout", "structure_break", "shock", "regime_cancel", "safety_cancel",
        ]
        for v in valid:
            assert v in ActivationExpiredReason.__args__  # type: ignore[attr-defined]

    def test_activation_refinement_mode_valid_values(self):
        valid: list[ActivationRefinementMode] = [
            "price_touch", "close_confirmed", "liquidity_sweep", "next_bar_open",
        ]
        for v in valid:
            assert v in ActivationRefinementMode.__args__  # type: ignore[attr-defined]

    def test_invalid_activation_refinement_mode_rejected(self):
        with pytest.raises(ValidationError):
            EntryRuleSet(activation_refinement_mode="market_order")  # not in Literal

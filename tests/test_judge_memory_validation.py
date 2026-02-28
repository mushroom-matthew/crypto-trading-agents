"""Tests for JudgePlanValidationService (Runbook 53) — memory-backed validation gate.

Covers:
- Deterministic hard-reject paths (THESIS_ARMED, HOLD_LOCK, cooldown, empty plan)
- Playbook consistency: regime eligibility, missing invalidation condition
- Memory failure-pattern scan
- Historical cluster support and confidence calibration
- Divergence explanation requirement
- Policy-boundary keyword labelling (STRUCTURAL / REVISE / MEMORY prefixes)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from schemas.episode_memory import (
    DiversifiedMemoryBundle,
    EpisodeMemoryRecord,
    MemoryRetrievalMeta,
)
from schemas.llm_strategist import StrategyPlan, TriggerCondition
from services.judge_validation_service import JudgePlanValidationService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_NOW = datetime.now(timezone.utc)


def make_trigger(
    direction: str = "long",
    category: str = "trend_continuation",
    trigger_id: str = "t1",
) -> TriggerCondition:
    return TriggerCondition(
        id=trigger_id,
        symbol="BTC-USD",
        direction=direction,
        category=category,
        timeframe="1h",
        entry_rule="price > vwap",
        exit_rule="price < vwap",
        stop_loss_pct=1.5,
    )


def make_exit_trigger(trigger_id: str = "t-exit") -> TriggerCondition:
    return TriggerCondition(
        id=trigger_id,
        symbol="BTC-USD",
        direction="exit",
        category="trend_continuation",
        timeframe="1h",
        entry_rule="invalidation_condition",
        exit_rule="close < structure_low",
    )


def make_plan(
    triggers: list[TriggerCondition] | None = None,
    regime: str = "bull",
    allowed_directions: list[str] | None = None,
) -> StrategyPlan:
    if triggers is None:
        triggers = [make_trigger()]
    return StrategyPlan(
        regime=regime,
        generated_at=_NOW,
        valid_until=_NOW + timedelta(hours=4),
        triggers=triggers,
        allowed_directions=allowed_directions or ["long"],
    )


def make_episode(
    outcome_class: str = "loss",
    failure_modes: list[str] | None = None,
    symbol: str = "BTC-USD",
) -> EpisodeMemoryRecord:
    return EpisodeMemoryRecord(
        episode_id=f"ep-{id(failure_modes)}",
        symbol=symbol,
        outcome_class=outcome_class,
        failure_modes=failure_modes or [],
    )


def make_bundle(
    wins: int = 3,
    losses: int = 1,
    failure_mode_episodes: list[EpisodeMemoryRecord] | None = None,
) -> DiversifiedMemoryBundle:
    win_eps = [make_episode(outcome_class="win") for _ in range(wins)]
    loss_eps = [make_episode(outcome_class="loss") for _ in range(losses)]
    return DiversifiedMemoryBundle(
        bundle_id="bundle-001",
        symbol="BTC-USD",
        created_at=_NOW,
        winning_contexts=win_eps,
        losing_contexts=loss_eps,
        failure_mode_patterns=failure_mode_episodes or [],
        retrieval_meta=MemoryRetrievalMeta(),
    )


_SVC = JudgePlanValidationService()


# ---------------------------------------------------------------------------
# Step 1: Deterministic constraints
# ---------------------------------------------------------------------------


class TestDeterministicHardReject:
    def test_thesis_armed_no_invalidation_rejects(self):
        """THESIS_ARMED without invalidation/safety override → hard reject."""
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            is_thesis_armed=True,
            has_invalidation_trigger=False,
            has_safety_override=False,
        )
        assert verdict.decision == "reject"
        assert verdict.finding_class == "structural_violation"
        assert any("THESIS_ARMED" in r for r in verdict.reasons)

    def test_thesis_armed_with_invalidation_allows_through(self):
        """THESIS_ARMED + invalidation trigger → not blocked by armed check."""
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            is_thesis_armed=True,
            has_invalidation_trigger=True,
        )
        # May approve or revise, but must NOT be a structural reject for armed
        structural = [r for r in verdict.reasons if "THESIS_ARMED" in r and r.startswith("STRUCTURAL:")]
        assert len(structural) == 0

    def test_thesis_armed_with_safety_override_allows_through(self):
        """THESIS_ARMED + safety override → not blocked by armed check."""
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            is_thesis_armed=True,
            has_safety_override=True,
        )
        structural_armed = [r for r in verdict.reasons if "THESIS_ARMED" in r and r.startswith("STRUCTURAL:")]
        assert len(structural_armed) == 0

    def test_hold_lock_no_override_rejects(self):
        """HOLD_LOCK without safety override → hard reject."""
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            is_hold_lock=True,
            has_safety_override=False,
        )
        assert verdict.decision == "reject"
        assert verdict.finding_class == "structural_violation"
        assert any("HOLD_LOCK" in r for r in verdict.reasons)

    def test_hold_lock_with_safety_override_passes(self):
        """HOLD_LOCK + safety override → not blocked by lock check."""
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            is_hold_lock=True,
            has_safety_override=True,
        )
        lock_structural = [r for r in verdict.reasons if "HOLD_LOCK" in r and r.startswith("STRUCTURAL:")]
        assert len(lock_structural) == 0

    def test_policy_cooldown_playbook_switch_no_exception_rejects(self):
        """Playbook switch inside cooldown without invalidation/safety → hard reject."""
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            policy_cooldown_active=True,
            is_playbook_switch=True,
            has_invalidation_trigger=False,
            has_safety_override=False,
        )
        assert verdict.decision == "reject"
        assert verdict.finding_class == "structural_violation"
        assert any("cooldown" in r.lower() for r in verdict.reasons)

    def test_policy_cooldown_no_switch_allowed(self):
        """Cooldown active but NO playbook switch → should not block."""
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            policy_cooldown_active=True,
            is_playbook_switch=False,
        )
        # Cooldown with no switch should not produce a structural block
        cooldown_structural = [r for r in verdict.reasons if "cooldown" in r.lower() and r.startswith("STRUCTURAL:")]
        assert len(cooldown_structural) == 0

    def test_empty_plan_triggers_hard_reject(self):
        """Plan with no triggers → hard reject."""
        plan = make_plan(triggers=[])
        verdict = _SVC.validate_plan(plan)
        assert verdict.decision == "reject"
        assert verdict.finding_class == "structural_violation"
        assert any("no triggers" in r.lower() for r in verdict.reasons)

    def test_hard_reject_returns_high_confidence(self):
        """Hard rejects carry high judge confidence (0.90+)."""
        plan = make_plan(triggers=[])
        verdict = _SVC.validate_plan(plan)
        assert verdict.judge_confidence_score >= 0.90


# ---------------------------------------------------------------------------
# Step 2: Playbook consistency
# ---------------------------------------------------------------------------


class TestPlaybookConsistency:
    def test_ineligible_regime_rejects(self):
        """Plan regime not in playbook eligible regimes → structural reject."""
        plan = make_plan(regime="bear")
        verdict = _SVC.validate_plan(
            plan,
            playbook_regime_tags=["bull", "range"],
        )
        assert verdict.decision == "reject"
        assert verdict.finding_class == "structural_violation"
        assert any("regime" in r.lower() for r in verdict.reasons)

    def test_eligible_regime_passes_consistency(self):
        """Plan regime in playbook eligible regimes → no regime block."""
        plan = make_plan(regime="bull")
        verdict = _SVC.validate_plan(
            plan,
            playbook_regime_tags=["bull", "range"],
        )
        regime_structural = [r for r in verdict.reasons if "regime" in r.lower() and r.startswith("STRUCTURAL:")]
        assert len(regime_structural) == 0

    def test_missing_exit_trigger_when_required_raises_revise(self):
        """Playbook requires invalidation exit but plan has none → REVISE finding."""
        plan = make_plan(triggers=[make_trigger()])
        verdict = _SVC.validate_plan(
            plan,
            playbook_requires_invalidation=True,
        )
        revise_reasons = [r for r in verdict.reasons if r.startswith("REVISE:")]
        assert any("exit" in r.lower() or "invalidation" in r.lower() for r in revise_reasons)

    def test_exit_trigger_present_satisfies_invalidation_requirement(self):
        """Playbook requires invalidation and plan has exit trigger → no REVISE for invalidation."""
        plan = make_plan(triggers=[make_trigger(), make_exit_trigger()])
        verdict = _SVC.validate_plan(
            plan,
            playbook_requires_invalidation=True,
        )
        exit_revise = [
            r for r in verdict.reasons
            if r.startswith("REVISE:") and ("exit" in r.lower() or "invalidation" in r.lower())
        ]
        assert len(exit_revise) == 0

    def test_no_playbook_tags_skips_regime_check(self):
        """No playbook_regime_tags → no regime eligibility block."""
        plan = make_plan(regime="bear")
        verdict = _SVC.validate_plan(plan, playbook_regime_tags=None)
        regime_structural = [r for r in verdict.reasons if "regime" in r.lower() and r.startswith("STRUCTURAL:")]
        assert len(regime_structural) == 0


# ---------------------------------------------------------------------------
# Step 3: Memory failure-pattern scan
# ---------------------------------------------------------------------------


class TestMemoryFailurePatternScan:
    def test_strong_failure_mode_triggers_revise(self):
        """A strong known failure mode recurring 2+ times → revise verdict."""
        # Create episodes with false_breakout_reversion appearing 3 times
        ep1 = make_episode(outcome_class="loss", failure_modes=["false_breakout_reversion"])
        ep2 = make_episode(outcome_class="loss", failure_modes=["false_breakout_reversion"])
        ep3 = make_episode(outcome_class="loss", failure_modes=["false_breakout_reversion"])
        bundle = DiversifiedMemoryBundle(
            bundle_id="b-strong",
            symbol="BTC-USD",
            created_at=_NOW,
            winning_contexts=[],
            losing_contexts=[ep1, ep2, ep3],
            failure_mode_patterns=[],
            retrieval_meta=MemoryRetrievalMeta(),
        )
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        # Should produce at least a revise verdict
        assert verdict.decision in {"revise", "reject"}
        assert any("false_breakout_reversion" in r for r in verdict.reasons)

    def test_failure_mode_with_single_occurrence_not_flagged(self):
        """Single occurrence of failure mode → not enough to flag (requires ≥2)."""
        ep1 = make_episode(outcome_class="loss", failure_modes=["false_breakout_reversion"])
        bundle = DiversifiedMemoryBundle(
            bundle_id="b-single",
            symbol="BTC-USD",
            created_at=_NOW,
            winning_contexts=[],
            losing_contexts=[ep1],
            failure_mode_patterns=[],
            retrieval_meta=MemoryRetrievalMeta(),
        )
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        # Should not flag the failure mode
        mem_reasons = [r for r in verdict.reasons if r.startswith("MEMORY:")]
        assert len(mem_reasons) == 0

    def test_memory_reasons_have_memory_prefix(self):
        """Memory-derived findings must have 'MEMORY:' prefix."""
        ep1 = make_episode(outcome_class="loss", failure_modes=["macro_news_whipsaw"])
        ep2 = make_episode(outcome_class="loss", failure_modes=["macro_news_whipsaw"])
        bundle = DiversifiedMemoryBundle(
            bundle_id="b-prefix",
            symbol="BTC-USD",
            created_at=_NOW,
            winning_contexts=[],
            losing_contexts=[ep1, ep2],
            failure_mode_patterns=[],
            retrieval_meta=MemoryRetrievalMeta(),
        )
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        for r in verdict.reasons:
            assert r.startswith("STRUCTURAL:") or r.startswith("REVISE:") or r.startswith("MEMORY:")

    def test_no_memory_bundle_does_not_block(self):
        """No memory bundle → memory checks skipped, can still approve."""
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=None)
        mem_reasons = [r for r in verdict.reasons if r.startswith("MEMORY:")]
        assert len(mem_reasons) == 0

    def test_failure_pattern_matches_propagated_to_verdict(self):
        """Matched failure patterns stored in failure_pattern_matches field."""
        ep1 = make_episode(outcome_class="loss", failure_modes=["false_breakout_reversion"])
        ep2 = make_episode(outcome_class="loss", failure_modes=["false_breakout_reversion"])
        bundle = DiversifiedMemoryBundle(
            bundle_id="b-matches",
            symbol="BTC-USD",
            created_at=_NOW,
            winning_contexts=[],
            losing_contexts=[ep1, ep2],
            failure_mode_patterns=[],
            retrieval_meta=MemoryRetrievalMeta(),
        )
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        assert "false_breakout_reversion" in verdict.failure_pattern_matches


# ---------------------------------------------------------------------------
# Step 4: Cluster support
# ---------------------------------------------------------------------------


class TestClusterSupport:
    def test_strong_cluster_produces_supported_calibration(self):
        """≥60% win rate cluster → confidence_calibration=supported."""
        bundle = make_bundle(wins=4, losses=1)
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        assert verdict.confidence_calibration == "supported"

    def test_weak_cluster_produces_unsupported_calibration(self):
        """<40% win rate cluster → confidence_calibration=unsupported."""
        bundle = make_bundle(wins=1, losses=4)
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        assert verdict.confidence_calibration == "unsupported"

    def test_mixed_cluster_produces_weakly_supported(self):
        """40-60% win rate → confidence_calibration=weakly_supported."""
        bundle = make_bundle(wins=2, losses=3)
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        assert verdict.confidence_calibration == "weakly_supported"

    def test_small_cluster_produces_weakly_supported(self):
        """Fewer than 3 total episodes → evidence inconclusive / weakly_supported."""
        bundle = make_bundle(wins=1, losses=1)
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        assert verdict.confidence_calibration == "weakly_supported"

    def test_cluster_summary_present_when_bundle_provided(self):
        """cluster_support_summary field is populated when bundle is present."""
        bundle = make_bundle(wins=3, losses=1)
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        assert verdict.cluster_support_summary is not None
        assert len(verdict.cluster_support_summary) > 0

    def test_episode_ids_cited_from_losers(self):
        """cited_episode_ids populated from loser episodes."""
        loss_eps = [
            make_episode(outcome_class="loss") for _ in range(3)
        ]
        bundle = DiversifiedMemoryBundle(
            bundle_id="b-cite",
            symbol="BTC-USD",
            created_at=_NOW,
            winning_contexts=[make_episode(outcome_class="win")],
            losing_contexts=loss_eps,
            failure_mode_patterns=[],
            retrieval_meta=MemoryRetrievalMeta(),
        )
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        assert len(verdict.cited_episode_ids) > 0


# ---------------------------------------------------------------------------
# Step 5: Confidence calibration
# ---------------------------------------------------------------------------


class TestConfidenceCalibration:
    def test_high_conviction_unsupported_cluster_triggers_revise(self):
        """high conviction + unsupported cluster → REVISE finding."""
        bundle = make_bundle(wins=0, losses=4)
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            memory_bundle=bundle,
            stated_conviction="high",
        )
        revise_cal = [r for r in verdict.reasons if r.startswith("REVISE:") and "conviction" in r.lower()]
        assert len(revise_cal) >= 1

    def test_high_conviction_weakly_supported_with_failure_modes_triggers_revise(self):
        """high conviction + weakly_supported + ≥2 failure modes → REVISE."""
        ep1 = make_episode(outcome_class="loss", failure_modes=["false_breakout_reversion"])
        ep2 = make_episode(outcome_class="loss", failure_modes=["false_breakout_reversion"])
        ep3 = make_episode(outcome_class="win")
        ep4 = make_episode(outcome_class="win")
        # 2W/2L = 50% = weakly_supported
        bundle = DiversifiedMemoryBundle(
            bundle_id="b-calib",
            symbol="BTC-USD",
            created_at=_NOW,
            winning_contexts=[ep3, ep4],
            losing_contexts=[ep1, ep2],
            failure_mode_patterns=[],
            retrieval_meta=MemoryRetrievalMeta(),
        )
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            memory_bundle=bundle,
            stated_conviction="high",
        )
        assert verdict.decision in {"revise", "reject"}

    def test_low_conviction_unsupported_cluster_does_not_add_calibration_reason(self):
        """low conviction + unsupported cluster → calibration does not trigger REVISE."""
        bundle = make_bundle(wins=0, losses=4)
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            memory_bundle=bundle,
            stated_conviction="low",
        )
        conviction_revise = [
            r for r in verdict.reasons
            if r.startswith("REVISE:") and "conviction" in r.lower()
        ]
        assert len(conviction_revise) == 0

    def test_no_stated_conviction_skips_calibration_check(self):
        """None stated_conviction → calibration check skipped."""
        bundle = make_bundle(wins=0, losses=4)
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            memory_bundle=bundle,
            stated_conviction=None,
        )
        conviction_revise = [
            r for r in verdict.reasons
            if r.startswith("REVISE:") and "conviction" in r.lower()
        ]
        assert len(conviction_revise) == 0


# ---------------------------------------------------------------------------
# Approval with divergence explanation
# ---------------------------------------------------------------------------


class TestApproveWithDivergence:
    def test_approve_despite_failure_modes_generates_divergence(self):
        """Approving with matched failure modes must include divergence_from_nearest_losers."""
        # 4W / 1L = strong cluster (approve), but has 2 failure mode occurrences
        ep_loss1 = make_episode(outcome_class="loss", failure_modes=["low_volume_breakout_failure"])
        ep_loss2 = make_episode(outcome_class="loss", failure_modes=["low_volume_breakout_failure"])
        bundle = DiversifiedMemoryBundle(
            bundle_id="b-diverge",
            symbol="BTC-USD",
            created_at=_NOW,
            winning_contexts=[
                make_episode(outcome_class="win"),
                make_episode(outcome_class="win"),
                make_episode(outcome_class="win"),
                make_episode(outcome_class="win"),
            ],
            losing_contexts=[ep_loss1, ep_loss2],
            failure_mode_patterns=[],
            retrieval_meta=MemoryRetrievalMeta(),
        )
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        if verdict.decision == "approve":
            assert verdict.divergence_from_nearest_losers is not None
            assert len(verdict.divergence_from_nearest_losers) > 0

    def test_approve_clean_no_divergence_required(self):
        """Approving with no failure modes → divergence_from_nearest_losers is None."""
        bundle = make_bundle(wins=4, losses=1)
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        if verdict.decision == "approve":
            assert verdict.divergence_from_nearest_losers is None


# ---------------------------------------------------------------------------
# Memory evidence refs
# ---------------------------------------------------------------------------


class TestMemoryEvidenceRefs:
    def test_bundle_id_propagated_to_memory_evidence_refs(self):
        """When a bundle is provided, its bundle_id appears in memory_evidence_refs."""
        bundle = make_bundle(wins=3, losses=1)
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=bundle)
        assert bundle.bundle_id in verdict.memory_evidence_refs

    def test_no_bundle_memory_evidence_empty(self):
        """No bundle → memory_evidence_refs empty."""
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=None)
        assert verdict.memory_evidence_refs == []


# ---------------------------------------------------------------------------
# Revision count threading
# ---------------------------------------------------------------------------


class TestRevisionCountThreading:
    def test_revision_count_propagated(self):
        """revision_count passed in is reflected in verdict."""
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, revision_count=2)
        assert verdict.revision_count == 2

    def test_revision_count_default_zero(self):
        """Default revision_count is 0."""
        plan = make_plan()
        verdict = _SVC.validate_plan(plan)
        assert verdict.revision_count == 0


# ---------------------------------------------------------------------------
# Full approve path
# ---------------------------------------------------------------------------


class TestFullApprovePath:
    def test_clean_plan_approves(self):
        """Plan with valid triggers, supported cluster, no failure modes → approve."""
        bundle = make_bundle(wins=4, losses=1)
        plan = make_plan()
        verdict = _SVC.validate_plan(
            plan,
            memory_bundle=bundle,
            playbook_regime_tags=["bull"],
            stated_conviction="medium",
        )
        assert verdict.decision == "approve"
        assert verdict.finding_class == "none"
        assert verdict.judge_confidence_score > 0.5

    def test_approve_no_memory_bundle(self):
        """Plan with no memory bundle → may still approve (weakly_supported)."""
        plan = make_plan()
        verdict = _SVC.validate_plan(plan, memory_bundle=None)
        assert verdict.decision == "approve"

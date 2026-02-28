"""Tests for policy-level (fast) reflection service (Runbook 50).

Verifies:
- Invariant checks block correctly (kill switch, THESIS_ARMED, HOLD_LOCK)
- Coherence checks produce revise findings (empty triggers, direction mismatch)
- Memory contradiction checks surface appropriate findings
- Status transitions: pass / revise / block
- Suppression during activation-window and HOLD_LOCK ticks
- Policy-level reflection only runs when POLICY_REFLECTION_ENABLED=true
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List

import pytest

from schemas.reflection import (
    PolicyLevelReflectionRequest,
    PolicyLevelReflectionResult,
    ReflectionInvocationMeta,
)
from services.low_level_reflection_service import (
    PolicyLevelReflectionService,
    build_reflection_request,
    is_enabled,
    _check_invariants,
    _check_coherence,
    _check_memory,
    _derive_status,
)


_NOW = datetime(2026, 2, 28, 12, 0, 0, tzinfo=timezone.utc)


def _meta() -> ReflectionInvocationMeta:
    return ReflectionInvocationMeta(
        invoked_at=_NOW,
        source="test",
        reflection_kind="policy",
    )


def _req(**kwargs) -> PolicyLevelReflectionRequest:
    """Helper: build a minimal valid request with optional overrides."""
    defaults = dict(
        trigger_count=2,
        allowed_directions=["long"],
        regime="range_bound",
        risk_constraints_present=True,
        meta=_meta(),
    )
    defaults.update(kwargs)
    return PolicyLevelReflectionRequest(**defaults)


# ---------------------------------------------------------------------------
# is_enabled()
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("POLICY_REFLECTION_ENABLED", raising=False)
        assert is_enabled() is False

    def test_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("POLICY_REFLECTION_ENABLED", "true")
        assert is_enabled() is True

    def test_enabled_via_1(self, monkeypatch):
        monkeypatch.setenv("POLICY_REFLECTION_ENABLED", "1")
        assert is_enabled() is True

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("POLICY_REFLECTION_ENABLED", "TRUE")
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# _check_invariants: deterministic blocks
# ---------------------------------------------------------------------------


class TestCheckInvariants:
    def test_kill_switch_blocks(self):
        req = _req(kill_switch_active=True)
        findings, hard_block = _check_invariants(req)
        assert hard_block is True
        assert any("kill switch" in f for f in findings)

    def test_activation_window_tick_blocks(self):
        req = _req(is_activation_window_tick=True, is_hold_lock_tick=False)
        findings, hard_block = _check_invariants(req)
        assert hard_block is True
        assert any("THESIS_ARMED" in f or "activation" in f for f in findings)

    def test_hold_lock_tick_blocks(self):
        req = _req(is_hold_lock_tick=True)
        findings, hard_block = _check_invariants(req)
        assert hard_block is True
        assert any("HOLD_LOCK" in f for f in findings)

    def test_no_risk_constraints_warns(self):
        req = _req(risk_constraints_present=False)
        findings, hard_block = _check_invariants(req)
        assert hard_block is False
        assert any("risk_constraints" in f for f in findings)

    def test_clean_request_passes(self):
        req = _req()
        findings, hard_block = _check_invariants(req)
        assert hard_block is False
        assert not any(f.startswith("BLOCK:") for f in findings)


# ---------------------------------------------------------------------------
# _check_coherence: structural / coherence checks
# ---------------------------------------------------------------------------


class TestCheckCoherence:
    def test_zero_triggers_revise(self):
        req = _req(trigger_count=0)
        findings = _check_coherence(req)
        assert any("REVISE" in f for f in findings)
        assert any("empty" in f.lower() or "no triggers" in f.lower() for f in findings)

    def test_direction_mismatch_revise(self):
        req = _req(direction_summary="long", allowed_directions=["short"])
        findings = _check_coherence(req)
        assert any("REVISE" in f for f in findings)
        assert any("allowed_directions" in f for f in findings)

    def test_playbook_without_template_warns(self):
        req = _req(playbook_id="bollinger_squeeze", template_id=None)
        findings = _check_coherence(req)
        assert any("WARN" in f for f in findings)
        assert any("template" in f.lower() for f in findings)

    def test_missing_regime_warns(self):
        req = _req(regime=None)
        findings = _check_coherence(req)
        assert any("regime" in f.lower() for f in findings)

    def test_consistent_proposal_clean(self):
        req = _req(
            direction_summary="long",
            allowed_directions=["long"],
            regime="bull_trending",
            trigger_count=3,
            playbook_id="donchian_breakout",
            template_id="compression_breakout_long",
        )
        findings = _check_coherence(req)
        # No REVISE findings expected
        assert not any(f.startswith("REVISE:") for f in findings)


# ---------------------------------------------------------------------------
# _check_memory: contrastive memory checks
# ---------------------------------------------------------------------------


class TestCheckMemory:
    def test_no_memory_bundle_returns_empty(self):
        req = _req(memory_bundle_id=None, memory_failure_modes=[])
        findings = _check_memory(req)
        assert findings == []

    def test_false_breakout_long_flagged(self):
        req = _req(
            direction_summary="long",
            memory_bundle_id="bundle-001",
            memory_failure_modes=["false_breakout_reversion"],
        )
        findings = _check_memory(req)
        assert any("false_breakout_reversion" in f for f in findings)
        assert any("long" in f.lower() or "reversed" in f.lower() for f in findings)

    def test_losing_skew_flagged(self):
        req = _req(
            memory_bundle_id="bundle-001",
            memory_failure_modes=[],
            memory_winning_count=1,
            memory_losing_count=5,
        )
        findings = _check_memory(req)
        assert any("skew" in f.lower() or "losing" in f.lower() for f in findings)

    def test_stop_too_tight_flagged(self):
        req = _req(
            memory_bundle_id="bundle-001",
            memory_failure_modes=["stop_too_tight_noise_out"],
        )
        findings = _check_memory(req)
        assert any("stop" in f.lower() for f in findings)


# ---------------------------------------------------------------------------
# _derive_status: status transitions
# ---------------------------------------------------------------------------


class TestDeriveStatus:
    def test_hard_block_gives_block(self):
        assert _derive_status(True, [], []) == "block"

    def test_block_finding_gives_block(self):
        assert _derive_status(False, ["BLOCK: kill switch"], []) == "block"

    def test_revise_finding_gives_revise(self):
        assert _derive_status(False, ["WARN: no risk constraints"], ["REVISE: no triggers"]) == "revise"

    def test_warn_only_gives_pass(self):
        assert _derive_status(False, ["WARN: no risk constraints"], ["WARN: regime missing"]) == "pass"

    def test_clean_gives_pass(self):
        assert _derive_status(False, [], []) == "pass"


# ---------------------------------------------------------------------------
# PolicyLevelReflectionService.reflect: integration
# ---------------------------------------------------------------------------


class TestPolicyLevelReflectionService:
    def setup_method(self):
        self.svc = PolicyLevelReflectionService()

    def test_pass_on_clean_request(self):
        req = _req()
        result = self.svc.reflect(req)
        assert isinstance(result, PolicyLevelReflectionResult)
        assert result.status == "pass"
        assert result.latency_ms >= 0

    def test_block_on_kill_switch(self):
        req = _req(kill_switch_active=True)
        result = self.svc.reflect(req)
        assert result.status == "block"
        assert any("kill switch" in f for f in result.invariant_findings)

    def test_block_on_activation_window_tick(self):
        req = _req(is_activation_window_tick=True)
        result = self.svc.reflect(req)
        assert result.status == "block"

    def test_block_on_hold_lock_tick(self):
        req = _req(is_hold_lock_tick=True)
        result = self.svc.reflect(req)
        assert result.status == "block"

    def test_revise_on_empty_triggers(self):
        req = _req(trigger_count=0)
        result = self.svc.reflect(req)
        assert result.status == "revise"
        assert len(result.requested_revisions) > 0

    def test_revise_on_direction_mismatch(self):
        req = _req(direction_summary="long", allowed_directions=["short"])
        result = self.svc.reflect(req)
        assert result.status == "revise"

    def test_memory_findings_present_when_bundle_given(self):
        req = _req(
            memory_bundle_id="b001",
            memory_failure_modes=["false_breakout_reversion"],
            direction_summary="long",
        )
        result = self.svc.reflect(req)
        assert len(result.memory_findings) > 0

    def test_kill_switch_short_circuits(self):
        """BLOCK from invariants must not incur expensive downstream checks."""
        req = _req(kill_switch_active=True, trigger_count=0, memory_failure_modes=["a", "b", "c"])
        result = self.svc.reflect(req)
        assert result.status == "block"
        # Coherence and memory checks not run on hard block
        assert result.coherence_findings == []
        assert result.memory_findings == []


# ---------------------------------------------------------------------------
# build_reflection_request helper
# ---------------------------------------------------------------------------


class TestBuildReflectionRequest:
    def test_builds_valid_request(self):
        req = build_reflection_request(
            plan_id="plan-001",
            playbook_id="bollinger_squeeze",
            template_id="compression_breakout_long",
            trigger_count=3,
            allowed_directions=["long"],
            regime="range_bound",
            rationale_excerpt="Tight compression...",
            risk_constraints_present=True,
            source="plan_provider",
            policy_event_type="plan_generation",
        )
        assert req.plan_id == "plan-001"
        assert req.meta.source == "plan_provider"
        assert req.meta.reflection_kind == "policy"

    def test_single_direction_summary(self):
        req = build_reflection_request(
            plan_id=None,
            playbook_id=None,
            template_id=None,
            trigger_count=1,
            allowed_directions=["short"],
            regime=None,
            rationale_excerpt=None,
            risk_constraints_present=False,
        )
        assert req.direction_summary == "short"

    def test_multiple_directions_mixed(self):
        req = build_reflection_request(
            plan_id=None,
            playbook_id=None,
            template_id=None,
            trigger_count=2,
            allowed_directions=["long", "short"],
            regime=None,
            rationale_excerpt=None,
            risk_constraints_present=False,
        )
        assert req.direction_summary == "mixed"

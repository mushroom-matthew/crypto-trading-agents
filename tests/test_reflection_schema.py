"""Tests for Runbook 50 reflection schemas.

Verifies that:
- All required fields validate correctly
- extra="forbid" rejects unknown fields
- Literal fields only accept valid values
- Optional fields default correctly
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from schemas.reflection import (
    HighLevelReflectionReport,
    HighLevelReflectionRequest,
    PlaybookFinding,
    PolicyLevelReflectionRequest,
    PolicyLevelReflectionResult,
    ReflectionInvocationMeta,
    RegimeClusterSummary,
    TickValidationResult,
)


_NOW = datetime(2026, 2, 28, 12, 0, 0, tzinfo=timezone.utc)


def _meta(kind: str = "policy") -> ReflectionInvocationMeta:
    return ReflectionInvocationMeta(
        invoked_at=_NOW,
        source="test",
        reflection_kind=kind,
    )


# ---------------------------------------------------------------------------
# ReflectionInvocationMeta
# ---------------------------------------------------------------------------


class TestReflectionInvocationMeta:
    def test_minimal(self):
        m = ReflectionInvocationMeta(
            invoked_at=_NOW,
            source="plan_provider",
            reflection_kind="policy",
        )
        assert m.source == "plan_provider"
        assert m.skip_reason is None
        assert m.latency_ms is None

    def test_with_skip(self):
        m = ReflectionInvocationMeta(
            invoked_at=_NOW,
            source="scheduler",
            reflection_kind="high_level",
            skip_reason="cadence_gate: last run 2h ago",
        )
        assert m.skip_reason == "cadence_gate: last run 2h ago"

    def test_invalid_reflection_kind(self):
        with pytest.raises(ValidationError):
            ReflectionInvocationMeta(
                invoked_at=_NOW,
                source="test",
                reflection_kind="unknown_kind",  # invalid
            )

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ReflectionInvocationMeta(
                invoked_at=_NOW,
                source="test",
                reflection_kind="tick",
                bogus_field="oops",
            )


# ---------------------------------------------------------------------------
# TickValidationResult
# ---------------------------------------------------------------------------


class TestTickValidationResult:
    def test_minimal(self):
        r = TickValidationResult(
            tick_ts=_NOW,
            symbol="BTC-USD",
            timeframe="1h",
            meta=_meta("tick"),
        )
        assert r.stop_breach is False
        assert r.target_breach is False
        assert r.actions_taken == []

    def test_breach_flags(self):
        r = TickValidationResult(
            tick_ts=_NOW,
            symbol="ETH-USD",
            timeframe="5m",
            stop_breach=True,
            active_stop_price=45000.0,
            close_price=44900.0,
            actions_taken=["stop_exit_triggered"],
            meta=_meta("tick"),
        )
        assert r.stop_breach is True
        assert r.actions_taken == ["stop_exit_triggered"]

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            TickValidationResult(
                tick_ts=_NOW,
                symbol="BTC-USD",
                timeframe="1h",
                meta=_meta("tick"),
                mystery_field=True,
            )


# ---------------------------------------------------------------------------
# PolicyLevelReflectionRequest
# ---------------------------------------------------------------------------


class TestPolicyLevelReflectionRequest:
    def test_minimal(self):
        req = PolicyLevelReflectionRequest(meta=_meta())
        assert req.trigger_count == 0
        assert req.is_activation_window_tick is False
        assert req.kill_switch_active is False

    def test_full(self):
        req = PolicyLevelReflectionRequest(
            snapshot_id="snap-001",
            snapshot_hash="abc123",
            plan_id="plan-001",
            playbook_id="bollinger_squeeze",
            template_id="compression_breakout_long",
            direction_summary="long",
            trigger_count=3,
            allowed_directions=["long"],
            regime="range_bound",
            rationale_excerpt="BTC in tight compression...",
            policy_state="THESIS_ARMED",
            is_activation_window_tick=False,
            is_hold_lock_tick=False,
            memory_failure_modes=["false_breakout_reversion"],
            memory_winning_count=2,
            memory_losing_count=4,
            memory_bundle_id="bundle-001",
            risk_constraints_present=True,
            disabled_trigger_ids=[],
            disabled_categories=[],
            kill_switch_active=False,
            playbook_expected_hold_bars_p50=12.0,
            playbook_mae_budget_pct=0.015,
            stated_conviction="high",
            meta=_meta(),
        )
        assert req.playbook_id == "bollinger_squeeze"
        assert req.memory_failure_modes == ["false_breakout_reversion"]

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            PolicyLevelReflectionRequest(
                meta=_meta(),
                not_a_real_field="bad",
            )


# ---------------------------------------------------------------------------
# PolicyLevelReflectionResult
# ---------------------------------------------------------------------------


class TestPolicyLevelReflectionResult:
    def test_pass_result(self):
        r = PolicyLevelReflectionResult(
            status="pass",
            latency_ms=45,
        )
        assert r.status == "pass"
        assert r.coherence_findings == []
        assert r.requested_revisions == []

    def test_revise_result(self):
        r = PolicyLevelReflectionResult(
            status="revise",
            coherence_findings=["REVISE: empty trigger list"],
            requested_revisions=["empty trigger list"],
            latency_ms=80,
        )
        assert r.status == "revise"
        assert "REVISE: empty trigger list" in r.coherence_findings

    def test_block_result(self):
        r = PolicyLevelReflectionResult(
            status="block",
            invariant_findings=["BLOCK: kill switch active"],
            latency_ms=5,
        )
        assert r.status == "block"

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            PolicyLevelReflectionResult(status="unknown", latency_ms=0)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            PolicyLevelReflectionResult(
                status="pass",
                latency_ms=10,
                extra="no",
            )


# ---------------------------------------------------------------------------
# HighLevelReflectionRequest
# ---------------------------------------------------------------------------


class TestHighLevelReflectionRequest:
    def test_minimal(self):
        req = HighLevelReflectionRequest(
            window_start=datetime(2026, 2, 21, tzinfo=timezone.utc),
            window_end=datetime(2026, 2, 28, tzinfo=timezone.utc),
            meta=_meta("high_level"),
        )
        assert req.min_episodes_for_structural_recommendation == 20
        assert req.scheduled_cadence == "daily"
        assert req.force_run is False

    def test_on_demand(self):
        req = HighLevelReflectionRequest(
            window_start=datetime(2026, 2, 21, tzinfo=timezone.utc),
            window_end=datetime(2026, 2, 28, tzinfo=timezone.utc),
            scheduled_cadence="on_demand",
            force_run=True,
            meta=_meta("high_level"),
        )
        assert req.force_run is True

    def test_invalid_cadence(self):
        with pytest.raises(ValidationError):
            HighLevelReflectionRequest(
                window_start=datetime(2026, 2, 21, tzinfo=timezone.utc),
                window_end=datetime(2026, 2, 28, tzinfo=timezone.utc),
                scheduled_cadence="hourly",  # invalid
                meta=_meta("high_level"),
            )


# ---------------------------------------------------------------------------
# RegimeClusterSummary and PlaybookFinding
# ---------------------------------------------------------------------------


class TestRegimeClusterSummary:
    def test_basic(self):
        s = RegimeClusterSummary(
            cluster_key="playbook=bollinger_squeeze|regime=range_bound",
            n_episodes=15,
            win_rate=0.53,
        )
        assert s.win_rate == 0.53
        assert s.dominant_failure_modes == []

    def test_extra_rejected(self):
        with pytest.raises(ValidationError):
            RegimeClusterSummary(
                cluster_key="k",
                n_episodes=5,
                win_rate=0.5,
                bad_field=True,
            )


class TestPlaybookFinding:
    def test_insufficient_sample(self):
        pf = PlaybookFinding(
            playbook_id="bollinger_squeeze",
            n_episodes=5,
            win_rate=0.4,
            structural_change_eligible=False,
            insufficient_sample_reason="Only 5 episodes; need >= 20",
        )
        assert not pf.structural_change_eligible
        assert pf.recommended_action == "hold"

    def test_invalid_recommended_action(self):
        with pytest.raises(ValidationError):
            PlaybookFinding(
                playbook_id="x",
                n_episodes=25,
                win_rate=0.6,
                recommended_action="do_something_else",  # invalid
            )


# ---------------------------------------------------------------------------
# HighLevelReflectionReport
# ---------------------------------------------------------------------------


class TestHighLevelReflectionReport:
    def test_insufficient_sample_report(self):
        report = HighLevelReflectionReport(
            window_start=datetime(2026, 2, 21, tzinfo=timezone.utc),
            window_end=datetime(2026, 2, 28, tzinfo=timezone.utc),
            n_episodes=8,
            insufficient_sample=True,
            insufficient_sample_reason="Only 8 episodes; need >= 20",
            structural_recommendations_suppressed=True,
            meta=_meta("high_level"),
        )
        assert report.insufficient_sample is True
        assert report.n_episodes == 8
        assert report.regime_cluster_summary == []
        assert report.playbook_findings == []

    def test_full_report(self):
        report = HighLevelReflectionReport(
            window_start=datetime(2026, 2, 21, tzinfo=timezone.utc),
            window_end=datetime(2026, 2, 28, tzinfo=timezone.utc),
            n_episodes=30,
            regime_cluster_summary=[
                RegimeClusterSummary(
                    cluster_key="playbook=bollinger_squeeze|regime=range_bound",
                    n_episodes=25,
                    win_rate=0.60,
                )
            ],
            playbook_findings=[
                PlaybookFinding(
                    playbook_id="bollinger_squeeze",
                    n_episodes=25,
                    win_rate=0.60,
                    structural_change_eligible=True,
                    recommended_action="hold",
                )
            ],
            drift_findings=["DRIFT: breakout failure co-occurrence observed"],
            recommendations=[],
            evidence_refs=["ep-001", "ep-002"],
            meta=_meta("high_level"),
        )
        assert report.n_episodes == 30
        assert not report.insufficient_sample
        assert len(report.regime_cluster_summary) == 1

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            HighLevelReflectionReport(
                window_start=_NOW,
                window_end=_NOW,
                n_episodes=0,
                meta=_meta("high_level"),
                hacker_field="pwned",
            )

"""Tests for high-level (slow) reflection service (Runbook 50).

Verifies:
- Cadence gating (should_run_high_level_reflection)
- Minimum sample size gate for structural recommendations
- Insufficient-sample mode: monitor-only output, no structural recommendations
- Outcome clustering and playbook-level findings computation
- Drift findings emission
- force_run bypasses time gate
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List
from uuid import uuid4

import pytest

from schemas.episode_memory import EpisodeMemoryRecord
from schemas.reflection import (
    HighLevelReflectionReport,
    HighLevelReflectionRequest,
    ReflectionInvocationMeta,
)
from services.high_level_reflection_service import (
    HighLevelReflectionService,
    should_run_high_level_reflection,
    _cluster_key,
    _win_rate,
    _dominant_failure_modes,
)


_NOW = datetime(2026, 2, 28, 12, 0, 0, tzinfo=timezone.utc)
_WIN = datetime(2026, 2, 21, 0, 0, 0, tzinfo=timezone.utc)
_WEND = datetime(2026, 2, 28, 0, 0, 0, tzinfo=timezone.utc)


def _meta() -> ReflectionInvocationMeta:
    return ReflectionInvocationMeta(
        invoked_at=_NOW,
        source="scheduler",
        reflection_kind="high_level",
    )


def _req(**kwargs) -> HighLevelReflectionRequest:
    defaults = dict(
        window_start=_WIN,
        window_end=_WEND,
        meta=_meta(),
    )
    defaults.update(kwargs)
    return HighLevelReflectionRequest(**defaults)


def _episode(
    playbook_id: str = "bollinger_squeeze",
    outcome: str = "win",
    failure_modes: List[str] | None = None,
    template_id: str | None = "compression_breakout_long",
    r_achieved: float = 1.5,
    hold_bars: int = 12,
    mae_pct: float = 0.008,
    mfe_pct: float = 0.025,
) -> EpisodeMemoryRecord:
    return EpisodeMemoryRecord(
        episode_id=str(uuid4()),
        symbol="BTC-USD",
        playbook_id=playbook_id,
        template_id=template_id,
        outcome_class=outcome,
        failure_modes=failure_modes or [],
        r_achieved=r_achieved,
        hold_bars=hold_bars,
        mae_pct=mae_pct,
        mfe_pct=mfe_pct,
    )


def _win_episodes(n: int, playbook_id: str = "bollinger_squeeze") -> List[EpisodeMemoryRecord]:
    return [_episode(playbook_id=playbook_id, outcome="win") for _ in range(n)]


def _loss_episodes(n: int, playbook_id: str = "bollinger_squeeze", failure_modes: List[str] | None = None) -> List[EpisodeMemoryRecord]:
    return [_episode(playbook_id=playbook_id, outcome="loss", r_achieved=-1.0, failure_modes=failure_modes) for _ in range(n)]


# ---------------------------------------------------------------------------
# should_run_high_level_reflection (cadence gate)
# ---------------------------------------------------------------------------


class TestCadenceGate:
    def test_no_prior_run_allows(self):
        ok, reason = should_run_high_level_reflection(None, "daily", now=_NOW)
        assert ok is True
        assert reason is None

    def test_recent_run_blocks_daily(self):
        last_run = _NOW - timedelta(hours=2)
        ok, reason = should_run_high_level_reflection(last_run, "daily", now=_NOW)
        assert ok is False
        assert reason is not None
        assert "gate" in reason.lower() or "ago" in reason.lower()

    def test_stale_daily_run_allows(self):
        last_run = _NOW - timedelta(hours=24)
        ok, reason = should_run_high_level_reflection(last_run, "daily", now=_NOW)
        assert ok is True

    def test_recent_run_blocks_weekly(self):
        last_run = _NOW - timedelta(days=3)
        ok, reason = should_run_high_level_reflection(last_run, "weekly", now=_NOW)
        assert ok is False

    def test_stale_weekly_run_allows(self):
        last_run = _NOW - timedelta(days=8)
        ok, reason = should_run_high_level_reflection(last_run, "weekly", now=_NOW)
        assert ok is True

    def test_force_run_bypasses_gate(self):
        last_run = _NOW - timedelta(minutes=5)
        ok, reason = should_run_high_level_reflection(last_run, "daily", now=_NOW, force_run=True)
        assert ok is True
        assert reason is None

    def test_naive_datetime_handled(self):
        """Naive last_run datetime should not crash — treated as UTC."""
        last_run = datetime(2026, 2, 28, 11, 0, 0)  # no tzinfo
        ok, reason = should_run_high_level_reflection(last_run, "daily", now=_NOW)
        # 1h ago — should still be within daily gate
        assert ok is False


# ---------------------------------------------------------------------------
# Cluster / grouping helpers
# ---------------------------------------------------------------------------


class TestClusterHelpers:
    def test_cluster_key_uses_playbook_and_template(self):
        ep = _episode(playbook_id="donchian_breakout", template_id="volatile_breakout_long")
        key = _cluster_key(ep)
        assert "donchian_breakout" in key
        assert "volatile_breakout_long" in key

    def test_win_rate_all_wins(self):
        eps = _win_episodes(5)
        assert _win_rate(eps) == 1.0

    def test_win_rate_all_losses(self):
        eps = _loss_episodes(5)
        assert _win_rate(eps) == 0.0

    def test_win_rate_mixed(self):
        eps = _win_episodes(3) + _loss_episodes(1)
        rate = _win_rate(eps)
        assert abs(rate - 0.75) < 1e-9

    def test_win_rate_empty(self):
        assert _win_rate([]) == 0.0

    def test_dominant_failure_modes(self):
        eps = (
            _loss_episodes(3, failure_modes=["false_breakout_reversion"])
            + _loss_episodes(2, failure_modes=["stop_too_tight_noise_out"])
            + _loss_episodes(1, failure_modes=["late_entry_poor_r_multiple"])
        )
        modes = _dominant_failure_modes(eps, top_n=2)
        assert modes[0] == "false_breakout_reversion"
        assert len(modes) == 2


# ---------------------------------------------------------------------------
# HighLevelReflectionService: insufficient sample
# ---------------------------------------------------------------------------


class TestInsufficientSample:
    def setup_method(self):
        self.svc = HighLevelReflectionService()

    def test_zero_episodes_insufficient(self):
        req = _req()
        report = self.svc.reflect(req, episodes=[])
        assert report.insufficient_sample is True
        assert report.n_episodes == 0
        assert report.insufficient_sample_reason is not None
        assert report.structural_recommendations_suppressed is True

    def test_below_threshold_insufficient(self):
        episodes = _win_episodes(10) + _loss_episodes(5)  # 15 total, need 20
        req = _req()
        report = self.svc.reflect(req, episodes=episodes)
        assert report.insufficient_sample is True
        assert "15" in report.insufficient_sample_reason

    def test_insufficient_sample_still_produces_monitor_findings(self):
        """Even with insufficient sample, cluster analysis should run."""
        episodes = _win_episodes(5) + _loss_episodes(10)  # 15 — below gate
        req = _req()
        report = self.svc.reflect(req, episodes=episodes)
        assert report.insufficient_sample is True
        # Monitor-only cluster summaries still emitted
        assert len(report.regime_cluster_summary) > 0

    def test_insufficient_no_structural_recommendations(self):
        """Recommendations must be monitor-only when sample is insufficient."""
        episodes = _loss_episodes(10, failure_modes=["false_breakout_reversion"])
        req = _req()
        report = self.svc.reflect(req, episodes=episodes)
        assert report.insufficient_sample is True
        for rec in report.recommendations:
            assert rec.get("monitor_only") is True or rec.get("structural_change_eligible") is False


# ---------------------------------------------------------------------------
# HighLevelReflectionService: sufficient sample
# ---------------------------------------------------------------------------


class TestSufficientSample:
    def setup_method(self):
        self.svc = HighLevelReflectionService()

    def _make_large_batch(self, n_wins: int = 15, n_losses: int = 10) -> List[EpisodeMemoryRecord]:
        return _win_episodes(n_wins) + _loss_episodes(n_losses, failure_modes=["false_breakout_reversion"])

    def test_sufficient_sample_not_insufficient(self):
        eps = self._make_large_batch(15, 10)  # 25 total
        req = _req()
        report = self.svc.reflect(req, episodes=eps)
        assert report.insufficient_sample is False
        assert report.n_episodes == 25

    def test_cluster_summaries_produced(self):
        eps = self._make_large_batch()
        req = _req()
        report = self.svc.reflect(req, episodes=eps)
        assert len(report.regime_cluster_summary) >= 1
        for cluster in report.regime_cluster_summary:
            assert cluster.n_episodes > 0
            assert 0.0 <= cluster.win_rate <= 1.0

    def test_playbook_findings_produced(self):
        eps = self._make_large_batch()
        req = _req()
        report = self.svc.reflect(req, episodes=eps)
        assert len(report.playbook_findings) >= 1
        pb = report.playbook_findings[0]
        assert pb.playbook_id is not None
        assert pb.n_episodes > 0

    def test_window_metadata_correct(self):
        eps = self._make_large_batch()
        req = _req()
        report = self.svc.reflect(req, episodes=eps)
        assert report.window_start == _WIN
        assert report.window_end == _WEND

    def test_evidence_refs_populated(self):
        eps = self._make_large_batch()
        req = _req()
        report = self.svc.reflect(req, episodes=eps)
        assert len(report.evidence_refs) > 0

    def test_meta_preserved(self):
        eps = self._make_large_batch()
        req = _req()
        report = self.svc.reflect(req, episodes=eps)
        assert report.meta.source == "scheduler"

    def test_drift_findings_for_low_win_rate(self):
        """A cluster with win_rate < 0.35 and >= 5 episodes should trigger drift."""
        eps = _win_episodes(1) + _loss_episodes(9)  # 10 total, 10% win rate
        req = _req()
        report = self.svc.reflect(req, episodes=eps)
        assert any("DRIFT" in f for f in report.drift_findings)

    def test_high_win_rate_no_negative_recommendations(self):
        """Strong win rate should not produce research_experiment recommendations."""
        eps = _win_episodes(25)  # all wins
        req = _req()
        report = self.svc.reflect(req, episodes=eps)
        for rec in report.recommendations:
            assert rec.get("action") != "research_experiment"

    def test_low_win_rate_policy_adjust_or_experiment(self):
        """Win rate < 0.40 with enough samples should flag policy_adjust or research."""
        eps = _win_episodes(5) + _loss_episodes(20)  # 25 total, 20% win rate
        req = _req()
        report = self.svc.reflect(req, episodes=eps)
        actions = [rec.get("action") for rec in report.recommendations]
        assert any(a in ("policy_adjust", "research_experiment") for a in actions)


# ---------------------------------------------------------------------------
# high_level_reflection: per-playbook insufficient sample gate
# ---------------------------------------------------------------------------


class TestPerPlaybookSampleGate:
    def setup_method(self):
        self.svc = HighLevelReflectionService()

    def test_per_playbook_structural_eligibility(self):
        """Each playbook finding must be individually gated by its own sample size."""
        # 30 total — structural eligible overall — but split across 4 playbooks
        eps = (
            _win_episodes(8, "pb_a") + _loss_episodes(2, "pb_a")
            + _win_episodes(8, "pb_b") + _loss_episodes(2, "pb_b")
            + _win_episodes(3, "pb_c") + _loss_episodes(2, "pb_c")  # only 5 — below regime gate
            + _win_episodes(3, "pb_d") + _loss_episodes(2, "pb_d")  # only 5 — below regime gate
        )
        req = _req(min_regime_cluster_samples=10)
        report = self.svc.reflect(req, episodes=eps)
        assert report.insufficient_sample is False  # 30 overall — ok

        by_pb = {pf.playbook_id: pf for pf in report.playbook_findings}

        # pb_a and pb_b have 10 each — should be structural eligible
        if "pb_a" in by_pb:
            assert by_pb["pb_a"].structural_change_eligible is True
        if "pb_b" in by_pb:
            assert by_pb["pb_b"].structural_change_eligible is True

        # pb_c and pb_d have 5 each — NOT structural eligible
        if "pb_c" in by_pb:
            assert by_pb["pb_c"].structural_change_eligible is False
            assert by_pb["pb_c"].insufficient_sample_reason is not None
        if "pb_d" in by_pb:
            assert by_pb["pb_d"].structural_change_eligible is False

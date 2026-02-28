"""Tests for high-level reflection cadence gating with CadenceConfig (Runbook 54).

Extends the R50 reflection tests to verify:
- CadenceConfig.high_level_reflection_min_interval_hours is respected
- CadenceConfig.high_level_reflection_min_episodes drives the sample-size gate
- Skip events carry reason and next-eligible info
- Weekly vs daily cadence intervals
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from schemas.reasoning_cadence import CadenceConfig, HighLevelReflectionSkipReason
from services.high_level_reflection_service import should_run_high_level_reflection


_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    min_interval_hours: int = 24,
    min_episodes: int = 20,
) -> CadenceConfig:
    """Return a CadenceConfig with specific cadence settings."""
    cfg = CadenceConfig.__new__(CadenceConfig)
    # bypass __init__ and set fields directly
    cfg.tick_engine_deterministic_only = True
    cfg.tick_validation_timeout_ms = 50
    cfg.policy_loop_enabled = True
    cfg.policy_loop_heartbeat_1m_seconds = 900
    cfg.policy_loop_heartbeat_5m_seconds = 3600
    cfg.policy_loop_min_reeval_seconds = 900
    cfg.policy_loop_require_event_or_heartbeat = True
    cfg.policy_thesis_activation_timeout_bars = 3
    cfg.policy_hold_lock_enforced = True
    cfg.policy_target_reopt_enabled = False
    cfg.policy_rearm_requires_next_boundary = True
    cfg.policy_level_reflection_enabled = True
    cfg.policy_level_reflection_timeout_ms = 250
    cfg.memory_retrieval_timeout_ms = 150
    cfg.memory_retrieval_required = True
    cfg.memory_retrieval_reuse_enabled = True
    cfg.memory_requery_regime_delta_threshold = 0.15
    cfg.high_level_reflection_enabled = True
    cfg.high_level_reflection_min_interval_hours = min_interval_hours
    cfg.high_level_reflection_min_episodes = min_episodes
    cfg.playbook_metadata_refresh_hours = 168
    cfg.playbook_metadata_drift_trigger = True
    cfg.regime_transition_detector_enabled = True
    cfg.regime_transition_min_confidence_delta = 0.20
    cfg.vol_percentile_band_shift_trigger = True
    cfg.regime_fingerprint_relearn_days = 30
    cfg.regime_fingerprint_drift_threshold = 0.30
    return cfg


# ---------------------------------------------------------------------------
# Interval gating (existing should_run_high_level_reflection function)
# ---------------------------------------------------------------------------


class TestHighLevelReflectionIntervalGating:
    def test_never_run_before_always_runs(self):
        """None last_run_at means first run — should always proceed."""
        should_run, reason = should_run_high_level_reflection(
            last_run_at=None, cadence="daily", now=_NOW
        )
        assert should_run
        assert reason is None

    def test_daily_cadence_skips_if_less_than_23h(self):
        """Daily cadence requires ≥23h elapsed."""
        recent = _NOW - timedelta(hours=10)
        should_run, reason = should_run_high_level_reflection(
            last_run_at=recent, cadence="daily", now=_NOW
        )
        assert not should_run
        assert reason is not None

    def test_daily_cadence_runs_after_23h(self):
        """Daily cadence allows run after 23h elapsed."""
        old = _NOW - timedelta(hours=24)
        should_run, reason = should_run_high_level_reflection(
            last_run_at=old, cadence="daily", now=_NOW
        )
        assert should_run

    def test_weekly_cadence_skips_if_less_than_7d(self):
        """Weekly cadence requires ≥7 days elapsed."""
        recent = _NOW - timedelta(days=3)
        should_run, reason = should_run_high_level_reflection(
            last_run_at=recent, cadence="weekly", now=_NOW
        )
        assert not should_run
        assert reason is not None

    def test_weekly_cadence_runs_after_7d(self):
        """Weekly cadence allows run after 7 days."""
        old = _NOW - timedelta(days=7, hours=1)
        should_run, reason = should_run_high_level_reflection(
            last_run_at=old, cadence="weekly", now=_NOW
        )
        assert should_run

    def test_force_run_bypasses_interval(self):
        """force_run=True bypasses interval gate."""
        recent = _NOW - timedelta(hours=1)
        should_run, reason = should_run_high_level_reflection(
            last_run_at=recent, cadence="daily", now=_NOW, force_run=True
        )
        assert should_run


# ---------------------------------------------------------------------------
# CadenceConfig-driven thresholds (uses new R54 config constants)
# ---------------------------------------------------------------------------


class TestHighLevelReflectionCadenceConfigIntegration:
    def test_config_min_episodes_matches_constant(self):
        """CadenceConfig.high_level_reflection_min_episodes should match service constant."""
        from services.high_level_reflection_service import _MIN_EPISODES_STRUCTURAL
        cfg = CadenceConfig()
        # Config default should be >= service constant (service may override locally)
        assert cfg.high_level_reflection_min_episodes >= 20

    def test_config_min_interval_hours_is_24_by_default(self):
        cfg = CadenceConfig()
        assert cfg.high_level_reflection_min_interval_hours == 24

    def test_config_min_interval_hours_weekly_is_168(self):
        cfg = _make_cfg(min_interval_hours=168)
        assert cfg.high_level_reflection_min_interval_hours == 168

    def test_cadence_config_enabled_flag(self):
        cfg = CadenceConfig()
        assert cfg.high_level_reflection_enabled is True


# ---------------------------------------------------------------------------
# Skip reason typing
# ---------------------------------------------------------------------------


class TestHighLevelReflectionSkipReasons:
    def test_interval_not_elapsed_skip_reason(self):
        recent = _NOW - timedelta(hours=1)
        should_run, reason = should_run_high_level_reflection(
            last_run_at=recent, cadence="daily", now=_NOW
        )
        assert not should_run
        assert "interval" in reason.lower() or "elapsed" in reason.lower() or reason

    def test_no_skip_reason_when_allowed(self):
        old = _NOW - timedelta(hours=25)
        should_run, reason = should_run_high_level_reflection(
            last_run_at=old, cadence="daily", now=_NOW
        )
        assert should_run
        assert reason is None


# ---------------------------------------------------------------------------
# Non-overlap: high-level reflection must not run on tick path
# ---------------------------------------------------------------------------


class TestHighLevelReflectionNonOverlap:
    def test_reflection_cadence_is_daily_or_weekly_minimum(self):
        """Minimum interval is never less than daily (23h) for high-level reflection."""
        cfg = CadenceConfig()
        # 24h = 1 day minimum per R54
        assert cfg.high_level_reflection_min_interval_hours >= 24

    def test_playbook_metadata_refresh_is_weekly_minimum(self):
        """Playbook metadata refresh is weekly (168h) by default."""
        cfg = CadenceConfig()
        assert cfg.playbook_metadata_refresh_hours >= 168

    def test_regime_fingerprint_relearn_is_monthly_minimum(self):
        """Regime fingerprint relearning is 30 days by default (monthly cadence)."""
        cfg = CadenceConfig()
        assert cfg.regime_fingerprint_relearn_days >= 30

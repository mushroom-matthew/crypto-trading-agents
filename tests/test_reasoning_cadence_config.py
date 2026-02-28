"""Tests for CadenceConfig schema (Runbook 54).

Verifies:
- All fields load from env vars with correct defaults
- Env override works for each field type (bool, int, float)
- heartbeat_for_timeframe returns correct values
- get_cadence_config() singleton behaviour
"""

import os

import pytest

from schemas.reasoning_cadence import (
    CadenceConfig,
    CadenceTelemetrySnapshot,
    PolicyLoopSkipEvent,
    PolicyLoopTriggerEvent,
    get_cadence_config,
)
from datetime import datetime, timezone

_NOW = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestCadenceConfigDefaults:
    def test_tick_engine_defaults(self):
        cfg = CadenceConfig()
        assert cfg.tick_engine_deterministic_only is True
        assert cfg.tick_validation_timeout_ms == 50

    def test_policy_loop_defaults(self):
        cfg = CadenceConfig()
        assert cfg.policy_loop_enabled is True
        assert cfg.policy_loop_heartbeat_1m_seconds == 900
        assert cfg.policy_loop_heartbeat_5m_seconds == 3600
        assert cfg.policy_loop_min_reeval_seconds == 900
        assert cfg.policy_loop_require_event_or_heartbeat is True

    def test_intra_policy_defaults(self):
        cfg = CadenceConfig()
        assert cfg.policy_thesis_activation_timeout_bars == 3
        assert cfg.policy_hold_lock_enforced is True
        assert cfg.policy_target_reopt_enabled is False
        assert cfg.policy_rearm_requires_next_boundary is True

    def test_policy_level_reflection_defaults(self):
        cfg = CadenceConfig()
        assert cfg.policy_level_reflection_enabled is True
        assert cfg.policy_level_reflection_timeout_ms == 250

    def test_memory_retrieval_defaults(self):
        cfg = CadenceConfig()
        assert cfg.memory_retrieval_timeout_ms == 150
        assert cfg.memory_retrieval_required is True
        assert cfg.memory_retrieval_reuse_enabled is True
        assert cfg.memory_requery_regime_delta_threshold == pytest.approx(0.15)

    def test_high_level_reflection_defaults(self):
        cfg = CadenceConfig()
        assert cfg.high_level_reflection_enabled is True
        assert cfg.high_level_reflection_min_interval_hours == 24
        assert cfg.high_level_reflection_min_episodes == 20

    def test_playbook_metadata_defaults(self):
        cfg = CadenceConfig()
        assert cfg.playbook_metadata_refresh_hours == 168   # weekly
        assert cfg.playbook_metadata_drift_trigger is True

    def test_regime_transition_defaults(self):
        cfg = CadenceConfig()
        assert cfg.regime_transition_detector_enabled is True
        assert cfg.regime_transition_min_confidence_delta == pytest.approx(0.20)
        assert cfg.vol_percentile_band_shift_trigger is True

    def test_regime_fingerprint_defaults(self):
        cfg = CadenceConfig()
        assert cfg.regime_fingerprint_relearn_days == 30
        assert cfg.regime_fingerprint_drift_threshold == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------


class TestCadenceConfigEnvOverrides:
    def test_bool_override_true(self, monkeypatch):
        monkeypatch.setenv("POLICY_TARGET_REOPT_ENABLED", "true")
        cfg = CadenceConfig()
        assert cfg.policy_target_reopt_enabled is True

    def test_bool_override_1(self, monkeypatch):
        monkeypatch.setenv("TICK_ENGINE_DETERMINISTIC_ONLY", "0")
        cfg = CadenceConfig()
        assert cfg.tick_engine_deterministic_only is False

    def test_int_override(self, monkeypatch):
        monkeypatch.setenv("POLICY_THESIS_ACTIVATION_TIMEOUT_BARS", "5")
        cfg = CadenceConfig()
        assert cfg.policy_thesis_activation_timeout_bars == 5

    def test_float_override(self, monkeypatch):
        monkeypatch.setenv("MEMORY_REQUERY_REGIME_DELTA_THRESHOLD", "0.25")
        cfg = CadenceConfig()
        assert cfg.memory_requery_regime_delta_threshold == pytest.approx(0.25)

    def test_invalid_int_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEMORY_RETRIEVAL_TIMEOUT_MS", "not_a_number")
        cfg = CadenceConfig()
        assert cfg.memory_retrieval_timeout_ms == 150   # default

    def test_invalid_float_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("REGIME_FINGERPRINT_DRIFT_THRESHOLD", "bad")
        cfg = CadenceConfig()
        assert cfg.regime_fingerprint_drift_threshold == pytest.approx(0.30)

    def test_heartbeat_custom(self, monkeypatch):
        monkeypatch.setenv("POLICY_LOOP_HEARTBEAT_1M_SECONDS", "600")
        cfg = CadenceConfig()
        assert cfg.policy_loop_heartbeat_1m_seconds == 600

    def test_high_level_min_episodes_override(self, monkeypatch):
        monkeypatch.setenv("HIGH_LEVEL_REFLECTION_MIN_EPISODES", "30")
        cfg = CadenceConfig()
        assert cfg.high_level_reflection_min_episodes == 30


# ---------------------------------------------------------------------------
# heartbeat_for_timeframe
# ---------------------------------------------------------------------------


class TestHeartbeatForTimeframe:
    def test_1m_timeframe_returns_1m_heartbeat(self):
        cfg = CadenceConfig()
        assert cfg.heartbeat_for_timeframe("1m") == cfg.policy_loop_heartbeat_1m_seconds

    def test_1min_alias_returns_1m_heartbeat(self):
        cfg = CadenceConfig()
        assert cfg.heartbeat_for_timeframe("1min") == cfg.policy_loop_heartbeat_1m_seconds

    def test_5m_timeframe_returns_5m_heartbeat(self):
        cfg = CadenceConfig()
        assert cfg.heartbeat_for_timeframe("5m") == cfg.policy_loop_heartbeat_5m_seconds

    def test_unknown_timeframe_returns_5m_heartbeat(self):
        cfg = CadenceConfig()
        assert cfg.heartbeat_for_timeframe("1h") == cfg.policy_loop_heartbeat_5m_seconds

    def test_empty_timeframe_returns_5m_heartbeat(self):
        cfg = CadenceConfig()
        assert cfg.heartbeat_for_timeframe("") == cfg.policy_loop_heartbeat_5m_seconds


# ---------------------------------------------------------------------------
# get_cadence_config singleton
# ---------------------------------------------------------------------------


class TestGetCadenceConfig:
    def test_returns_cadence_config_instance(self):
        cfg = get_cadence_config()
        assert isinstance(cfg, CadenceConfig)

    def test_singleton_returns_same_object(self):
        cfg1 = get_cadence_config()
        cfg2 = get_cadence_config()
        assert cfg1 is cfg2


# ---------------------------------------------------------------------------
# Telemetry schema round-trips
# ---------------------------------------------------------------------------


class TestTelemetrySchemas:
    def test_policy_loop_skip_event_valid(self):
        event = PolicyLoopSkipEvent(
            skip_reason="no_trigger_and_no_heartbeat",
            skipped_at=_NOW,
        )
        assert event.skip_reason == "no_trigger_and_no_heartbeat"
        assert event.next_eligible_at is None

    def test_policy_loop_trigger_event_valid(self):
        event = PolicyLoopTriggerEvent(
            trigger_type="regime_state_changed",
            fired_at=_NOW,
            source_detail="distance=0.35",
        )
        assert event.trigger_type == "regime_state_changed"

    def test_cadence_telemetry_snapshot_defaults(self):
        snap = CadenceTelemetrySnapshot(snapshot_at=_NOW)
        assert snap.policy_loop_skip_counts == {}
        assert snap.high_level_reflection_skip_counts == {}
        assert snap.current_policy_state is None

    def test_cadence_telemetry_snapshot_extra_fields_forbidden(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CadenceTelemetrySnapshot(snapshot_at=_NOW, unknown_field="x")  # type: ignore[call-arg]

    def test_skip_event_extra_fields_forbidden(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PolicyLoopSkipEvent(
                skip_reason="already_running",
                skipped_at=_NOW,
                surprise="bad",  # type: ignore[call-arg]
            )

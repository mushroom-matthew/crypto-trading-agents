from datetime import datetime, timezone

from tools.paper_trading import (
    _should_emit_session_intent_generated_event,
    _should_generate_initial_session_intent,
    _should_refresh_session_intent_on_regime_drift,
)


def test_initial_session_intent_patch_enabled_always_generates():
    assert _should_generate_initial_session_intent(
        "paper-trading-anything",
        workflow_started_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        patch_enabled=True,
    ) is True


def test_initial_session_intent_pre_r76_defaults_to_skip():
    assert _should_generate_initial_session_intent(
        "paper-trading-legacy",
        workflow_started_at=datetime(2026, 4, 8, 23, 59, 59, tzinfo=timezone.utc),
        patch_enabled=False,
    ) is False


def test_initial_session_intent_post_r76_defaults_to_generate():
    assert _should_generate_initial_session_intent(
        "paper-trading-r76",
        workflow_started_at=datetime(2026, 4, 9, 0, 45, 39, tzinfo=timezone.utc),
        patch_enabled=False,
    ) is True


def test_session_intent_emit_patch_enabled_always_emits():
    assert _should_emit_session_intent_generated_event(
        "paper-trading-anything",
        patch_enabled=True,
    ) is True


def test_session_intent_emit_known_legacy_skip_session():
    assert _should_emit_session_intent_generated_event(
        "paper-trading-7d9d40a8",
        patch_enabled=False,
    ) is False


def test_session_intent_emit_known_legacy_emit_session():
    assert _should_emit_session_intent_generated_event(
        "paper-trading-774b71a4",
        patch_enabled=False,
    ) is True


def test_session_intent_emit_unknown_prepatch_defaults_to_skip():
    assert _should_emit_session_intent_generated_event(
        "paper-trading-unknown",
        patch_enabled=False,
    ) is False


def test_regime_drift_refresh_patch_enabled_always_refreshes():
    assert _should_refresh_session_intent_on_regime_drift(
        "paper-trading-anything",
        workflow_started_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        patch_enabled=True,
    ) is True


def test_regime_drift_refresh_pre_r94_defaults_to_skip():
    assert _should_refresh_session_intent_on_regime_drift(
        "paper-trading-pre-r94",
        workflow_started_at=datetime(2026, 4, 26, 17, 21, 16, tzinfo=timezone.utc),
        patch_enabled=False,
    ) is False


def test_regime_drift_refresh_post_r94_defaults_to_refresh():
    assert _should_refresh_session_intent_on_regime_drift(
        "paper-trading-post-r94",
        workflow_started_at=datetime(2026, 4, 26, 17, 21, 17, tzinfo=timezone.utc),
        patch_enabled=False,
    ) is True

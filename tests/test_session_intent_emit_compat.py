from tools.paper_trading import _should_emit_session_intent_generated_event


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

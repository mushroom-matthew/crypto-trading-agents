from tools.paper_trading import _should_refresh_live_structure_snapshots


def test_live_structure_refresh_patch_enabled_always_refreshes():
    assert _should_refresh_live_structure_snapshots(
        "paper-trading-anything",
        patch_enabled=True,
    ) is True


def test_live_structure_refresh_known_legacy_skip_session():
    assert _should_refresh_live_structure_snapshots(
        "paper-trading-06c35370",
        patch_enabled=False,
    ) is False


def test_live_structure_refresh_unknown_prepatch_defaults_to_skip():
    assert _should_refresh_live_structure_snapshots(
        "paper-trading-unknown",
        patch_enabled=False,
    ) is False

import sys
import importlib
import pytest


def test_agent_worker_blocks_legacy_import(monkeypatch):
    from worker import agent_worker

    # Inject a fake legacy module to simulate accidental import.
    sys.modules["legacy.fake"] = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("legacy.fake", None)
    )
    with pytest.raises(RuntimeError):
        agent_worker._assert_no_legacy_modules()
    sys.modules.pop("legacy.fake", None)


def test_agent_worker_allows_clean_imports():
    from worker import agent_worker

    agent_worker._assert_no_legacy_modules()  # should not raise

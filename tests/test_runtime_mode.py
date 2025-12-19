import os
import importlib
import pytest


def test_live_requires_ack(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.delenv("LIVE_TRADING_ACK", raising=False)
    monkeypatch.delenv("TRADING_STACK", raising=False)
    from agents import runtime_mode

    importlib.reload(runtime_mode)
    with pytest.raises(RuntimeError):
        runtime_mode.get_runtime_mode.cache_clear()
        runtime_mode.get_runtime_mode()


def test_paper_defaults(monkeypatch):
    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.delenv("LIVE_TRADING_ACK", raising=False)
    monkeypatch.delenv("TRADING_STACK", raising=False)
    from agents import runtime_mode

    importlib.reload(runtime_mode)
    runtime_mode.get_runtime_mode.cache_clear()
    mode = runtime_mode.get_runtime_mode()
    assert mode.mode == "paper"
    assert mode.stack == "agent"
    assert mode.live_trading_ack is False

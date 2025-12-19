from decimal import Decimal
import importlib

import pytest


def test_paper_wallet_provider_default(monkeypatch):
    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.delenv("LIVE_TRADING_ACK", raising=False)
    monkeypatch.delenv("TRADING_STACK", raising=False)

    from agents import wallet_provider, runtime_mode

    importlib.reload(wallet_provider)
    runtime_mode.get_runtime_mode.cache_clear()
    provider = wallet_provider.get_wallet_provider({"CASH": Decimal("1000")})

    assert isinstance(provider, wallet_provider.PaperWalletProvider)
    assert provider.get_balance("CASH") == Decimal("1000")


def test_live_wallet_requires_ack(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.delenv("LIVE_TRADING_ACK", raising=False)

    from agents import wallet_provider, runtime_mode

    importlib.reload(wallet_provider)
    runtime_mode.get_runtime_mode.cache_clear()
    with pytest.raises(RuntimeError):
        wallet_provider.LiveWalletProvider()


def test_live_wallet_placeholder(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("LIVE_TRADING_ACK", "true")

    from agents import wallet_provider, runtime_mode

    importlib.reload(wallet_provider)
    runtime_mode.get_runtime_mode.cache_clear()
    provider = wallet_provider.get_wallet_provider()

    assert isinstance(provider, wallet_provider.LiveWalletProvider)
    with pytest.raises(RuntimeError):
        provider.get_balance("CASH")

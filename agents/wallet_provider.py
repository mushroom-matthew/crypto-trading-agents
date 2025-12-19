"""Wallet provider interface with paper/live gating."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict

from agents.runtime_mode import get_runtime_mode


class WalletProvider(ABC):
    """Interface for wallet operations."""

    @abstractmethod
    def get_balance(self, symbol: str) -> Decimal: ...

    @abstractmethod
    def debit(self, symbol: str, amount: Decimal) -> None: ...

    @abstractmethod
    def credit(self, symbol: str, amount: Decimal) -> None: ...


@dataclass
class PaperWalletProvider(WalletProvider):
    balances: Dict[str, Decimal]

    def get_balance(self, symbol: str) -> Decimal:
        return self.balances.get(symbol, Decimal("0"))

    def debit(self, symbol: str, amount: Decimal) -> None:
        self.balances[symbol] = self.get_balance(symbol) - amount

    def credit(self, symbol: str, amount: Decimal) -> None:
        self.balances[symbol] = self.get_balance(symbol) + amount


class LiveWalletProvider(WalletProvider):
    """Placeholder for live wallet integration (Coinbase etc.)."""

    def __init__(self) -> None:
        runtime = get_runtime_mode()
        if runtime.mode != "live" or not runtime.live_trading_ack:
            raise RuntimeError("Live wallet provider requires TRADING_MODE=live and LIVE_TRADING_ACK=true")
        self.runtime = runtime

    def get_balance(self, symbol: str) -> Decimal:
        raise RuntimeError("Live wallet provider not yet implemented")

    def debit(self, symbol: str, amount: Decimal) -> None:
        raise RuntimeError("Live wallet provider not yet implemented")

    def credit(self, symbol: str, amount: Decimal) -> None:
        raise RuntimeError("Live wallet provider not yet implemented")


def get_wallet_provider(initial_balances: Dict[str, Decimal] | None = None) -> WalletProvider:
    """Return a wallet provider respecting runtime mode and latches."""
    runtime = get_runtime_mode()
    if runtime.mode == "live":
        return LiveWalletProvider()
    return PaperWalletProvider(initial_balances or {})

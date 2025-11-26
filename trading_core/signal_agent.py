"""Deterministic signal generation logic for breakout momentum strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from trading_core.config import StrategyConfig


@dataclass
class MarketSnapshot:
    symbol: str
    price: float
    rolling_high: float
    rolling_low: float
    recent_max: float
    atr: float
    atr_band: float
    volume_multiple: float
    volume_floor: float
    is_leader: bool = True
    go_to_cash: bool = False


@dataclass
class Intent:
    symbol: str
    action: str  # "BUY", "SELL", "CLOSE"
    size_hint: float
    reason: str


def generate_intents(config: StrategyConfig, snapshots: Dict[str, MarketSnapshot]) -> List[Intent]:
    """Return trade intents for each enabled asset based on snapshot data."""

    intents: List[Intent] = []
    for asset in config.assets:
        if not asset.enabled or asset.symbol not in snapshots:
            continue
        snap = snapshots[asset.symbol]
        weight = asset.leader_weight if snap.is_leader else asset.follower_weight
        size_hint = min(weight, config.risk.tradable_fraction)

        if snap.go_to_cash:
            intents.append(Intent(asset.symbol, "CLOSE", size_hint, "risk_off_flag"))
            continue

        entry_allowed = (
            snap.price >= snap.rolling_high
            and snap.atr <= snap.atr_band
            and snap.volume_multiple >= snap.volume_floor
        )
        exit_condition = (
            snap.price <= snap.recent_max * 0.98
            or snap.price <= snap.rolling_low
            or snap.atr > snap.atr_band * 1.3
        )

        if entry_allowed:
            intents.append(Intent(asset.symbol, "BUY", size_hint, "breakout_entry"))
        elif exit_condition:
            intents.append(Intent(asset.symbol, "SELL", size_hint, "exit_drawdown"))
        else:
            intents.append(Intent(asset.symbol, "HOLD", 0.0, "no_op"))
    return intents

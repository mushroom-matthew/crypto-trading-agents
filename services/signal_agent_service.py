"""Signal agent service bridging planner configs to trading_core logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from trading_core.config import StrategyConfig
from trading_core.signal_agent import MarketSnapshot, generate_intents


@dataclass
class SignalAgentService:
    strategy: StrategyConfig

    def generate(self, snapshots: Dict[str, MarketSnapshot]) -> List[dict]:
        intents = generate_intents(self.strategy, snapshots)
        return [intent.__dict__ for intent in intents]


def build_market_snapshots(
    indicator_summaries: Dict[str, dict],
    plan_map: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, MarketSnapshot]:
    snapshots: Dict[str, MarketSnapshot] = {}
    for symbol, summary in indicator_summaries.items():
        atr_band_mult = 1.5
        volume_floor = 1.0
        go_to_cash = summary.get("go_to_cash", False)
        if plan_map and symbol in plan_map:
            metadata = plan_map[symbol].get("metadata") or {}
            volatility = metadata.get("volatility", {})
            volume_meta = metadata.get("volume", {})
            atr_band_mult = volatility.get("atr_band_mult", atr_band_mult)
            volume_floor = volume_meta.get("volume_floor", volume_floor)
            go_to_cash = metadata.get("allow_go_to_cash", go_to_cash)
        snapshots[symbol] = MarketSnapshot(
            symbol=symbol,
            price=summary["price"],
            rolling_high=summary["rolling_high"],
            rolling_low=summary["rolling_low"],
            recent_max=summary["recent_max"],
            atr=summary["atr"],
            atr_band=summary["atr"] * atr_band_mult,
            volume_multiple=summary["volume_multiple"],
            volume_floor=volume_floor,
            is_leader=summary.get("is_leader", True),
            go_to_cash=go_to_cash,
        )
    return snapshots

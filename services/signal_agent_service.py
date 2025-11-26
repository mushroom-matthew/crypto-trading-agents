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


def build_market_snapshots(indicator_summaries: Dict[str, dict], plan: Dict[str, Any] | None = None) -> Dict[str, MarketSnapshot]:
    snapshots: Dict[str, MarketSnapshot] = {}
    for symbol, summary in indicator_summaries.items():
        atr_band = summary.get("atr_band", summary["atr"] * 1.5)
        volume_floor = summary.get("volume_floor", 1.0)
        if plan and plan.get("symbol") == symbol:
            atr_band = plan.get("atr_band", atr_band)
            volume_floor = plan.get("volume_floor", volume_floor)
        snapshots[symbol] = MarketSnapshot(
            symbol=symbol,
            price=summary["price"],
            rolling_high=summary["rolling_high"],
            rolling_low=summary["rolling_low"],
            recent_max=summary["recent_max"],
            atr=summary["atr"],
            atr_band=atr_band,
            volume_multiple=summary["volume_multiple"],
            volume_floor=volume_floor,
            is_leader=summary.get("is_leader", True),
            go_to_cash=summary.get("go_to_cash", False),
        )
    return snapshots

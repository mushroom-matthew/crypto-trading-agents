"""Adapters that allow reusing live execution logic inside the backtester."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from trading_core.config import DEFAULT_STRATEGY_CONFIG, StrategyConfig
from trading_core.signal_agent import MarketSnapshot, Intent
from trading_core.judge_agent import PortfolioState
from services.signal_agent_service import SignalAgentService
from services.judge_agent_service import JudgeAgentService


@dataclass
class StrategyParameters:
    atr_band_mult: float = 1.5
    volume_floor: float = 1.0
    go_to_cash: bool = False


@dataclass
class StrategyWrapperConfig:
    name: str = "execution-agent"
    strategy_config: StrategyConfig = field(default_factory=lambda: DEFAULT_STRATEGY_CONFIG)
    parameters: StrategyParameters = field(default_factory=StrategyParameters)
    per_symbol_parameters: Dict[str, StrategyParameters] = field(default_factory=dict)

    def parameters_for(self, symbol: str) -> StrategyParameters:
        return self.per_symbol_parameters.get(symbol, self.parameters)


class ExecutionAgentStrategy:
    """Wraps the deterministic trading_core signal/judge logic for simulation."""

    def __init__(self, config: StrategyWrapperConfig) -> None:
        self.config = config
        self.signal_service = SignalAgentService(config.strategy_config)
        self.judge_service = JudgeAgentService(config.strategy_config)

    def decide(
        self,
        feature_vector: Dict[str, Any],
        portfolio_state: PortfolioState,
    ) -> List[Intent]:
        params = self.config.parameters_for(feature_vector["symbol"])
        snapshot = MarketSnapshot(
            symbol=feature_vector["symbol"],
            price=feature_vector["price"],
            rolling_high=feature_vector["rolling_high"],
            rolling_low=feature_vector["rolling_low"],
            recent_max=feature_vector["recent_max"],
            atr=feature_vector["atr"],
            atr_band=feature_vector["atr"] * params.atr_band_mult,
            volume_multiple=feature_vector["volume_multiple"],
            volume_floor=params.volume_floor,
            go_to_cash=params.go_to_cash,
        )
        intents = self.signal_service.generate({snapshot.symbol: snapshot})
        intent_objs = [Intent(**intent) for intent in intents]
        approved = self.judge_service.evaluate(portfolio_state, intent_objs)
        approved_intents: List[Intent] = []
        for judgement in approved:
            if not judgement.get("approved"):
                continue
            intent_obj = judgement.get("intent")
            if isinstance(intent_obj, Intent):
                approved_intents.append(intent_obj)
            else:
                approved_intents.append(Intent(**intent_obj))
        return approved_intents

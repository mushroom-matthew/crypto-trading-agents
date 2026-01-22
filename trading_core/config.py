"""Shared configuration objects for LLM-planned breakout strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AssetConfig:
    symbol: str
    enabled: bool = True
    leader_weight: float = 0.5
    follower_weight: float = 0.5


@dataclass
class RiskSettings:
    max_per_trade_risk_fraction: float = 0.02
    max_portfolio_drawdown_before_kill: float = 0.45
    max_leverage: float = 1.0
    tradable_fraction: float = 0.4


@dataclass
class PlannerSettings:
    max_calls_per_day: int = 1
    llm_model: str = "gpt-4o-mini"
    min_lookback_bars: int = 50
    max_lookback_bars: int = 300


@dataclass
class StrategyConfig:
    name: str
    assets: List[AssetConfig]
    risk: RiskSettings
    planner: PlannerSettings
    metadata: Dict[str, str] = field(default_factory=dict)


DEFAULT_STRATEGY_CONFIG = StrategyConfig(
    name="LLM-Planned Breakout Momentum",
    assets=[
        AssetConfig(symbol="BTC-USD", leader_weight=0.7, follower_weight=0.3),
        AssetConfig(symbol="ETH-USD", leader_weight=0.3, follower_weight=0.7),
    ],
    risk=RiskSettings(),
    planner=PlannerSettings(),
    metadata={
        "mode": "spot",
        "notes": "Go-to-cash allowed, spot only, leverage disabled",
    },
)

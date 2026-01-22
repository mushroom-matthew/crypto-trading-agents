"""Portfolio-level risk checks for strategy intents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from trading_core.config import StrategyConfig
from trading_core.signal_agent import Intent


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, float]
    equity: float
    max_equity: float

    @property
    def drawdown_pct(self) -> float:
        if self.max_equity <= 0:
            return 0.0
        return max(0.0, (self.max_equity - self.equity) / self.max_equity)


@dataclass
class Judgement:
    intent: Intent
    approved: bool
    reason: str


def evaluate_intents(
    config: StrategyConfig,
    portfolio: PortfolioState,
    intents: Iterable[Intent],
) -> List[Judgement]:
    """Approve or reject intents given risk settings and drawdowns."""

    judgements: List[Judgement] = []
    max_drawdown = config.risk.max_portfolio_drawdown_before_kill
    tradable_fraction = config.risk.tradable_fraction
    drawdown_exceeded = portfolio.drawdown_pct >= max_drawdown

    for intent in intents:
        if drawdown_exceeded and intent.action == "BUY":
            judgements.append(Judgement(intent, False, "drawdown_halt"))
            continue
        if intent.action == "BUY" and intent.size_hint > tradable_fraction:
            judgements.append(Judgement(intent, False, "size_exceeds_tradable_fraction"))
            continue
        if intent.action == "SELL" and portfolio.positions.get(intent.symbol, 0.0) <= 0:
            judgements.append(Judgement(intent, False, "no_position_to_sell"))
            continue
        judgements.append(Judgement(intent, True, "approved"))
    return judgements

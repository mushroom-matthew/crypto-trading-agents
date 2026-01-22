"""Judge agent service enforcing portfolio-level risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from trading_core.config import StrategyConfig
from trading_core.judge_agent import PortfolioState, evaluate_intents
from trading_core.signal_agent import Intent


@dataclass
class JudgeAgentService:
    strategy: StrategyConfig

    def evaluate(self, portfolio: PortfolioState, intents: List[Intent]) -> List[dict]:
        judgements = evaluate_intents(self.strategy, portfolio, intents)
        return [judgement.__dict__ for judgement in judgements]

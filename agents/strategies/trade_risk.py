"""Unified trade-level risk checks shared across trigger categories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from schemas.llm_strategist import IndicatorSnapshot, PortfolioState, TriggerCondition

from .risk_engine import RiskEngine


@dataclass
class RiskCheckResult:
    """Outcome of a risk gate evaluation."""

    allowed: bool
    quantity: float
    reason: str | None = None


class TradeRiskEvaluator:
    """Central guard that applies RiskEngine sizing to any candidate order."""

    def __init__(self, engine: RiskEngine) -> None:
        self.engine = engine

    def evaluate(
        self,
        trigger: TriggerCondition,
        action: Literal["entry", "flatten"],
        price: float,
        portfolio: PortfolioState,
        indicator: IndicatorSnapshot | None = None,
        stop_distance: float | None = None,
    ) -> RiskCheckResult:
        """Return whether the trade should proceed under configured limits.

        Emergency exits and flatten orders always pass (they reduce risk), but we still
        normalize the quantity to ensure downstream consumers receive a sensible value.
        """

        # Flattening or protective exits always reduce exposure; bypass caps.
        if action == "flatten" or trigger.category == "emergency_exit":
            return RiskCheckResult(allowed=True, quantity=0.0)

        if indicator is None:
            return RiskCheckResult(allowed=False, quantity=0.0, reason="missing_indicator")

        quantity = self.engine.size_position(trigger.symbol, price, portfolio, indicator, stop_distance=stop_distance)
        if quantity <= 0:
            reason = self.engine.last_block_reason or "sizing_zero"
            return RiskCheckResult(allowed=False, quantity=0.0, reason=reason)
        return RiskCheckResult(allowed=True, quantity=quantity)


__all__ = ["TradeRiskEvaluator", "RiskCheckResult"]

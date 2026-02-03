"""Unified trade-level risk checks shared across trigger categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Literal, Tuple

from schemas.llm_strategist import IndicatorSnapshot, PortfolioState, TriggerCondition
from schemas.experiment_spec import ExperimentSpec
from schemas.strategy_run import LearningBookSettings

from .risk_engine import RiskEngine


@dataclass
class RiskCheckResult:
    """Outcome of a risk gate evaluation."""

    allowed: bool
    quantity: float
    reason: str | None = None


class TradeRiskEvaluator:
    """Central guard that applies RiskEngine sizing to any candidate order."""

    def __init__(
        self,
        engine: RiskEngine,
        learning_settings: LearningBookSettings | None = None,
        experiment_spec: ExperimentSpec | None = None,
    ) -> None:
        self.engine = engine
        self.learning_settings = learning_settings or LearningBookSettings()
        self.experiment_spec = experiment_spec
        # Learning book daily tracking
        self._learning_trades_today: int = 0
        self._learning_daily_risk_used: float = 0.0
        self._learning_day: date | None = None

    def reset_learning_daily(self) -> None:
        """Reset learning book daily counters (call at day boundaries)."""
        self._learning_trades_today = 0
        self._learning_daily_risk_used = 0.0

    def _check_learning_day(self, current_day: date) -> None:
        """Auto-reset daily counters when the day rolls over."""
        if self._learning_day != current_day:
            self._learning_day = current_day
            self.reset_learning_daily()

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

        # --- Learning book gate ---
        if trigger.learning_book:
            return self._evaluate_learning(trigger, price, portfolio, indicator)

        if indicator is None:
            return RiskCheckResult(allowed=False, quantity=0.0, reason="missing_indicator")

        as_of = getattr(indicator, "as_of", None)
        hour = as_of.hour if as_of is not None else None
        symbol_archetype = None
        if trigger.symbol:
            symbol_lower = trigger.symbol.lower()
            symbol_archetype = symbol_lower.split("-")[0]
        archetype = symbol_archetype or trigger.category
        side = trigger.direction if trigger.direction in {"long", "short"} else None
        quantity = self.engine.size_position(
            trigger.symbol,
            price,
            portfolio,
            indicator,
            stop_distance=stop_distance,
            archetype=archetype,
            hour=hour,
            side=side,
        )
        if quantity <= 0:
            reason = self.engine.last_block_reason or "sizing_zero"
            return RiskCheckResult(allowed=False, quantity=0.0, reason=reason)
        return RiskCheckResult(allowed=True, quantity=quantity)

    def _evaluate_learning(
        self,
        trigger: TriggerCondition,
        price: float,
        portfolio: PortfolioState,
        indicator: IndicatorSnapshot | None,
    ) -> RiskCheckResult:
        """Apply learning book-specific risk checks and sizing."""
        ls = self.learning_settings

        # Check learning book is enabled
        if not ls.enabled:
            return RiskCheckResult(allowed=False, quantity=0.0, reason="learning_disabled")

        # Check short restriction
        if trigger.direction == "short" and not ls.allow_short:
            return RiskCheckResult(allowed=False, quantity=0.0, reason="learning_short_disabled")

        # Experiment spec filtering (when experiment is active + trigger has experiment_id)
        if self.experiment_spec and self.experiment_spec.status == "running" and trigger.experiment_id:
            exp = self.experiment_spec
            # Symbol filter
            if exp.exposure.target_symbols and trigger.symbol not in exp.exposure.target_symbols:
                return RiskCheckResult(allowed=False, quantity=0.0, reason="experiment_symbol_filter")
            # Category filter
            if exp.exposure.trigger_categories and trigger.category not in exp.exposure.trigger_categories:
                return RiskCheckResult(allowed=False, quantity=0.0, reason="experiment_category_filter")

        # Auto-reset on day boundary
        as_of = getattr(indicator, "as_of", None) if indicator else None
        if as_of is not None:
            self._check_learning_day(as_of.date())
        elif portfolio.timestamp:
            self._check_learning_day(portfolio.timestamp.date())

        # Daily trade cap
        if self._learning_trades_today >= ls.max_trades_per_day:
            return RiskCheckResult(allowed=False, quantity=0.0, reason="learning_daily_trade_cap")

        # Daily risk budget
        equity = portfolio.equity
        if equity <= 0:
            return RiskCheckResult(allowed=False, quantity=0.0, reason="learning_no_equity")

        budget_remaining = (ls.daily_risk_budget_pct / 100.0) * equity - self._learning_daily_risk_used
        if budget_remaining <= 0:
            return RiskCheckResult(allowed=False, quantity=0.0, reason="learning_daily_risk_exhausted")

        # Position risk cap
        max_risk_per_trade = (ls.max_position_risk_pct / 100.0) * equity

        # Experiment notional cap
        experiment_notional_cap = float("inf")
        if self.experiment_spec and self.experiment_spec.status == "running" and trigger.experiment_id:
            experiment_notional_cap = self.experiment_spec.exposure.max_notional_usd

        # Sizing: notional mode
        if ls.sizing_mode == "notional":
            notional = min(ls.notional_usd, budget_remaining, max_risk_per_trade, experiment_notional_cap)
            if price <= 0:
                return RiskCheckResult(allowed=False, quantity=0.0, reason="learning_zero_price")
            quantity = notional / price
        else:
            # Default to notional sizing for other modes (extendable later)
            notional = min(ls.notional_usd, budget_remaining, max_risk_per_trade, experiment_notional_cap)
            if price <= 0:
                return RiskCheckResult(allowed=False, quantity=0.0, reason="learning_zero_price")
            quantity = notional / price

        if quantity <= 0:
            return RiskCheckResult(allowed=False, quantity=0.0, reason="learning_sizing_zero")

        # Portfolio exposure check
        learning_exposure = self._current_learning_exposure(portfolio)
        new_notional = quantity * price
        max_exposure = (ls.max_portfolio_exposure_pct / 100.0) * equity
        if learning_exposure + new_notional > max_exposure:
            return RiskCheckResult(allowed=False, quantity=0.0, reason="learning_exposure_cap")

        # Accept: update counters
        self._learning_trades_today += 1
        self._learning_daily_risk_used += new_notional

        return RiskCheckResult(allowed=True, quantity=quantity)

    def _current_learning_exposure(self, portfolio: PortfolioState) -> float:
        """Placeholder: returns 0.0. In production, sum learning-book positions only."""
        return 0.0


__all__ = ["TradeRiskEvaluator", "RiskCheckResult"]

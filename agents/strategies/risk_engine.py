"""Risk-constraint enforcement and position sizing."""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Mapping

from schemas.llm_strategist import IndicatorSnapshot, PortfolioState, PositionSizingRule, RiskConstraint

logger = logging.getLogger(__name__)

@dataclass
class RiskProfile:
    """Composition of judge-directed risk multipliers."""

    global_multiplier: float = 1.0
    symbol_multipliers: Dict[str, float] = field(default_factory=dict)

    def multiplier_for(self, symbol: str | None = None) -> float:
        base = max(0.0, self.global_multiplier)
        if symbol and symbol in self.symbol_multipliers:
            return base * max(0.0, self.symbol_multipliers[symbol])
        return base


class RiskEngine:
    """Applies plan-level RiskConstraint values to candidate orders."""

    def __init__(
        self,
        constraints: RiskConstraint,
        sizing_rules: Mapping[str, PositionSizingRule],
        daily_anchor_equity: float | None = None,
        risk_profile: RiskProfile | None = None,
    ) -> None:
        self.constraints = constraints
        self.sizing_rules: Dict[str, PositionSizingRule] = dict(sizing_rules)
        self.daily_anchor_equity = daily_anchor_equity
        self.risk_profile = risk_profile or RiskProfile()
        self.last_block_reason: str | None = None
        # Snapshot of last sizing decision for telemetry (allocated vs actual risk).
        self.last_risk_snapshot: Dict[str, float | None] = {}

    def _fraction(self, pct: float | None) -> float:
        return (pct or 0.0) / 100.0

    def _profile_multiplier(self, symbol: str | None = None) -> float:
        multiplier = self.risk_profile.multiplier_for(symbol)
        return multiplier if multiplier > 0 else 0.0

    def _scaled_fraction(self, pct: float | None, symbol: str | None = None) -> float:
        return self._fraction(pct) * self._profile_multiplier(symbol)

    def _within_daily_loss(self, equity: float) -> bool:
        if self.daily_anchor_equity is None:
            self.daily_anchor_equity = equity
            return True
        anchor = self.daily_anchor_equity
        if anchor <= 0:
            return True
        loss = (anchor - equity) / anchor
        return loss <= self._scaled_fraction(self.constraints.max_daily_loss_pct, None)

    def _symbol_notional(self, symbol: str, price: float, portfolio: PortfolioState) -> float:
        position = portfolio.positions.get(symbol, 0.0)
        return abs(position) * price

    def _available_symbol_capacity(self, symbol: str, price: float, portfolio: PortfolioState) -> float:
        max_symbol = portfolio.equity * self._scaled_fraction(self.constraints.max_symbol_exposure_pct, symbol)
        if max_symbol <= 0:
            return float("inf")
        return max(0.0, max_symbol - self._symbol_notional(symbol, price, portfolio))

    def _portfolio_exposure(self, portfolio: PortfolioState) -> float:
        return max(0.0, portfolio.equity - portfolio.cash)

    def _available_portfolio_capacity(self, portfolio: PortfolioState) -> float:
        max_portfolio = portfolio.equity * self._scaled_fraction(self.constraints.max_portfolio_exposure_pct, None)
        if max_portfolio <= 0:
            return float("inf")
        return max(0.0, max_portfolio - self._portfolio_exposure(portfolio))

    def _position_risk_cap(self, portfolio: PortfolioState, symbol: str) -> float:
        return portfolio.equity * self._scaled_fraction(self.constraints.max_position_risk_pct, symbol)

    def _rule_for(self, symbol: str) -> PositionSizingRule:
        if symbol not in self.sizing_rules:
            self.sizing_rules[symbol] = PositionSizingRule(
                symbol=symbol,
                sizing_mode="fixed_fraction",
                target_risk_pct=self.constraints.max_position_risk_pct,
            )
        return self.sizing_rules[symbol]

    def _notional_from_rule(
        self,
        rule: PositionSizingRule,
        portfolio: PortfolioState,
        indicator: IndicatorSnapshot,
    ) -> float:
        equity = max(portfolio.equity, 1e-9)
        if rule.sizing_mode == "fixed_fraction":
            pct = rule.target_risk_pct or self.constraints.max_position_risk_pct
            return equity * self._fraction(pct)
        if rule.sizing_mode == "notional":
            return max(0.0, rule.notional or 0.0)
        if rule.sizing_mode == "vol_target":
            target = rule.vol_target_annual or 0.0
            realized = indicator.realized_vol_short or indicator.realized_vol_medium or 0.0
            if target <= 0 or realized <= 0:
                return 0.0
            daily_target = target / math.sqrt(365.0)
            scale = daily_target / realized
            scale = max(scale, 0.0)
            if scale > 1.0:
                logger.debug("vol_target uncapped scale used", extra={"scale": scale, "symbol": indicator.symbol})
            return equity * scale
        raise ValueError(f"unsupported sizing mode {rule.sizing_mode}")

    def size_position(
        self,
        symbol: str,
        price: float,
        portfolio: PortfolioState,
        indicator: IndicatorSnapshot,
        stop_distance: float | None = None,
    ) -> float:
        self.last_block_reason = None
        if price <= 0:
            self.last_block_reason = "invalid_price"
            return 0.0
        if not self._within_daily_loss(portfolio.equity):
            self.last_block_reason = "max_daily_loss_pct"
            return 0.0
        rule = self._rule_for(symbol)
        multiplier = self._profile_multiplier(symbol)
        desired_notional = self._notional_from_rule(rule, portfolio, indicator) * multiplier
        if desired_notional <= 0:
            self.last_block_reason = "sizing_zero"
            return 0.0

        # Derive a stop-distance proxy (true stop preferred; fallback to ATR).
        if stop_distance is None and indicator.atr_14 is not None:
            stop_distance = float(indicator.atr_14)

        # Compute per-trade risk cap in notional terms; if a stop exists, convert allowed loss to notional.
        risk_cap_abs = portfolio.equity * self._scaled_fraction(self.constraints.max_position_risk_pct, symbol)
        if stop_distance and price > 0:
            qty_cap = risk_cap_abs / max(stop_distance, 1e-9)
            position_cap = qty_cap * price
        else:
            position_cap = risk_cap_abs

        caps = {
            "max_position_risk_pct": position_cap,
            "max_symbol_exposure_pct": self._available_symbol_capacity(symbol, price, portfolio),
            "max_portfolio_exposure_pct": self._available_portfolio_capacity(portfolio),
        }
        for name, value in caps.items():
            if value <= 0:
                self.last_block_reason = name
                return 0.0
        positive_caps = {name: value for name, value in caps.items() if value > 0}
        if not positive_caps:
            self.last_block_reason = "insufficient_capacity"
            return 0.0
        limiting_name, limiting_value = min(positive_caps.items(), key=lambda item: item[1])
        if limiting_value <= 0:
            self.last_block_reason = limiting_name
            return 0.0
        final = min(desired_notional, limiting_value)
        if final <= 0:
            self.last_block_reason = limiting_name
            return 0.0
        quantity = max(0.0, final / price)
        actual_risk = None
        if stop_distance and price > 0 and quantity > 0:
            actual_risk = quantity * stop_distance
        self.last_risk_snapshot = {
            "equity": portfolio.equity,
            "stop_distance": stop_distance,
            "risk_cap_abs": risk_cap_abs,
            "risk_cap_notional": position_cap,
            "desired_notional": desired_notional,
            "final_notional": final,
            "allocated_risk_pct": self.constraints.max_position_risk_pct,
            "allocated_risk_abs": risk_cap_abs,
            "actual_risk_abs": actual_risk,
        }
        if quantity <= 0:
            self.last_block_reason = limiting_name
        else:
            self.last_block_reason = None
        return quantity

"""Tests for risk-at-stop sizing (stop-aware caps)."""

from datetime import datetime
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.strategies.risk_engine import RiskEngine, RiskProfile
from schemas.llm_strategist import IndicatorSnapshot, PortfolioState, PositionSizingRule, RiskConstraint


def _portfolio(equity: float = 1000.0) -> PortfolioState:
    return PortfolioState(
        timestamp=datetime(2021, 1, 1),
        equity=equity,
        cash=equity,
        positions={},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )


def _indicator(price: float = 100.0, atr: float | None = None) -> IndicatorSnapshot:
    return IndicatorSnapshot(symbol="BTC", timeframe="1h", as_of=datetime(2021, 1, 1), close=price, atr_14=atr)


def test_stop_distance_expands_notional_to_match_risk_cap() -> None:
    constraints = RiskConstraint(
        max_position_risk_pct=1.0,
        max_symbol_exposure_pct=100.0,
        max_portfolio_exposure_pct=100.0,
        max_daily_loss_pct=100.0,
        max_daily_risk_budget_pct=None,
    )
    # Sizing rule desires 50% notional, but risk cap is 1% of equity. With stop=5, allowed notional is expanded.
    rule = PositionSizingRule(symbol="BTC", sizing_mode="fixed_fraction", target_risk_pct=50.0)
    engine = RiskEngine(constraints, {"BTC": rule}, daily_anchor_equity=1000.0, risk_profile=RiskProfile())
    qty = engine.size_position("BTC", price=100.0, portfolio=_portfolio(1000.0), indicator=_indicator(100.0, atr=5.0))
    # Risk cap abs = 10; stop=5 => qty_cap=2 => notional cap=200; desired notional=500 -> final should be capped near 200/price=2.
    assert qty == pytest.approx(2.0, rel=1e-6)
    assert engine.last_risk_snapshot.get("actual_risk_abs") == pytest.approx(10.0, rel=1e-6)
    assert engine.last_risk_snapshot.get("allocated_risk_abs") == pytest.approx(10.0, rel=1e-6)


def test_no_stop_uses_notional_cap() -> None:
    constraints = RiskConstraint(
        max_position_risk_pct=1.0,
        max_symbol_exposure_pct=100.0,
        max_portfolio_exposure_pct=100.0,
        max_daily_loss_pct=100.0,
        max_daily_risk_budget_pct=None,
    )
    rule = PositionSizingRule(symbol="BTC", sizing_mode="fixed_fraction", target_risk_pct=50.0)
    engine = RiskEngine(constraints, {"BTC": rule}, daily_anchor_equity=1000.0, risk_profile=RiskProfile())
    qty = engine.size_position("BTC", price=100.0, portfolio=_portfolio(1000.0), indicator=_indicator(100.0, atr=None))
    # Without stop, risk cap = 1% equity => notional cap=10 -> qty=0.1
    assert qty == pytest.approx(0.1, rel=1e-6)


def test_tighter_stop_allows_larger_size() -> None:
    constraints = RiskConstraint(
        max_position_risk_pct=2.0,
        max_symbol_exposure_pct=100.0,
        max_portfolio_exposure_pct=100.0,
        max_daily_loss_pct=100.0,
        max_daily_risk_budget_pct=None,
    )
    rule = PositionSizingRule(symbol="BTC", sizing_mode="fixed_fraction", target_risk_pct=50.0)
    engine = RiskEngine(constraints, {"BTC": rule}, daily_anchor_equity=1000.0, risk_profile=RiskProfile())
    qty_tight = engine.size_position("BTC", price=100.0, portfolio=_portfolio(1000.0), indicator=_indicator(100.0, atr=1.0))
    qty_wide = engine.size_position("BTC", price=100.0, portfolio=_portfolio(1000.0), indicator=_indicator(100.0, atr=5.0))
    # Tighter stop permits larger size; ratio reflects cap interplay (desired notional vs risk cap).
    assert qty_tight > qty_wide * 1.1
    # Cap enforcement: actual risk should not exceed 2% of equity.
    assert engine.last_risk_snapshot.get("actual_risk_abs") <= 20.0 + 1e-6

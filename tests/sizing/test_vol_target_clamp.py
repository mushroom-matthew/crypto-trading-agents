"""Tests that vol-target sizing is not hard-capped at 1x equity (caps should come from exposure limits)."""

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


def _indicator(price: float, realized_vol: float) -> IndicatorSnapshot:
    return IndicatorSnapshot(symbol="BTC", timeframe="1h", as_of=datetime(2021, 1, 1), close=price, realized_vol_short=realized_vol)


def test_vol_target_can_exceed_one_x_equity_before_caps() -> None:
    constraints = RiskConstraint(
        max_position_risk_pct=1000.0,  # effectively off for this test
        max_symbol_exposure_pct=200.0,
        max_portfolio_exposure_pct=200.0,
        max_daily_loss_pct=100.0,
        max_daily_risk_budget_pct=None,
    )
    # Target vol very high vs realized vol -> scale > 1
    rule = PositionSizingRule(symbol="BTC", sizing_mode="vol_target", target_risk_pct=None, vol_target_annual=2.0)
    engine = RiskEngine(constraints, {"BTC": rule}, daily_anchor_equity=1000.0, risk_profile=RiskProfile())
    qty = engine.size_position("BTC", price=100.0, portfolio=_portfolio(1000.0), indicator=_indicator(100.0, realized_vol=0.01))
    notional = qty * 100.0
    assert notional > 1000.0  # exceeds 1x equity before exposure caps intervene
    # Exposure caps still enforce a limit
    assert notional <= 2000.0  # due to 200% exposure cap

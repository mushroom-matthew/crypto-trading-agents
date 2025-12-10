"""Ensure exit/emergency_exit cannot increase exposure and only flatten when position exists."""

from datetime import datetime
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.strategies.trigger_engine import Bar, TriggerEngine, Order
from agents.strategies.risk_engine import RiskEngine, RiskProfile
from agents.strategies.trade_risk import TradeRiskEvaluator
from schemas.llm_strategist import (
    AssetState,
    IndicatorSnapshot,
    PortfolioState,
    PositionSizingRule,
    RiskConstraint,
    StrategyPlan,
    TriggerCondition,
)


def _plan(direction: str) -> StrategyPlan:
    now = datetime(2024, 1, 1)
    return StrategyPlan(
        generated_at=now,
        valid_until=now,
        global_view=None,
        regime="range",
        triggers=[
            TriggerCondition(
                id="exit_trigger",
                symbol="BTC",
                direction=direction,
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="timeframe=='1h'",
                category="emergency_exit",
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=1.0,
            max_symbol_exposure_pct=100.0,
            max_portfolio_exposure_pct=100.0,
            max_daily_loss_pct=100.0,
            max_daily_risk_budget_pct=None,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=10,
        max_triggers_per_symbol_per_day=None,
        trigger_budgets={},
    )


def _indicator() -> IndicatorSnapshot:
    return IndicatorSnapshot(symbol="BTC", timeframe="1h", as_of=datetime(2024, 1, 1), close=100.0)


def _portfolio(position: float) -> PortfolioState:
    return PortfolioState(
        timestamp=datetime(2024, 1, 1),
        equity=1000.0,
        cash=1000.0,
        positions={"BTC": position},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )


def _engine(plan: StrategyPlan) -> TriggerEngine:
    risk = RiskEngine(plan.risk_constraints, {rule.symbol: rule for rule in plan.sizing_rules}, daily_anchor_equity=1000.0, risk_profile=RiskProfile())
    return TriggerEngine(plan, risk, trade_risk=TradeRiskEvaluator(risk))


def test_exit_does_nothing_when_flat() -> None:
    plan = _plan("exit")
    engine = _engine(plan)
    bar = Bar(symbol="BTC", timeframe="1h", timestamp=datetime(2024, 1, 1), open=100, high=100, low=100, close=100, volume=0)
    orders, blocks = engine.on_bar(bar, _indicator(), _portfolio(0.0), None)
    assert not orders
    assert not blocks


def test_exit_flattens_when_position_exists() -> None:
    plan = _plan("exit")
    engine = _engine(plan)
    bar = Bar(symbol="BTC", timeframe="1h", timestamp=datetime(2024, 1, 1), open=100, high=100, low=100, close=100, volume=0)
    orders, blocks = engine.on_bar(bar, _indicator(), _portfolio(1.0), None)
    assert len(orders) == 1
    assert isinstance(orders[0], Order)
    assert orders[0].side == "sell"
    assert orders[0].quantity == pytest.approx(1.0)

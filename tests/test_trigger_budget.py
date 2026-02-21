from __future__ import annotations

from datetime import datetime, timedelta, timezone

from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition
from trading_core.trigger_budget import enforce_trigger_budget


def _plan_with_triggers(symbol: str) -> list[TriggerCondition]:
    combos = [
        ("trend_continuation", "long", "1h"),
        ("mean_reversion", "short", "1h"),
        ("volatility_breakout", "long", "4h"),
        ("reversal", "short", "4h"),
    ]
    triggers: list[TriggerCondition] = []
    for idx, (category, direction, timeframe) in enumerate(combos):
        triggers.append(
            TriggerCondition(
                id=f"{symbol}_trigger_{idx}",
                symbol=symbol,
                direction=direction,
                timeframe=timeframe,
                entry_rule=f"close > {idx}",
                exit_rule="close < 0",
                category=category,
                confidence_grade="A" if idx % 2 == 0 else "B",
                stop_loss_pct=2.0,
            )
        )
    return triggers


def _strategy_plan() -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    triggers = _plan_with_triggers("BTC-USD") + _plan_with_triggers("ETH-USD")
    plan = StrategyPlan(
        plan_id="plan_test",
        run_id="run",
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=triggers,
        risk_constraints=RiskConstraint(
            max_position_risk_pct=1.0,
            max_symbol_exposure_pct=25.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=10,
    )
    plan.allowed_symbols = ["BTC-USD", "ETH-USD"]
    plan.allowed_directions = ["long", "short"]
    plan.allowed_trigger_categories = ["trend_continuation", "reversal"]
    plan.max_triggers_per_symbol_per_day = 2
    plan.trigger_budgets = {"BTC-USD": 3}
    return plan


def test_enforce_trigger_budget_limits_per_symbol():
    plan = _strategy_plan()
    trimmed, stats = enforce_trigger_budget(plan, default_cap=2)
    btc_triggers = [t for t in trimmed.triggers if t.symbol == "BTC-USD"]
    eth_triggers = [t for t in trimmed.triggers if t.symbol == "ETH-USD"]
    assert len(btc_triggers) == 2  # capped by plan-level max
    assert len(eth_triggers) == 2  # falls back to plan cap
    assert stats["BTC-USD"] == 2
    assert stats["ETH-USD"] == 2

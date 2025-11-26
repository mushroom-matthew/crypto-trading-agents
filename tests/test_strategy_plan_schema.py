from datetime import datetime, timezone

import pytest

from schemas.strategy_plan import (
    StrategyPlan,
    LookbackConfig,
    RiskManagementConfig,
    EntryRule,
    ExitRule,
    Condition,
    PositionSizing,
    StopLossConfig,
    TakeProfitConfig,
    ReplanTriggers,
    LLMMetadata,
)


def _build_plan() -> StrategyPlan:
    return StrategyPlan(
        plan_id="plan-123",
        created_at=datetime.now(timezone.utc),
        symbol="BTC-USD",
        timeframe="1h",
        lookback=LookbackConfig(preferred_bars=300, min_bars=100, max_bars=500),
        risk_management=RiskManagementConfig(
            max_position_pct=0.25,
            max_daily_loss_pct=0.05,
            max_total_drawdown_pct=0.3,
            per_trade_risk_pct=0.01,
        ),
        entry_rules=[
            EntryRule(
                id="trend_follow_long",
                direction="long",
                conditions=[
                    Condition(indicator="ema_fast", operator=">", value="ema_slow"),
                    Condition(indicator="rsi", operator="<", value=60),
                ],
                position_sizing=PositionSizing(type="volatility_target", target_volatility=0.02, max_leverage=2.0),
            )
        ],
        exit_rules=[
            ExitRule(
                id="trend_exit",
                applies_to_entry_id="trend_follow_long",
                conditions=[Condition(indicator="ema_fast", operator="<", value="ema_slow")],
                stop_loss=StopLossConfig(type="atr_multiple", atr_period=14, multiple=2.0),
                take_profit=TakeProfitConfig(type="rr_multiple", reward_risk=3.0),
            )
        ],
        replan_triggers=ReplanTriggers(),
        llm_metadata=LLMMetadata(model_name="gpt-4.1", prompt_version="v1", notes="test"),
    )


def test_strategy_plan_validation() -> None:
    plan = _build_plan()
    assert plan.symbol == "BTC-USD"
    assert plan.entry_rules[0].position_sizing is not None


def test_lookback_clamping_respects_available_bars() -> None:
    plan = _build_plan()
    # when history shorter than preferred
    assert plan.clamp_lookback(50) == 50  # min between min_bars (100) and available (50)
    assert plan.clamp_lookback(400) == 300
    assert plan.clamp_lookback(800) == 300


def test_preferred_outside_bounds_raises() -> None:
    with pytest.raises(ValueError):
        LookbackConfig(preferred_bars=50, min_bars=100, max_bars=200)

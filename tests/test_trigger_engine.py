from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agents.strategies.risk_engine import RiskEngine
from agents.strategies.trade_risk import TradeRiskEvaluator
from agents.strategies.trigger_engine import Bar, TriggerEngine
from schemas.judge_feedback import JudgeConstraints
from schemas.llm_strategist import IndicatorSnapshot, PortfolioState, RiskConstraint, StrategyPlan, TriggerCondition
from schemas.strategy_run import LearningBookSettings


def _portfolio() -> PortfolioState:
    return PortfolioState(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        equity=100000.0,
        cash=100000.0,
        positions={},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )


def _portfolio_with_position() -> PortfolioState:
    state = _portfolio()
    state.positions = {"BTC-USD": 1.0}
    return state


def _indicator() -> IndicatorSnapshot:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return IndicatorSnapshot(symbol="BTC-USD", timeframe="1h", as_of=ts, close=50000.0, atr_14=500.0, sma_medium=49000.0)


def _plan(trigger: TriggerCondition) -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return StrategyPlan(
        generated_at=now,
        valid_until=now,
        global_view="test",
        regime="range",
        triggers=[trigger],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=0.0,
            max_symbol_exposure_pct=5.0,
            max_portfolio_exposure_pct=50.0,
            max_daily_loss_pct=3.0,
        ),
    )


def _plan_with_triggers(triggers: list[TriggerCondition]) -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return StrategyPlan(
        generated_at=now,
        valid_until=now,
        global_view="test",
        regime="range",
        triggers=triggers,
        risk_constraints=RiskConstraint(
            max_position_risk_pct=2.0,
            max_symbol_exposure_pct=100.0,
            max_portfolio_exposure_pct=100.0,
            max_daily_loss_pct=10.0,
        ),
    )


def test_trigger_engine_records_block_when_risk_denies_entry():
    trigger = TriggerCondition(
        id="btc_long",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
    )
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, max_triggers_per_symbol_per_bar=2)
    bar = Bar(symbol="BTC-USD", timeframe="1h", timestamp=_portfolio().timestamp, open=50000.0, high=50050.0, low=49950.0, close=50000.0, volume=1.0)

    orders, blocks = engine.on_bar(bar, _indicator(), _portfolio())
    assert not orders
    assert blocks
    assert blocks[0]["reason"] in {"max_position_risk_pct", "sizing_zero"}


def test_emergency_exit_trigger_bypasses_risk_checks():
    trigger = TriggerCondition(
        id="btc_exit",
        symbol="BTC-USD",
        direction="flat",
        timeframe="1h",
        entry_rule="false",
        exit_rule="timeframe=='1h'",
        category="emergency_exit",
    )
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, max_triggers_per_symbol_per_bar=2)
    bar = Bar(symbol="BTC-USD", timeframe="1h", timestamp=_portfolio().timestamp, open=50000.0, high=50050.0, low=49950.0, close=50000.0, volume=1.0)
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert orders  # flatten order should be produced even though limits are zero
    assert not blocks
    assert orders[0].side == "sell"


def test_emergency_exit_vetoes_same_bar_entry():
    trigger = TriggerCondition(
        id="btc_emergency_exit",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        category="emergency_exit",
    )
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    engine.record_fill("BTC-USD", is_entry=True, timestamp=bar.timestamp)
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert not orders
    assert blocks
    assert blocks[0]["reason"] == "emergency_exit_veto_same_bar"
    assert blocks[0]["cooldown_recommendation_bars"] == max(1, engine.trade_cooldown_bars, engine.min_hold_bars)


def test_emergency_exit_vetoes_min_hold_on_next_bar():
    trigger = TriggerCondition(
        id="btc_emergency_exit",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        category="emergency_exit",
    )
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=3, trade_cooldown_bars=0)
    entry_time = _portfolio().timestamp
    entry_bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=entry_time,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    engine.record_fill("BTC-USD", is_entry=True, timestamp=entry_bar.timestamp)

    next_bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=entry_time + timedelta(hours=1),
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(next_bar, _indicator(), portfolio)
    assert not orders
    assert blocks
    assert blocks[0]["reason"] == "emergency_exit_veto_min_hold"
    assert blocks[0]["cooldown_recommendation_bars"] == max(1, engine.trade_cooldown_bars, engine.min_hold_bars)


def test_emergency_exit_min_hold_allows_on_threshold_bar():
    trigger = TriggerCondition(
        id="btc_emergency_exit",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        category="emergency_exit",
    )
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=3, trade_cooldown_bars=0)
    entry_time = _portfolio().timestamp
    engine.record_fill("BTC-USD", is_entry=True, timestamp=entry_time)
    portfolio = _portfolio_with_position()

    bar_1 = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=entry_time + timedelta(hours=1),
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    bar_2 = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=entry_time + timedelta(hours=2),
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    bar_3 = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=entry_time + timedelta(hours=3),
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )

    orders, blocks = engine.on_bar(bar_1, _indicator(), portfolio)
    assert not orders
    assert blocks
    assert blocks[0]["reason"] == "emergency_exit_veto_min_hold"

    orders, blocks = engine.on_bar(bar_2, _indicator(), portfolio)
    assert not orders
    assert blocks
    assert blocks[0]["reason"] == "emergency_exit_veto_min_hold"

    orders, blocks = engine.on_bar(bar_3, _indicator(), portfolio)
    assert orders
    assert not blocks


def test_emergency_exit_dedup_overrides_high_conf_entry():
    entry_trigger = TriggerCondition(
        id="btc_entry",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="False",
        category="trend_continuation",
        confidence_grade="A",
    )
    emergency_trigger = TriggerCondition(
        id="btc_emergency_exit",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        category="emergency_exit",
        confidence_grade="A",
    )
    plan = _plan_with_triggers([entry_trigger, emergency_trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, max_triggers_per_symbol_per_bar=2)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio()
    portfolio.positions = {"BTC-USD": -1.0}

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert len(orders) == 1
    assert orders[0].emergency is True
    # Preemption should be recorded as a block event
    preempt_blocks = [b for b in blocks if b["reason"] == "emergency_exit_preempts_entry"]
    assert len(preempt_blocks) == 1
    assert preempt_blocks[0]["trigger_id"] == "btc_entry"
    assert preempt_blocks[0]["preempted_by"] == "btc_emergency_exit_exit"


def test_emergency_exit_dedup_wins_even_with_permissive_risk():
    """Guardrail: emergency exits win dedup unconditionally, even when risk
    constraints are fully permissive (simulating a 'risk-on' regime).
    This invariant must never be weakened.

    Trigger order: emergency first so it generates a flatten order, then
    the entry fires on the rebuilt (zero) portfolio.  Dedup must still
    pick the emergency exit and preempt the entry."""
    emergency_trigger = TriggerCondition(
        id="btc_emergency_exit",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        category="emergency_exit",
        confidence_grade="A",
    )
    entry_trigger = TriggerCondition(
        id="btc_entry",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="False",
        category="trend_continuation",
        confidence_grade="A",
    )
    # Emergency first: produces flatten, then entry fires on rebuilt portfolio
    plan = _plan_with_triggers([emergency_trigger, entry_trigger])
    # Fully permissive risk constraints — "risk-on" regime
    plan.risk_constraints = RiskConstraint(
        max_position_risk_pct=100.0,
        max_symbol_exposure_pct=100.0,
        max_portfolio_exposure_pct=100.0,
        max_daily_loss_pct=100.0,
    )
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, max_triggers_per_symbol_per_bar=2)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio()
    portfolio.positions = {"BTC-USD": 1.0}

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert len(orders) == 1
    assert orders[0].emergency is True
    preempt_blocks = [b for b in blocks if b["reason"] == "emergency_exit_preempts_entry"]
    assert len(preempt_blocks) == 1


# --- Runbook 05: Hold-rule bypass ---


def test_emergency_exit_bypasses_hold_rule():
    """Emergency exits must ignore hold_rule and fire even when hold_rule is active."""
    trigger = TriggerCondition(
        id="btc_emergency_exit",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        hold_rule="True",  # hold_rule always active — would suppress a normal exit
        category="emergency_exit",
    )
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert orders, "Emergency exit should fire despite active hold_rule"
    assert orders[0].emergency is True
    hold_blocks = [b for b in blocks if b["reason"] == "HOLD_RULE"]
    assert not hold_blocks, "Emergency exits must not produce HOLD_RULE blocks"


def test_regular_exit_respects_hold_rule():
    """Non-emergency exits must be suppressed when hold_rule evaluates True."""
    trigger = TriggerCondition(
        id="btc_exit",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        hold_rule="True",  # hold_rule always active
        category="trend_continuation",
    )
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert not orders, "Regular exit should be suppressed by active hold_rule"
    hold_blocks = [b for b in blocks if b["reason"] == "HOLD_RULE"]
    assert len(hold_blocks) == 1


# --- Runbook 05: Judge category disabling ---


def test_judge_disabled_category_blocks_emergency_exit():
    """disabled_categories=['emergency_exit'] must block emergency exits with CATEGORY reason."""
    trigger = TriggerCondition(
        id="btc_emergency_exit",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        category="emergency_exit",
    )
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    constraints = JudgeConstraints(
        disabled_trigger_ids=[],
        disabled_categories=["emergency_exit"],
        risk_mode="normal",
    )
    engine = TriggerEngine(
        plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0,
        judge_constraints=constraints,
    )
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert not orders, "Emergency exit must be blocked when its category is disabled"
    cat_blocks = [b for b in blocks if b["reason"] == "category"]
    assert len(cat_blocks) == 1
    assert "emergency_exit" in cat_blocks[0]["detail"]


# --- Runbook 06: Missing exit_rule edge cases ---


@pytest.mark.parametrize("exit_rule", ["", "   "], ids=["empty", "whitespace"])
def test_emergency_exit_missing_exit_rule_rejected_by_schema(exit_rule):
    """Pydantic schema rejects emergency_exit triggers with empty/whitespace exit_rule."""
    with pytest.raises(Exception):
        TriggerCondition(
            id="btc_emergency_exit",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="False",
            exit_rule=exit_rule,
            category="emergency_exit",
        )


def test_emergency_exit_none_exit_rule_rejected_by_schema():
    """Pydantic schema rejects emergency_exit triggers with exit_rule=None."""
    with pytest.raises(Exception):
        TriggerCondition(
            id="btc_emergency_exit",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="False",
            exit_rule=None,
            category="emergency_exit",
        )


@pytest.mark.parametrize("exit_rule", ["", "   "], ids=["empty", "whitespace"])
def test_emergency_exit_missing_exit_rule_runtime_defense(exit_rule):
    """Defense-in-depth: if a TriggerCondition with empty exit_rule bypasses
    schema validation, the trigger engine blocks it safely with
    'emergency_exit_missing_exit_rule' and cooldown metadata."""
    # Bypass Pydantic validation via model_construct for both trigger and plan
    trigger = TriggerCondition.model_construct(
        id="btc_emergency_exit",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule=exit_rule,
        category="emergency_exit",
        confidence_grade=None,
        stop_loss_pct=None,
        hold_rule=None,
    )
    # Use a valid plan then swap triggers bypassing Pydantic validate_assignment
    valid_trigger = TriggerCondition(
        id="placeholder",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        category="emergency_exit",
    )
    plan = _plan_with_triggers([valid_trigger])
    plan.__dict__["triggers"] = [trigger]
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=4, trade_cooldown_bars=2)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert not orders, "Missing exit_rule must not produce orders"
    assert len(blocks) == 1
    assert blocks[0]["reason"] == "emergency_exit_missing_exit_rule"
    assert "cooldown_recommendation_bars" in blocks[0]
    assert blocks[0]["cooldown_recommendation_bars"] == max(1, engine.trade_cooldown_bars, engine.min_hold_bars)


@pytest.mark.parametrize("policy", ["ignore", "defer"])
def test_conflicting_signal_policy_skips_orders(policy: str):
    trigger = TriggerCondition(
        id="btc_short",
        symbol="BTC-USD",
        direction="short",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        confidence_grade="A",
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(
        plan,
        risk_engine,
        min_hold_bars=0,
        trade_cooldown_bars=0,
        conflicting_signal_policy=policy,
    )
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert not orders
    assert blocks
    assert blocks[0]["reason"] == "conflicting_signal_detected"
    assert blocks[0]["policy"] == policy


def test_conflicting_signal_policy_exit_flattens():
    trigger = TriggerCondition(
        id="btc_short",
        symbol="BTC-USD",
        direction="short",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        confidence_grade="A",
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(
        plan,
        risk_engine,
        min_hold_bars=0,
        trade_cooldown_bars=0,
        conflicting_signal_policy="exit",
    )
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert orders
    assert orders[0].reason == "btc_short_exit"
    assert orders[0].intent == "conflict_exit"
    assert orders[0].side == "sell"
    assert blocks
    assert blocks[0]["reason"] == "conflicting_signal_detected"


def test_learning_tags_propagate_to_order():
    """Learning tags on TriggerCondition propagate to generated Order objects."""
    trigger = TriggerCondition(
        id="btc_learning",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        confidence_grade="A",
        learning_book=True,
        experiment_id="exp-001",
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    learning_settings = LearningBookSettings(enabled=True, notional_usd=500.0)
    trade_risk = TradeRiskEvaluator(risk_engine, learning_settings=learning_settings)
    engine = TriggerEngine(plan, risk_engine, trade_risk=trade_risk, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )

    orders, _ = engine.on_bar(bar, _indicator(), _portfolio())
    assert orders
    assert orders[0].learning_book is True
    assert orders[0].experiment_id == "exp-001"


def test_default_order_has_no_learning_tags():
    """Orders from triggers without learning tags have default False/None values."""
    trigger = TriggerCondition(
        id="btc_normal",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        confidence_grade="A",
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )

    orders, _ = engine.on_bar(bar, _indicator(), _portfolio())
    assert orders
    assert orders[0].learning_book is False
    assert orders[0].experiment_id is None
    assert orders[0].experiment_variant is None


def test_conflicting_signal_policy_reverse_flips():
    trigger = TriggerCondition(
        id="btc_short",
        symbol="BTC-USD",
        direction="short",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        confidence_grade="A",
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(
        plan,
        risk_engine,
        min_hold_bars=0,
        trade_cooldown_bars=0,
        conflicting_signal_policy="reverse",
    )
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert orders
    assert orders[0].reason == "btc_short"
    assert orders[0].intent == "conflict_reverse"
    assert orders[0].side == "sell"
    assert blocks
    assert blocks[0]["reason"] == "conflicting_signal_detected"


# =============================================================================
# Partial Exit Tests (Phase 2: Graduated De-risk)
# =============================================================================


def test_partial_exit_produces_correct_quantity():
    """Exit trigger with exit_fraction=0.5 should close half the position."""
    trigger = TriggerCondition(
        id="btc_risk_reduce",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="false",
        exit_rule="True",
        category="risk_reduce",
        confidence_grade="B",
        exit_fraction=0.5,
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    # Start with 1.0 BTC long position
    portfolio = _portfolio_with_position()

    orders, _ = engine.on_bar(bar, _indicator(), portfolio)
    assert orders
    assert orders[0].intent == "exit"
    assert orders[0].side == "sell"
    # Position is 1.0, exit_fraction=0.5 means exit 0.5
    assert abs(orders[0].quantity - 0.5) < 1e-9
    assert orders[0].exit_fraction == 0.5


def test_partial_exit_small_fraction():
    """Exit trigger with exit_fraction=0.25 should close quarter of position."""
    trigger = TriggerCondition(
        id="btc_trim",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="false",
        exit_rule="True",
        category="risk_reduce",
        confidence_grade="C",
        exit_fraction=0.25,
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, _ = engine.on_bar(bar, _indicator(), portfolio)
    assert orders
    assert abs(orders[0].quantity - 0.25) < 1e-9
    assert orders[0].exit_fraction == 0.25


def test_full_exit_has_no_exit_fraction():
    """Exit trigger without exit_fraction should close full position with None."""
    trigger = TriggerCondition(
        id="btc_full_exit",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="false",
        exit_rule="True",
        category="risk_off",
        confidence_grade="A",
        # No exit_fraction - should be full exit
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, _ = engine.on_bar(bar, _indicator(), portfolio)
    assert orders
    # Full exit - quantity should be full position (1.0)
    assert abs(orders[0].quantity - 1.0) < 1e-9
    # exit_fraction should be None for full exits
    assert orders[0].exit_fraction is None


def test_partial_exit_with_flat_direction():
    """Flat direction trigger with exit_fraction should close partial position.

    Note: When exit_rule fires, the intent is "exit" (not "flat"), since flat
    direction only affects order creation via entry_rule path.
    """
    trigger = TriggerCondition(
        id="btc_defensive",
        symbol="BTC-USD",
        direction="flat",
        timeframe="1h",
        entry_rule="false",
        exit_rule="True",
        category="risk_reduce",
        confidence_grade="B",
        exit_fraction=0.5,
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, _ = engine.on_bar(bar, _indicator(), portfolio)
    assert orders
    # When exit_rule fires, intent is "exit" (flat direction affects entry_rule path)
    assert orders[0].intent == "exit"
    assert orders[0].side == "sell"
    assert abs(orders[0].quantity - 0.5) < 1e-9
    assert orders[0].exit_fraction == 0.5


def test_partial_exit_short_position():
    """Partial exit on short position should buy back partial quantity."""
    trigger = TriggerCondition(
        id="btc_cover_partial",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="false",
        exit_rule="True",
        category="risk_reduce",
        confidence_grade="B",
        exit_fraction=0.5,
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    # Short position: -1.0 BTC
    portfolio = PortfolioState(
        timestamp=bar.timestamp,
        equity=100000.0,
        cash=150000.0,
        positions={"BTC-USD": -1.0},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )

    orders, _ = engine.on_bar(bar, _indicator(), portfolio)
    assert orders
    assert orders[0].intent == "exit"
    # Covering a short means buying
    assert orders[0].side == "buy"
    # Cover half of short position
    assert abs(orders[0].quantity - 0.5) < 1e-9
    assert orders[0].exit_fraction == 0.5


def test_risk_reduce_category_valid_in_trigger():
    """risk_reduce category can be used in triggers."""
    trigger = TriggerCondition(
        id="btc_trim",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="false",
        exit_rule="rsi_14 > 70",
        category="risk_reduce",
        confidence_grade="B",
        exit_fraction=0.5,
    )
    assert trigger.category == "risk_reduce"
    assert trigger.exit_fraction == 0.5


def test_risk_off_category_valid_in_trigger():
    """risk_off category can be used in triggers."""
    trigger = TriggerCondition(
        id="btc_defensive",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="false",
        exit_rule="vol_state == 'extreme'",
        category="risk_off",
        confidence_grade="A",
    )
    assert trigger.category == "risk_off"
    assert trigger.exit_fraction is None  # Full exit by default


# =============================================================================
# Phase 3: risk_reduce guardrail enforcement tests
# =============================================================================


def test_risk_reduce_respects_hold_rule():
    """risk_reduce exits should be blocked when hold_rule is active."""
    trigger = TriggerCondition(
        id="btc_risk_reduce",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="false",
        exit_rule="True",  # Would fire
        hold_rule="True",  # But hold rule is active
        category="risk_reduce",
        confidence_grade="B",
        exit_fraction=0.5,
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    # No orders because hold rule is active
    assert not orders
    # Block should be recorded
    assert any(b["reason"] == "HOLD_RULE" for b in blocks)


def test_risk_reduce_respects_min_hold():
    """risk_reduce exits should be blocked during min-hold period."""
    trigger = TriggerCondition(
        id="btc_risk_reduce",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="false",
        exit_rule="True",
        category="risk_reduce",
        confidence_grade="B",
        exit_fraction=0.5,
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    # Set min_hold_bars=3 so exit is blocked
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=3, trade_cooldown_bars=0)

    # Simulate that position was entered on bar 0
    bar0 = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    # Set the bar counter and position entry bar for min-hold tracking
    engine._bar_counter = 0
    engine._position_entry_bar["BTC-USD"] = 0  # Position entered at bar 0
    engine._last_entry_timestamp["BTC-USD"] = bar0.timestamp

    # Increment bar counter to bar 1 (within min-hold period of 3 bars)
    engine._bar_counter = 1

    from datetime import timedelta
    bar1 = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=bar0.timestamp + timedelta(hours=1),
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar1, _indicator(), portfolio)
    # No orders because within min-hold (bar 1 - bar 0 = 1 bar held, need 3)
    assert not orders
    # Block should be recorded
    assert any(b["reason"] == "MIN_HOLD_PERIOD" for b in blocks)


def test_emergency_exit_beats_risk_reduce_in_dedup():
    """emergency_exit should always win over risk_reduce in dedup.

    Note: Due to simulated portfolio state updates during trigger evaluation,
    the emergency_exit order may have a reduced quantity if risk_reduce was
    evaluated first. The key assertion is that emergency_exit WINS (is the
    only order kept) and is marked as emergency.
    """
    emergency_trigger = TriggerCondition(
        id="btc_emergency",
        symbol="BTC-USD",
        direction="flat",
        timeframe="1h",
        entry_rule="false",
        exit_rule="True",
        category="emergency_exit",
        confidence_grade="A",
    )
    risk_reduce_trigger = TriggerCondition(
        id="btc_risk_reduce",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="false",
        exit_rule="True",
        category="risk_reduce",
        confidence_grade="A",
        exit_fraction=0.5,
    )
    plan = _plan_with_triggers([risk_reduce_trigger, emergency_trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    # Should have exactly one order
    assert len(orders) == 1
    # The order should be from emergency_exit - it always wins dedup
    assert orders[0].trigger_category == "emergency_exit"
    assert orders[0].emergency is True
    # Emergency exits don't use partial exit_fraction
    assert orders[0].exit_fraction is None
    # Order quantity > 0 (at least some position to exit)
    assert orders[0].quantity > 0


def test_risk_reduce_goes_through_normal_exit_path():
    """risk_reduce uses the normal exit path with no special bypass.

    This verifies that risk_reduce behaves like a normal exit, not like an
    emergency_exit which bypasses certain checks.
    """
    trigger = TriggerCondition(
        id="btc_risk_reduce",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="false",
        exit_rule="True",
        category="risk_reduce",
        confidence_grade="B",
        exit_fraction=0.5,
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, _ = engine.on_bar(bar, _indicator(), portfolio)
    assert len(orders) == 1
    assert orders[0].trigger_category == "risk_reduce"
    # Not marked as emergency
    assert orders[0].emergency is False
    # Uses partial exit
    assert orders[0].exit_fraction == 0.5
    # Normal exit intent
    assert orders[0].intent == "exit"


def test_risk_off_goes_through_normal_exit_path():
    """risk_off uses the normal exit path with no special bypass.

    This verifies that risk_off behaves like a normal exit, not like an
    emergency_exit which bypasses certain checks.
    """
    trigger = TriggerCondition(
        id="btc_risk_off",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="false",
        exit_rule="True",
        category="risk_off",
        confidence_grade="A",
    )
    plan = _plan_with_triggers([trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio_with_position()

    orders, _ = engine.on_bar(bar, _indicator(), portfolio)
    assert len(orders) == 1
    assert orders[0].trigger_category == "risk_off"
    # Not marked as emergency
    assert orders[0].emergency is False
    # Full exit (no exit_fraction)
    assert orders[0].exit_fraction is None
    # Normal exit intent
    assert orders[0].intent == "exit"

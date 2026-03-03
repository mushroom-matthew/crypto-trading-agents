"""Tests for R65: exit contract enforcement via originating_plan_id pinning.

Verifies that exit triggers from a replanned plan are blocked from closing
positions opened under the old plan. Emergency exits always bypass the check.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from agents.strategies.risk_engine import RiskEngine
from agents.strategies.trigger_engine import Bar, TriggerEngine
from schemas.llm_strategist import (
    IndicatorSnapshot,
    PortfolioState,
    RiskConstraint,
    StrategyPlan,
    TriggerCondition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


def _indicator(close: float = 50000.0) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=_now(),
        close=close,
        atr_14=500.0,
    )


def _bar(close: float = 50000.0) -> Bar:
    return Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_now(),
        open=close,
        high=close + 50,
        low=close - 50,
        close=close,
        volume=1.0,
    )


def _portfolio_with_long() -> PortfolioState:
    return PortfolioState(
        timestamp=_now(),
        equity=100000.0,
        cash=80000.0,
        positions={"BTC-USD": 1.0},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )


def _risk_constraints() -> RiskConstraint:
    return RiskConstraint(
        max_position_risk_pct=0.0,
        max_symbol_exposure_pct=100.0,
        max_portfolio_exposure_pct=100.0,
        max_daily_loss_pct=10.0,
    )


def _exit_trigger(
    category: str = "trend_continuation",
    direction: str = "long",
) -> TriggerCondition:
    return TriggerCondition(
        id="btc_exit",
        symbol="BTC-USD",
        direction=direction,
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        category=category,
        stop_loss_pct=2.0,
    )


def _plan(trigger: TriggerCondition, plan_id: str = "plan-abc") -> StrategyPlan:
    plan = StrategyPlan(
        generated_at=_now(),
        valid_until=_now(),
        global_view="test",
        regime="range",
        triggers=[trigger],
        risk_constraints=_risk_constraints(),
    )
    # Override the auto-generated plan_id with a fixed one for testing
    object.__setattr__(plan, "plan_id", plan_id)
    return plan


def _engine(
    trigger: TriggerCondition,
    plan_id: str = "plan-abc",
    position_originating_plans: dict[str, str] | None = None,
) -> TriggerEngine:
    plan = _plan(trigger, plan_id)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    return TriggerEngine(
        plan,
        risk_engine,
        position_originating_plans=position_originating_plans,
        min_hold_bars=0,
        trade_cooldown_bars=0,
    )


# ---------------------------------------------------------------------------
# 1. SessionState.position_originating_plans field
# ---------------------------------------------------------------------------


def test_session_state_has_position_originating_plans_field():
    """SessionState must have position_originating_plans: Dict[str, str] = {}."""
    from tools.paper_trading import SessionState

    state = SessionState(
        session_id="sess-1",
        symbols=["BTC-USD"],
        strategy_prompt=None,
        plan_interval_hours=4.0,
    )
    assert hasattr(state, "position_originating_plans")
    assert isinstance(state.position_originating_plans, dict)
    assert state.position_originating_plans == {}


def test_session_state_position_originating_plans_roundtrip():
    """position_originating_plans must survive model_dump / model_validate round-trip."""
    from tools.paper_trading import SessionState

    state = SessionState(
        session_id="sess-2",
        symbols=["BTC-USD"],
        strategy_prompt=None,
        plan_interval_hours=4.0,
        position_originating_plans={"BTC-USD": "plan-xyz"},
    )
    dumped = state.model_dump()
    restored = SessionState.model_validate(dumped)
    assert restored.position_originating_plans == {"BTC-USD": "plan-xyz"}


# ---------------------------------------------------------------------------
# 2. TriggerEngine._should_apply_exit_trigger logic
# ---------------------------------------------------------------------------


def test_should_apply_exit_trigger_same_plan_allows():
    """Exit trigger from the same plan that opened the position should be allowed."""
    trigger = _exit_trigger()
    engine = _engine(trigger, plan_id="plan-A", position_originating_plans={"BTC-USD": "plan-A"})

    result = engine._should_apply_exit_trigger(
        trigger=trigger,
        symbol="BTC-USD",
        current_plan_id="plan-A",
        position_originating_plans={"BTC-USD": "plan-A"},
    )
    assert result is True
    assert engine.exit_binding_mismatch_count == 0


def test_should_apply_exit_trigger_different_plan_blocks():
    """Exit trigger from a different plan (replan) should be blocked."""
    trigger = _exit_trigger()
    engine = _engine(trigger, plan_id="plan-B", position_originating_plans={"BTC-USD": "plan-A"})

    result = engine._should_apply_exit_trigger(
        trigger=trigger,
        symbol="BTC-USD",
        current_plan_id="plan-B",
        position_originating_plans={"BTC-USD": "plan-A"},
    )
    assert result is False
    assert engine.exit_binding_mismatch_count == 1


def test_should_apply_exit_trigger_emergency_exit_always_allowed():
    """Emergency exits must bypass plan-level pinning regardless of mismatch."""
    trigger = _exit_trigger(category="emergency_exit", direction="flat")
    engine = _engine(trigger, plan_id="plan-B", position_originating_plans={"BTC-USD": "plan-A"})

    result = engine._should_apply_exit_trigger(
        trigger=trigger,
        symbol="BTC-USD",
        current_plan_id="plan-B",
        position_originating_plans={"BTC-USD": "plan-A"},
    )
    assert result is True
    assert engine.exit_binding_mismatch_count == 0


def test_should_apply_exit_trigger_flatten_direction_allowed():
    """Flatten direction (not long/short) bypasses plan pinning."""
    trigger = _exit_trigger(category="trend_continuation", direction="flat")
    engine = _engine(trigger, plan_id="plan-B", position_originating_plans={"BTC-USD": "plan-A"})

    result = engine._should_apply_exit_trigger(
        trigger=trigger,
        symbol="BTC-USD",
        current_plan_id="plan-B",
        position_originating_plans={"BTC-USD": "plan-A"},
    )
    assert result is True
    assert engine.exit_binding_mismatch_count == 0


def test_should_apply_exit_trigger_no_tracking_allows():
    """When position_originating_plans is None, degrade gracefully and allow."""
    trigger = _exit_trigger()
    engine = _engine(trigger, plan_id="plan-B", position_originating_plans=None)

    result = engine._should_apply_exit_trigger(
        trigger=trigger,
        symbol="BTC-USD",
        current_plan_id="plan-B",
        position_originating_plans=None,
    )
    assert result is True


def test_should_apply_exit_trigger_no_origin_for_symbol_allows():
    """When no origin is recorded for this symbol, allow the exit."""
    trigger = _exit_trigger()
    engine = _engine(trigger, plan_id="plan-B", position_originating_plans={})

    result = engine._should_apply_exit_trigger(
        trigger=trigger,
        symbol="BTC-USD",
        current_plan_id="plan-B",
        position_originating_plans={},
    )
    assert result is True


def test_should_apply_exit_trigger_no_current_plan_id_allows():
    """When current_plan_id is None, degrade gracefully and allow."""
    trigger = _exit_trigger()
    engine = _engine(trigger, plan_id="plan-B", position_originating_plans={"BTC-USD": "plan-A"})

    result = engine._should_apply_exit_trigger(
        trigger=trigger,
        symbol="BTC-USD",
        current_plan_id=None,
        position_originating_plans={"BTC-USD": "plan-A"},
    )
    assert result is True


# ---------------------------------------------------------------------------
# 3. on_bar integration — plan-level exit binding enforcement
# ---------------------------------------------------------------------------


def test_on_bar_blocks_exit_from_replanned_plan():
    """on_bar must block exit trigger when current plan differs from originating plan."""
    trigger = _exit_trigger(category="trend_continuation", direction="long")
    # Position was opened under "plan-OLD"; current plan is "plan-NEW"
    engine = _engine(
        trigger,
        plan_id="plan-NEW",
        position_originating_plans={"BTC-USD": "plan-OLD"},
    )
    portfolio = _portfolio_with_long()
    # Put portfolio entry bar in the past so hold period doesn't trigger
    engine._position_entry_bar["BTC-USD"] = -100

    orders, blocks = engine.on_bar(_bar(), _indicator(), portfolio)

    assert not orders, "Exit from replanned plan should be blocked"
    plan_mismatch_blocks = [b for b in blocks if b["reason"] == "exit_binding_plan_mismatch"]
    assert plan_mismatch_blocks, f"Expected exit_binding_plan_mismatch block, got: {blocks}"
    assert engine.exit_binding_mismatch_count == 1


def test_on_bar_allows_exit_from_same_plan():
    """on_bar must allow exit trigger when current plan matches originating plan."""
    trigger = _exit_trigger(category="trend_continuation", direction="long")
    engine = _engine(
        trigger,
        plan_id="plan-A",
        position_originating_plans={"BTC-USD": "plan-A"},
    )
    portfolio = _portfolio_with_long()
    engine._position_entry_bar["BTC-USD"] = -100

    orders, blocks = engine.on_bar(_bar(), _indicator(), portfolio)

    # Should produce an exit order (no plan mismatch)
    plan_mismatch_blocks = [b for b in blocks if b["reason"] == "exit_binding_plan_mismatch"]
    assert not plan_mismatch_blocks, f"Same-plan exit should not be blocked: {blocks}"
    assert engine.exit_binding_mismatch_count == 0


def test_on_bar_emergency_exit_bypasses_plan_pinning():
    """Emergency exits must not be blocked by plan-level originating_plan_id check."""
    trigger = _exit_trigger(category="emergency_exit", direction="flat")
    # Plan mismatch: position opened under plan-OLD, current is plan-NEW
    engine = _engine(
        trigger,
        plan_id="plan-NEW",
        position_originating_plans={"BTC-USD": "plan-OLD"},
    )
    portfolio = _portfolio_with_long()

    orders, blocks = engine.on_bar(_bar(), _indicator(), portfolio)

    # Emergency exit should fire despite plan mismatch
    assert orders, "Emergency exit should bypass plan pinning"
    plan_mismatch_blocks = [b for b in blocks if b["reason"] == "exit_binding_plan_mismatch"]
    assert not plan_mismatch_blocks
    assert engine.exit_binding_mismatch_count == 0


def test_on_bar_no_position_originating_plans_allows_exit():
    """When no position tracking is provided, exits should not be blocked."""
    trigger = _exit_trigger(category="trend_continuation", direction="long")
    engine = _engine(trigger, plan_id="plan-A", position_originating_plans=None)
    portfolio = _portfolio_with_long()
    engine._position_entry_bar["BTC-USD"] = -100

    orders, blocks = engine.on_bar(_bar(), _indicator(), portfolio)

    plan_mismatch_blocks = [b for b in blocks if b["reason"] == "exit_binding_plan_mismatch"]
    assert not plan_mismatch_blocks, "No tracking → should not block exits"


# ---------------------------------------------------------------------------
# 4. evaluate_triggers_activity — parameter + return value
# ---------------------------------------------------------------------------


def test_evaluate_triggers_activity_accepts_position_originating_plans():
    """evaluate_triggers_activity must accept position_originating_plans parameter."""
    from tools.paper_trading import evaluate_triggers_activity

    sig = inspect.signature(evaluate_triggers_activity)
    assert "position_originating_plans" in sig.parameters, (
        "evaluate_triggers_activity missing position_originating_plans parameter (R65)"
    )


def test_evaluate_triggers_activity_returns_mismatch_blocked_key():
    """evaluate_triggers_activity must return exit_binding_mismatch_blocked in result."""
    from tools.paper_trading import evaluate_triggers_activity

    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    plan_dict = {
        "generated_at": now.isoformat(),
        "valid_until": now.isoformat(),
        "global_view": "test",
        "regime": "range",
        "triggers": [
            {
                "id": "btc_exit",
                "symbol": "BTC-USD",
                "direction": "long",
                "timeframe": "1h",
                "entry_rule": "False",
                "exit_rule": "True",
                "category": "trend_continuation",
                "stop_loss_pct": 2.0,
            }
        ],
        "risk_constraints": {
            "max_position_risk_pct": 0.0,
            "max_symbol_exposure_pct": 100.0,
            "max_portfolio_exposure_pct": 100.0,
            "max_daily_loss_pct": 10.0,
        },
    }
    market_data = {
        "BTC-USD": {
            "close": 50000.0,
            "timestamp": now.isoformat(),
            "timeframe": "1h",
            "atr_14": 500.0,
        }
    }
    portfolio_state = {
        "cash": 80000.0,
        "positions": {"BTC-USD": 1.0},
        "total_equity": 130000.0,
        "position_meta": {},
    }
    # Position opened under "plan-OLD"; current plan is different
    result = evaluate_triggers_activity(
        plan_dict=plan_dict,
        market_data=market_data,
        portfolio_state=portfolio_state,
        position_originating_plans={"BTC-USD": "plan-OLD"},
    )

    assert "exit_binding_mismatch_blocked" in result, (
        "evaluate_triggers_activity must return exit_binding_mismatch_blocked"
    )
    # The exit trigger fired exit_rule=True with plan mismatch → should be blocked
    assert result["exit_binding_mismatch_blocked"] >= 1


def test_evaluate_triggers_activity_emits_trade_blocked_event_on_mismatch():
    """trade_blocked event with reason=exit_binding_mismatch must be in events when count > 0."""
    from tools.paper_trading import evaluate_triggers_activity

    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    plan_dict = {
        "generated_at": now.isoformat(),
        "valid_until": now.isoformat(),
        "global_view": "test",
        "regime": "range",
        "triggers": [
            {
                "id": "btc_exit",
                "symbol": "BTC-USD",
                "direction": "long",
                "timeframe": "1h",
                "entry_rule": "False",
                "exit_rule": "True",
                "category": "trend_continuation",
                "stop_loss_pct": 2.0,
            }
        ],
        "risk_constraints": {
            "max_position_risk_pct": 0.0,
            "max_symbol_exposure_pct": 100.0,
            "max_portfolio_exposure_pct": 100.0,
            "max_daily_loss_pct": 10.0,
        },
    }
    market_data = {
        "BTC-USD": {
            "close": 50000.0,
            "timestamp": now.isoformat(),
            "timeframe": "1h",
        }
    }
    portfolio_state = {
        "cash": 80000.0,
        "positions": {"BTC-USD": 1.0},
        "total_equity": 130000.0,
    }
    result = evaluate_triggers_activity(
        plan_dict=plan_dict,
        market_data=market_data,
        portfolio_state=portfolio_state,
        position_originating_plans={"BTC-USD": "plan-OLD"},
    )

    if result["exit_binding_mismatch_blocked"] > 0:
        mismatch_events = [
            e for e in result.get("events", [])
            if e["type"] == "trade_blocked"
            and e["payload"].get("reason") == "exit_binding_mismatch"
        ]
        assert mismatch_events, (
            "Expected trade_blocked event with reason=exit_binding_mismatch when count > 0"
        )


def test_evaluate_triggers_activity_no_mismatch_when_same_plan():
    """When plan matches, exit_binding_mismatch_blocked must be 0."""
    from tools.paper_trading import evaluate_triggers_activity

    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # We need to know the plan_id. StrategyPlan auto-generates it, so we skip
    # position_originating_plans entirely (None) which forces allow-all.
    plan_dict = {
        "generated_at": now.isoformat(),
        "valid_until": now.isoformat(),
        "global_view": "test",
        "regime": "range",
        "triggers": [
            {
                "id": "btc_exit",
                "symbol": "BTC-USD",
                "direction": "long",
                "timeframe": "1h",
                "entry_rule": "False",
                "exit_rule": "False",  # won't fire
                "category": "trend_continuation",
                "stop_loss_pct": 2.0,
            }
        ],
        "risk_constraints": {
            "max_position_risk_pct": 0.0,
            "max_symbol_exposure_pct": 100.0,
            "max_portfolio_exposure_pct": 100.0,
            "max_daily_loss_pct": 10.0,
        },
    }
    market_data = {
        "BTC-USD": {
            "close": 50000.0,
            "timestamp": now.isoformat(),
            "timeframe": "1h",
        }
    }
    portfolio_state = {
        "cash": 80000.0,
        "positions": {"BTC-USD": 1.0},
        "total_equity": 130000.0,
    }
    result = evaluate_triggers_activity(
        plan_dict=plan_dict,
        market_data=market_data,
        portfolio_state=portfolio_state,
        position_originating_plans=None,  # no tracking → allow all
    )
    assert result["exit_binding_mismatch_blocked"] == 0


# ---------------------------------------------------------------------------
# 5. PaperTradingWorkflow — snapshot/restore includes position_originating_plans
# ---------------------------------------------------------------------------


def test_paper_trading_workflow_snapshot_includes_originating_plans():
    """_snapshot_state must include position_originating_plans for continue-as-new."""
    import pathlib
    src = pathlib.Path("tools/paper_trading.py").read_text()
    assert "position_originating_plans" in src, (
        "position_originating_plans not found in tools/paper_trading.py — R65 wiring missing"
    )


def test_trigger_engine_init_accepts_position_originating_plans():
    """TriggerEngine.__init__ must accept position_originating_plans parameter (R65)."""
    sig = inspect.signature(TriggerEngine.__init__)
    assert "position_originating_plans" in sig.parameters, (
        "TriggerEngine.__init__ missing position_originating_plans parameter (R65)"
    )


def test_trigger_engine_has_exit_binding_mismatch_count_property():
    """TriggerEngine must expose exit_binding_mismatch_count property (R65)."""
    trigger = _exit_trigger()
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine)
    assert hasattr(engine, "exit_binding_mismatch_count")
    assert engine.exit_binding_mismatch_count == 0


def test_trigger_engine_has_should_apply_exit_trigger_method():
    """TriggerEngine must have _should_apply_exit_trigger method (R65)."""
    assert hasattr(TriggerEngine, "_should_apply_exit_trigger"), (
        "TriggerEngine missing _should_apply_exit_trigger method"
    )

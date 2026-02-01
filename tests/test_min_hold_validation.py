"""Runbook 15 – min_hold vs exit timeframe validation and binding pct."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from schemas.llm_strategist import RiskConstraint, StrategyPlan, TriggerCondition
from trading_core.trigger_compiler import validate_min_hold_vs_exits
from backtesting.reports import build_run_summary

_NOW = datetime.now(tz=timezone.utc)
_RISK = RiskConstraint(
    max_position_risk_pct=2.0,
    max_symbol_exposure_pct=10.0,
    max_portfolio_exposure_pct=50.0,
    max_daily_loss_pct=5.0,
)


def _make_plan(triggers: list[TriggerCondition]) -> StrategyPlan:
    return StrategyPlan(
        run_id="test-run",
        plan_id="plan-1",
        generated_at=_NOW,
        valid_until=_NOW + timedelta(hours=8),
        regime="range",
        triggers=triggers,
        sizing_rules=[],
        risk_constraints=_RISK,
    )


def _exit_trigger(timeframe: str = "1h") -> TriggerCondition:
    return TriggerCondition(
        id="exit_rsi",
        symbol="BTC-USD",
        direction="exit",
        timeframe=timeframe,
        entry_rule="rsi_14 > 70",
        exit_rule="rsi_14 < 30",
    )


def _entry_trigger(timeframe: str = "5m") -> TriggerCondition:
    return TriggerCondition(
        id="entry_ema",
        symbol="BTC-USD",
        direction="long",
        timeframe=timeframe,
        entry_rule="close > ema_short",
        exit_rule="close < ema_short",
    )


def _emergency_trigger(timeframe: str = "5m") -> TriggerCondition:
    return TriggerCondition(
        id="emergency_1",
        symbol="BTC-USD",
        direction="exit",
        timeframe=timeframe,
        category="emergency_exit",
        entry_rule="rsi_14 < 20",
        exit_rule="rsi_14 < 10",
    )


# -- validate_min_hold_vs_exits tests --

def test_warning_when_min_hold_gte_smallest_exit():
    """Should emit warning when min_hold >= smallest exit timeframe."""
    plan = _make_plan([_exit_trigger("1h")])
    warnings = validate_min_hold_vs_exits(plan, min_hold_hours=1.0)
    assert len(warnings) == 1
    assert "min_hold" in warnings[0]
    assert "1.00h" in warnings[0]


def test_warning_when_min_hold_exceeds_exit():
    plan = _make_plan([_exit_trigger("15m")])
    warnings = validate_min_hold_vs_exits(plan, min_hold_hours=1.0)
    assert len(warnings) == 1


def test_no_warning_when_min_hold_below_exit():
    """No warning when min_hold < smallest exit timeframe."""
    plan = _make_plan([_exit_trigger("4h")])
    warnings = validate_min_hold_vs_exits(plan, min_hold_hours=1.0)
    assert warnings == []


def test_no_warning_when_min_hold_zero():
    """No warning when min_hold disabled."""
    plan = _make_plan([_exit_trigger("5m")])
    warnings = validate_min_hold_vs_exits(plan, min_hold_hours=0.0)
    assert warnings == []


def test_emergency_exit_excluded():
    """Emergency exits should not count toward the smallest exit timeframe."""
    plan = _make_plan([_emergency_trigger("5m"), _exit_trigger("4h")])
    warnings = validate_min_hold_vs_exits(plan, min_hold_hours=1.0)
    assert warnings == []


def test_entry_triggers_with_exit_rule_included():
    """Long/short triggers with exit_rule count as having exit behavior."""
    plan = _make_plan([_entry_trigger("15m")])
    warnings = validate_min_hold_vs_exits(plan, min_hold_hours=1.0)
    # 15m = 0.25h, min_hold=1.0h => warning
    assert len(warnings) == 1


def test_no_exit_triggers():
    """No exit triggers means no warnings."""
    plan = _make_plan([
        TriggerCondition(
            id="entry_only",
            symbol="BTC-USD",
            direction="long",
            timeframe="5m",
            entry_rule="close > ema_short",
            exit_rule="",
        )
    ])
    warnings = validate_min_hold_vs_exits(plan, min_hold_hours=1.0)
    assert warnings == []


# -- min_hold_binding_pct tests --

def _make_daily_report_with_min_hold(blocked_min_hold: int, executed: int) -> dict:
    return {
        "date": "2025-01-01",
        "equity_return_pct": 0.5,
        "trade_count": executed,
        "limit_stats": {
            "blocked_by_daily_cap": 0,
            "blocked_by_plan_limits": 0,
            "blocked_by_direction": 0,
            "attempted_triggers": blocked_min_hold + executed,
            "executed_trades": executed,
            "blocked_by_min_hold": blocked_min_hold,
            "min_hold_binding_pct": (
                (blocked_min_hold / (blocked_min_hold + executed)) * 100.0
                if (blocked_min_hold + executed) > 0
                else 0.0
            ),
        },
    }


def test_min_hold_binding_pct_zero_when_no_blocks():
    report = _make_daily_report_with_min_hold(blocked_min_hold=0, executed=5)
    summary = build_run_summary([report])
    assert summary["min_hold_binding_pct_mean"] == 0.0


def test_min_hold_binding_pct_positive_when_blocks():
    report = _make_daily_report_with_min_hold(blocked_min_hold=3, executed=7)
    summary = build_run_summary([report])
    expected = (3 / (3 + 7)) * 100.0  # 30%
    assert abs(summary["min_hold_binding_pct_mean"] - expected) < 0.01


def test_min_hold_binding_pct_mean_across_days():
    r1 = _make_daily_report_with_min_hold(blocked_min_hold=0, executed=10)
    r2 = _make_daily_report_with_min_hold(blocked_min_hold=5, executed=5)
    summary = build_run_summary([r1, r2])
    # day1: 0%, day2: 50%  -> mean = 25%
    assert abs(summary["min_hold_binding_pct_mean"] - 25.0) < 0.01

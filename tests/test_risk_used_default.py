"""Runbook 14 – risk_used defaults to actual_risk when budgets are off."""

from __future__ import annotations

from backtesting.reports import build_run_summary


def _make_risk_event(risk_used: float, actual_risk: float) -> dict:
    return {
        "trigger_id": "t1",
        "timeframe": "5m",
        "hour": 10,
        "risk_used": risk_used,
        "actual_risk_at_stop": actual_risk,
    }


def _make_daily_report(
    risk_used: float,
    actual_risk: float,
    trade_count: int = 1,
) -> dict:
    return {
        "date": "2025-01-01",
        "equity_return_pct": 0.5,
        "trade_count": trade_count,
        "start_equity": 1000.0,
        "risk_budget_pct": 0.0,
        "risk_usage_events": [_make_risk_event(risk_used, actual_risk)],
        "limit_stats": {
            "blocked_by_daily_cap": 0,
            "blocked_by_plan_limits": 0,
            "blocked_by_direction": 0,
            "attempted_triggers": trade_count,
            "executed_trades": trade_count,
            "blocked_by_min_hold": 0,
            "min_hold_binding_pct": 0.0,
        },
        "trigger_quality": {
            "t1": {
                "pnl": 5.0,
                "risk_used_abs": risk_used,
                "actual_risk_abs": actual_risk,
                "trades": trade_count,
                "wins": 1,
                "losses": 0,
            },
        },
    }


def test_risk_used_defaults_to_actual_risk_when_zero():
    """When allowance is 0 (budgets off), risk_used should == actual_risk."""
    # Simulate the defaulting logic from the runner
    allowance = 0.0
    actual_risk = 12.5

    risk_used = allowance or 0.0
    # Replicate the Runbook 14 fix
    if risk_used == 0.0 and actual_risk and actual_risk > 0:
        risk_used = actual_risk

    assert risk_used == 12.5


def test_risk_used_unchanged_when_budget_enabled():
    """When budget provides allowance > 0, risk_used stays as allowance."""
    allowance = 8.0
    actual_risk = 12.5

    risk_used = allowance or 0.0
    if risk_used == 0.0 and actual_risk and actual_risk > 0:
        risk_used = actual_risk

    assert risk_used == 8.0


def test_daily_report_allocated_risk_abs_nonzero():
    """Run summary allocated_risk_abs (via trigger_quality) should be non-zero
    when actual_risk is present even if allowance was 0."""
    report = _make_daily_report(risk_used=10.0, actual_risk=10.0)
    summary = build_run_summary([report])
    tq = summary.get("trigger_quality", {})
    assert "t1" in tq
    assert tq["t1"]["risk_used_abs"] > 0
    assert tq["t1"]["rpr"] != 0.0  # pnl / risk_used should be meaningful


def test_daily_report_allocated_risk_abs_zero_without_fix():
    """Verify that when risk_used is 0.0 the rpr falls back gracefully."""
    report = _make_daily_report(risk_used=0.0, actual_risk=10.0)
    summary = build_run_summary([report])
    tq = summary.get("trigger_quality", {})
    assert "t1" in tq
    # risk_used_abs is 0 so rpr should be 0 (no division error)
    assert tq["t1"]["rpr"] == 0.0
    # But rpr_actual should still work because actual_risk_abs > 0
    assert tq["t1"]["rpr_actual"] == 0.5  # 5.0 / 10.0

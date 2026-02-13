import pytest

from backtesting.reports import build_run_summary


def test_build_run_summary_aggregates_limits_and_risk_usage():
    daily = [
        {
            "equity_return_pct": 0.1,
            "risk_budget": {"used_pct": 5.0, "utilization_pct": 5.0},
            "trade_count": 4,
            "attempted_triggers": 10,
            "executed_trades": 4,
            "pnl_breakdown": {"flattening_pct": 1.0, "fees_pct": -0.5},
            "flatten_positions_daily": True,
            "limit_stats": {"blocked_by_daily_cap": 6, "blocked_by_plan_limits": 0, "blocked_by_direction": 1},
            "cap_state": {
                "policy": {"max_trades_per_day": 30, "max_triggers_per_symbol_per_day": 40},
                "derived": {"max_trades_per_day": 12, "max_triggers_per_symbol_per_day": 12},
                "resolved": {"max_trades_per_day": 30, "max_triggers_per_symbol_per_day": 40},
                "session_caps": {"0-12": 18},
                "flags": {"strict_fixed_caps": True},
            },
        },
        {
            "equity_return_pct": -0.2,
            "risk_budget": {"used_pct": 15.0, "utilization_pct": 15.0},
            "trade_count": 2,
            "attempted_triggers": 8,
            "executed_trades": 2,
            "pnl_breakdown": {"flattening_pct": 0.0, "fees_pct": -0.25},
            "limit_stats": {"blocked_by_daily_cap": 0, "blocked_by_plan_limits": 5, "blocked_by_direction": 0},
            "cap_state": {
                "policy": {"max_trades_per_day": 20, "max_triggers_per_symbol_per_day": 25},
                "derived": {"max_trades_per_day": 10, "max_triggers_per_symbol_per_day": 10},
                "resolved": {"max_trades_per_day": 10, "max_triggers_per_symbol_per_day": 10},
                "session_caps": {"0-12": 5},
                "flags": {"strict_fixed_caps": False},
            },
        },
    ]
    summary = build_run_summary(daily)
    assert summary["days"] == 2
    assert summary["risk_budget_used_pct_mean"] == 10.0
    assert summary["risk_budget_used_pct_median"] == 10.0
    assert summary["risk_budget_utilization_pct_mean"] == 10.0
    assert summary["risk_budget_utilization_pct_median"] == 10.0
    assert summary["risk_budget_under_25_pct_days"] == 100.0
    assert summary["risk_budget_25_to_75_pct_days"] == 0.0
    assert summary["risk_budget_over_75_pct_days"] == 0.0
    assert summary["flattening_pct_mean"] == 0.5
    assert summary["fees_pct_mean"] == -0.375
    assert summary["flatten_trade_days_pct"] == 50.0
    assert summary["trade_count_mean"] == 3.0
    assert summary["blocked_by_daily_cap_mean"] == 3.0
    assert summary["blocked_by_plan_limits_mean"] == 2.5
    assert summary["blocked_by_direction_mean"] == 0.5
    assert 0.0 <= summary["execution_rate_mean"] <= 1.0
    assert summary["brake_distribution"]["daily_cap"] == 1
    assert summary["brake_distribution"]["plan_limit"] == 1
    cap_state = summary["cap_state"]
    assert cap_state["policy"]["max_trades_per_day"]["mean"] == 25.0
    assert cap_state["derived"]["max_trades_per_day"]["max"] == 12.0
    assert cap_state["resolved"]["max_triggers_per_symbol_per_day"]["min"] == 10.0
    assert cap_state["flags"]["strict_fixed_caps_days"] == 1
    assert cap_state["flags"]["legacy_mode_days"] == 1


def test_risk_utilization_from_budget_pct_and_usage_events():
    """When risk_budget is empty but risk_budget_pct and risk_usage_events exist,
    the summary should derive non-zero risk utilization via the fallback path."""
    daily = [
        {
            "equity_return_pct": 0.05,
            "risk_budget": {},  # Empty — primary path yields nothing
            "risk_budget_pct": 10.0,  # 10% of equity
            "start_equity": 10000.0,
            "risk_usage_events": [
                {"risk_used": 200.0},
                {"risk_used": 300.0},
            ],
            "trade_count": 3,
            "limit_stats": {
                "blocked_by_daily_cap": 0,
                "blocked_by_plan_limits": 0,
                "blocked_by_direction": 0,
            },
        },
    ]
    summary = build_run_summary(daily)
    # Budget = 10% of 10000 = 1000. Used = 200+300 = 500. Utilization = 50%.
    assert summary["risk_budget_used_pct_mean"] == pytest.approx(50.0)
    assert summary["risk_budget_utilization_pct_mean"] == pytest.approx(50.0)
    # Not all under 25% — should be in the 25-75 bucket
    assert summary["risk_budget_25_to_75_pct_days"] == 100.0

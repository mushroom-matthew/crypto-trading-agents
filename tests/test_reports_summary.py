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
        },
        {
            "equity_return_pct": -0.2,
            "risk_budget": {"used_pct": 15.0, "utilization_pct": 15.0},
            "trade_count": 2,
            "attempted_triggers": 8,
            "executed_trades": 2,
            "pnl_breakdown": {"flattening_pct": 0.0, "fees_pct": -0.25},
            "limit_stats": {"blocked_by_daily_cap": 0, "blocked_by_plan_limits": 5, "blocked_by_direction": 0},
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

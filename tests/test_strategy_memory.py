from __future__ import annotations

from agents.strategies.strategy_memory import build_strategy_memory


def test_build_strategy_memory_tracks_recent_scores():
    history = [
        {
            "date": "2024-01-01",
            "start_equity": 100000.0,
            "end_equity": 101000.0,
            "return_pct": 1.0,
            "trade_count": 8,
            "symbol_pnl": {"BTC-USD": {"gross_pct": 0.8, "net_pct": 0.6}},
            "plan_limits": {
                "trigger_catalog": {
                    "btc_pullback": {"symbol": "BTC-USD", "category": "trend_continuation"},
                    "eth_breakout": {"symbol": "ETH-USD", "category": "volatility_breakout"},
                }
            },
            "trigger_stats": {
                "btc_pullback": {"executed": 4, "blocked": 1, "blocked_by_reason": {"daily_cap": 1}},
                "eth_breakout": {"executed": 0, "blocked": 3, "blocked_by_reason": {"risk_budget": 3}},
            },
            "judge_feedback": {
                "score": 70.0,
                "notes": "Positive daily performance.",
                "strategist_constraints": {"must_fix": ["none"], "vetoes": [], "boost": ["trend"], "regime_correction": "bull", "sizing_adjustments": {}},
            },
        },
        {
            "date": "2024-01-02",
            "start_equity": 101000.0,
            "end_equity": 100000.0,
            "return_pct": -0.99,
            "trade_count": 14,
            "symbol_pnl": {"ETH-USD": {"gross_pct": -0.7, "net_pct": -0.9}},
            "plan_limits": {
                "trigger_catalog": {
                    "eth_breakout": {"symbol": "ETH-USD", "category": "volatility_breakout"},
                    "btc_pullback": {"symbol": "BTC-USD", "category": "trend_continuation"},
                }
            },
            "trigger_stats": {
                "eth_breakout": {"executed": 6, "blocked": 0, "blocked_by_reason": {}},
                "btc_pullback": {"executed": 0, "blocked": 5, "blocked_by_reason": {"risk_budget": 2}},
            },
            "judge_feedback": {
                "score": 45.0,
                "notes": "Drawdown detected; tighten stops.",
                "strategist_constraints": {"must_fix": ["tighten"], "vetoes": ["overtrading"], "boost": [], "regime_correction": "mixed", "sizing_adjustments": {}},
            },
        },
    ]

    summary = build_strategy_memory(history, limit=8)
    assert len(summary["recent_evaluations"]) == 2
    assert summary["risk_events"]["overtrading"] >= 1
    assert "eth_breakout" in summary["trigger_performance"]
    assert summary["category_performance"]
    assert summary["trade_behavior"]["avg_trades_per_day"] > 0
    assert summary["equity_trend"]["direction"] in {"up", "down", "flat"}
    assert summary["constraints_snapshot"].get("regime_correction")

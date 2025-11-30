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
            "top_triggers": [["btc_pullback", 4]],
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
            "top_triggers": [["eth_breakout", 6]],
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
    assert summary["trade_behavior"]["avg_trades_per_day"] > 0
    assert summary["equity_trend"]["direction"] in {"up", "down", "flat"}
    assert summary["constraints_snapshot"].get("regime_correction")

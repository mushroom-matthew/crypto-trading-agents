"""Ensure block counters surface in run summaries."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.reports import build_run_summary


def test_block_totals_aggregated() -> None:
    daily_reports = [
        {
            "limit_stats": {"risk_block_breakdown": {"session_cap": 1, "trigger_load": 2}},
            "daily_loss_blocks": 0,
            "daily_cap_blocks": 0,
            "risk_budget_blocks": 0,
            "session_cap_blocks": 1,
            "archetype_load_blocks": 0,
            "trigger_load_blocks": 2,
            "symbol_cap_blocks": 0,
            "risk_budget": {"used_pct": 10.0},
        },
        {
            "limit_stats": {"risk_block_breakdown": {"archetype_load": 3, "daily_cap": 1}},
            "daily_loss_blocks": 0,
            "daily_cap_blocks": 1,
            "risk_budget_blocks": 0,
            "session_cap_blocks": 0,
            "archetype_load_blocks": 3,
            "trigger_load_blocks": 0,
            "symbol_cap_blocks": 0,
            "risk_budget": {"used_pct": 20.0},
        },
    ]
    summary = build_run_summary(daily_reports)
    blocks = summary["block_totals"]
    assert blocks["session_cap"] == 1
    assert blocks["trigger_load"] == 2
    assert blocks["archetype_load"] == 3
    assert blocks["daily_cap"] == 1

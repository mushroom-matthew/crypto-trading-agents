"""Tests that trigger_quality load counts are not duplicated in aggregation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.reports import build_run_summary


def test_trigger_quality_load_counts_not_duplicated() -> None:
    # Build a minimal daily report with known load_count.
    daily = {
        "limit_stats": {},
        "risk_budget": {"used_pct": 0.0, "utilization_pct": 0.0},
        "trigger_quality": {
            "t1": {
                "pnl": 0.0,
                "risk_used_abs": 0.0,
                "trades": 1,
                "wins": 0,
                "losses": 1,
                "latency_seconds": 0.0,
                "mae_pct": 0.0,
                "mfe_pct": 0.0,
                "response_decay_pct": 0.0,
                "load_count": 2,
                "avg_load": 5.0,
            }
        },
    }
    summary = build_run_summary([daily])
    tq = summary["trigger_quality"]["t1"]
    # load_count should match input (2), not 2x/4x due to duplication.
    assert tq["load_count"] == 2
    assert tq["load_sum"] == 10.0

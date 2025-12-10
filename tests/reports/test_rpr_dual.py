"""Dual RPR (allocated vs actual) aggregation test."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.reports import build_run_summary


def test_dual_rpr_matches_expected() -> None:
    daily = {
        "limit_stats": {},
        "risk_budget": {"used_pct": 0.0, "utilization_pct": 0.0},
        "trigger_quality": {
            "t1": {
                "pnl": 10.0,
                "risk_used_abs": 5.0,
                "actual_risk_abs": 4.0,
                "trades": 1,
                "wins": 1,
                "losses": 0,
                "latency_seconds": 0.0,
                "mae_pct": 0.0,
                "mfe_pct": 0.0,
                "response_decay_pct": 0.0,
                "load_count": 0,
                "avg_load": 0.0,
            }
        },
    }
    summary = build_run_summary([daily])
    tq = summary["trigger_quality"]["t1"]
    assert tq["rpr_allocated"] == 2.0  # 10 / 5
    assert tq["rpr_actual"] == 2.5  # 10 / 4

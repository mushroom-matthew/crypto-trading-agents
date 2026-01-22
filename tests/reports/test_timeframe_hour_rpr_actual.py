import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtesting.reports as reports


def test_timeframe_and_hour_rpr_actual_non_zero():
    daily_report = {
        "equity_return_pct": 0.0,
        "trade_count": 1,
        "risk_budget": {"used_pct": 0.0, "utilization_pct": 0.0},
        "limit_stats": {"risk_block_breakdown": {}},
        "timeframe_quality": {
            "1h": {
                "trades": 1,
                "pnl": 5.0,
                "risk_used_abs": 10.0,
                "actual_risk_abs": 5.0,
            }
        },
        "hour_quality": {
            "3": {
                "trades": 1,
                "pnl": 5.0,
                "risk_used_abs": 10.0,
                "actual_risk_abs": 5.0,
            }
        },
    }

    summary = reports.build_run_summary([daily_report])

    tf_rpr_actual = summary["timeframe_quality"]["1h"]["rpr_actual"]
    hr_rpr_actual = summary["hour_quality"]["3"]["rpr_actual"]

    assert tf_rpr_actual == 1.0
    assert hr_rpr_actual == 1.0

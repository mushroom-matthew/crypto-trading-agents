from __future__ import annotations

import pytest

from tools.paper_trading import PaperTradingWorkflow, _compute_trigger_preview


def test_compute_trigger_preview_projects_stop_target_and_rr_for_pct_trigger():
    trigger = {
        "id": "btc_long_rr2",
        "symbol": "BTC-USD",
        "category": "trend_continuation",
        "direction": "long",
        "timeframe": "1h",
        "entry_rule": "close > ema_50",
        "exit_rule": "target_hit or stop_hit",
        "hold_rule": "",
        "stop_anchor_type": "pct",
        "stop_loss_pct": 2.0,
        "target_anchor_type": "r_multiple_2",
    }
    indicator = {
        "symbol": "BTC-USD",
        "timeframe": "1h",
        "as_of": "2026-04-21T12:00:00Z",
        "close": 100.0,
        "atr_14": 2.0,
    }

    preview = _compute_trigger_preview(trigger, indicator, entry_reference_price=100.0)

    assert preview["entry_reference_price"] == pytest.approx(100.0)
    assert preview["stop_price"] == pytest.approx(98.0)
    assert preview["target_price"] == pytest.approx(104.0)
    assert preview["rr_ratio"] == pytest.approx(2.0)


def test_get_current_plan_enriches_triggers_with_preview_prices():
    workflow = PaperTradingWorkflow()
    workflow.current_plan = {
        "plan_id": "plan_test",
        "generated_at": "2026-04-21T12:00:00Z",
        "valid_until": "2026-04-21T16:00:00Z",
        "regime": "range",
        "triggers": [
            {
                "id": "eth_long_rr2",
                "symbol": "ETH-USD",
                "category": "mean_reversion",
                "direction": "long",
                "timeframe": "1h",
                "entry_rule": "rsi_14 < 35",
                "exit_rule": "target_hit or stop_hit",
                "hold_rule": "",
                "stop_anchor_type": "pct",
                "stop_loss_pct": 1.5,
                "target_anchor_type": "r_multiple_2",
            }
        ],
    }
    workflow.last_indicators = {
        "ETH-USD": {
            "symbol": "ETH-USD",
            "timeframe": "1h",
            "as_of": "2026-04-21T12:00:00Z",
            "close": 2000.0,
            "atr_14": 25.0,
        }
    }
    workflow.last_known_prices = {"ETH-USD": 2000.0}

    current_plan = workflow.get_current_plan()
    assert current_plan is not None

    trigger = current_plan["triggers"][0]
    assert trigger["stop_price"] == pytest.approx(1970.0)
    assert trigger["target_price"] == pytest.approx(2060.0)
    assert trigger["rr_ratio"] == pytest.approx(2.0)

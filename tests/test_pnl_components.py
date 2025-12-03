from datetime import datetime, timezone

from backtesting.llm_strategist_runner import LLMStrategistBacktester


class _DummyPortfolio:
    def __init__(self, trade_log, fills):
        self.trade_log = trade_log
        self.fills = fills


def test_daily_costs_components_match_equity():
    day_key = "2024-01-01"
    start_equity = 1000.0
    # Components: +10 non-flatten, +5 flatten, -2 fees = +13 net
    end_equity = start_equity + 13.0
    trade_log = [
        {"timestamp": datetime(2024, 1, 1, 1, tzinfo=timezone.utc), "symbol": "BTC-USD", "pnl": 10.0, "reason": "entry_exit"},
        {"timestamp": datetime(2024, 1, 1, 23, tzinfo=timezone.utc), "symbol": "BTC-USD", "pnl": 5.0, "reason": "eod_flatten"},
    ]
    fills = [
        {"timestamp": datetime(2024, 1, 1, 1, tzinfo=timezone.utc), "symbol": "BTC-USD", "fee": 1.0},
        {"timestamp": datetime(2024, 1, 1, 23, tzinfo=timezone.utc), "symbol": "BTC-USD", "fee": 1.0},
    ]

    backtester = object.__new__(LLMStrategistBacktester)
    backtester.portfolio = _DummyPortfolio(trade_log=trade_log, fills=fills)

    breakdown, symbol_pnl, totals = backtester._daily_costs(day_key, start_equity, end_equity)
    expected_return_pct = (end_equity / start_equity - 1) * 100.0
    component_sum = (
        breakdown["gross_trade_pct"]
        + breakdown["flattening_pct"]
        + breakdown["fees_pct"]
        + breakdown.get("carryover_pct", 0.0)
    )

    assert abs(breakdown["net_equity_pct"] - expected_return_pct) < 1e-9
    assert abs(component_sum - expected_return_pct) < 1e-9
    assert abs(breakdown["net_equity_pct_delta"]) < 1e-9
    assert totals["realized_pnl_abs"] == 10.0
    assert totals["flattening_pnl_abs"] == 5.0
    assert totals["fees_abs"] == 2.0
    assert totals["carryover_pnl_abs"] == 0.0
    per_symbol = symbol_pnl["BTC-USD"]
    assert round(per_symbol["net_pct"], 10) == round(expected_return_pct, 10)

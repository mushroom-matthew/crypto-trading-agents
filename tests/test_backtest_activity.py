from unittest.mock import MagicMock

import pandas as pd

from backtesting.backtest_activity import BacktestRequest, run_backtest_activity


class _FakeResult:
    def __init__(self) -> None:
        self.equity_curve = pd.Series([1000.0, 1010.0])
        self.fills = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "symbol": "BTC-USD",
                    "side": "buy",
                    "qty": 0.1,
                    "price": 100.0,
                    "fee": 0.0,
                    "reason": "test",
                }
            ]
        )
        self.plan_log = []
        self.summary = {"final_equity": 1010.0}
        self.llm_costs = {"num_llm_calls": 0}
        self.final_cash = 900.0
        self.final_positions = {"BTC-USD": 0.1}
        self.daily_reports = [
            {
                "date": "2024-01-01",
                "start_equity": 1000.0,
                "end_equity": 1010.0,
                "return_pct": 1.0,
                "equity_return_pct": 1.0,
                "gross_trade_return_pct": 1.0,
                "realized_pnl_abs": 10.0,
                "flattening_pnl_abs": 0.0,
                "fees_abs": 0.0,
                "carryover_pnl": 0.0,
                "daily_cash_flows": 0.0,
                "attempted_triggers": 1,
                "executed_trades": 1,
                "flatten_positions_daily": False,
                "pnl_breakdown": {
                    "gross_trade_pct": 1.0,
                    "fees_pct": 0.0,
                    "flattening_pct": 0.0,
                    "carryover_pct": 0.0,
                    "component_net_pct": 1.0,
                    "net_equity_pct": 1.0,
                    "net_equity_pct_delta": 0.0,
                },
                "symbol_pnl": {"BTC-USD": {"gross_pct": 1.0, "net_pct": 1.0, "flattening_pct": 0.0, "fees_pct": 0.0}},
                "overnight_exposure": {},
                "limit_stats": {"blocked_by_daily_cap": 0, "blocked_by_risk_limits": 0, "blocked_details": [], "executed_details": []},
                "judge_feedback": {"score": 55.0},
            }
        ]


class _FakeBacktester:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def run(self, run_id: str) -> _FakeResult:  # noqa: D401
        return _FakeResult()


def test_run_backtest_activity(monkeypatch):
    candles = [
        {"timestamp": f"2024-01-01T0{i}:00:00Z", "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100 + i, "volume": 1000}
        for i in range(5)
    ]
    monkeypatch.setattr("backtesting.backtest_activity.LLMStrategistBacktester", _FakeBacktester)
    request = BacktestRequest(symbol="BTC-USD", timeframe="1h", candles=candles, initial_cash=5000)
    response = run_backtest_activity(request.model_dump())
    assert response["symbol"] == "BTC-USD"
    assert response["num_trades"] == 1
    assert response["final_positions"]["BTC-USD"] == 0.1

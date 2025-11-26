from unittest.mock import MagicMock

from backtesting.backtest_activity import run_backtest_activity, BacktestRequest
from tests.test_llm_backtest_engine import _make_plan


def test_run_backtest_activity_uses_llm(monkeypatch):
    candles = [
        {"open": 100 + (i % 5), "high": 101 + (i % 5), "low": 99 + (i % 5), "close": 100 + (i % 5), "volume": 1000 + i}
        for i in range(80)
    ]
    fake_plan = _make_plan()
    fake_plan.lookback = fake_plan.lookback.model_copy(update={"preferred_bars": 40, "min_bars": 30, "max_bars": 60})
    fake_client = MagicMock()
    fake_client.responses.create.return_value = MagicMock(
        output=[MagicMock(content=[MagicMock(text=fake_plan.model_dump_json())])]
    )
    monkeypatch.setattr("backtesting.backtest_activity._get_llm_client", lambda: fake_client)
    request = BacktestRequest(symbol="BTC-USD", timeframe="1h", candles=candles, initial_cash=5000)
    response = run_backtest_activity(request.model_dump())
    assert response["symbol"] == "BTC-USD"
    assert response["final_cash"] <= 5000

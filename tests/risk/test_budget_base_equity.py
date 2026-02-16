"""Tests that daily budget is based on start-of-day equity and remains constant intraday."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.llm_strategist_runner import LLMStrategistBacktester
from agents.strategies.trigger_engine import Order


def _bt_stub(start_equity: float = 1000.0) -> LLMStrategistBacktester:
    bt = LLMStrategistBacktester.__new__(LLMStrategistBacktester)  # type: ignore
    bt.daily_risk_budget_pct = 10.0
    bt.daily_risk_budget_state = {}
    bt.latest_daily_summary = None
    bt.initial_cash = start_equity
    bt.portfolio = type("PF", (), {"equity_records": [], "portfolio_state": lambda self, ts: None})()
    return bt


def test_budget_abs_uses_start_of_day_equity() -> None:
    bt = _bt_stub(start_equity=2000.0)
    bt._reset_risk_budget_for_day("2021-01-01", start_equity=2000.0)
    entry = bt.daily_risk_budget_state["2021-01-01"]
    assert entry["budget_abs"] == pytest.approx(200.0)  # 10% of 2000
    # Intraday equity move should not change budget_abs
    bt.daily_risk_budget_state["2021-01-01"]["start_equity"] = 1500.0  # simulate equity drift; budget_abs unchanged
    assert bt.daily_risk_budget_state["2021-01-01"]["budget_abs"] == pytest.approx(200.0)


def test_first_trade_starts_at_zero_usage() -> None:
    bt = _bt_stub(start_equity=1000.0)
    bt._reset_risk_budget_for_day("2021-01-02", start_equity=1000.0)
    remaining = bt._risk_budget_gate(
        "2021-01-02",
        Order(symbol="BTC", side="buy", quantity=1.0, price=10.0, timeframe="1h", reason="t", timestamp=datetime.utcnow()),
    )
    assert remaining is not None
    assert bt.daily_risk_budget_state["2021-01-02"]["used_abs"] == 0.0

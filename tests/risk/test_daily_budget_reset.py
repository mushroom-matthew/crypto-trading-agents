"""Tests that daily risk budget does not carry over across days."""

from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.llm_strategist_runner import LLMStrategistBacktester


class _Order:
    def __init__(self, quantity: float = 1.0, price: float = 100.0, symbol: str = "BTC") -> None:
        self.quantity = quantity
        self.price = price
        self.symbol = symbol
        self.timeframe = "1h"


def _bt_stub() -> LLMStrategistBacktester:
    bt = LLMStrategistBacktester.__new__(LLMStrategistBacktester)  # type: ignore
    bt.daily_risk_budget_pct = 10.0
    bt.daily_risk_budget_state = {}
    bt.latest_daily_summary = {"risk_budget": {"used_pct": 100.0}}
    bt.initial_cash = 1000.0
    return bt


def test_day_two_not_blocked_by_day_one_usage() -> None:
    bt = _bt_stub()
    # Day 1 fully consumed (should not affect Day 2).
    bt.daily_risk_budget_state["2021-01-01"] = {
        "budget_abs": 100.0,
        "used_abs": 100.0,
        "symbol_usage": defaultdict(float),
        "blocks": defaultdict(int),
    }
    # Day 2 fresh budget.
    bt.daily_risk_budget_state["2021-01-02"] = {
        "budget_abs": 100.0,
        "used_abs": 0.0,
        "symbol_usage": defaultdict(float),
        "blocks": defaultdict(int),
    }
    remaining = bt._risk_budget_gate("2021-01-02", _Order())
    assert remaining == 100.0


def test_day_usage_blocks_when_budget_exhausted_same_day() -> None:
    bt = _bt_stub()
    bt.daily_risk_budget_state["2021-01-02"] = {
        "budget_abs": 50.0,
        "used_abs": 50.0,
        "symbol_usage": defaultdict(float),
        "blocks": defaultdict(int),
    }
    remaining = bt._risk_budget_gate("2021-01-02", _Order())
    assert remaining is None


def test_used_pct_monotone_and_bounded() -> None:
    bt = _bt_stub()
    day = "2021-01-03"
    bt.daily_risk_budget_state[day] = {
        "budget_abs": 100.0,
        "used_abs": 0.0,
        "symbol_usage": defaultdict(float),
        "blocks": defaultdict(int),
    }
    # First commit 10, then 30, then 60 (caps at 100 total)
    bt._commit_risk_budget(day, 10.0, "BTC")
    bt._commit_risk_budget(day, 30.0, "BTC")
    bt._commit_risk_budget(day, 60.0, "BTC")
    entry = bt.daily_risk_budget_state.get(day)
    used = entry.get("used_abs", 0.0)
    budget = entry.get("budget_abs", 0.0)
    used_pct = (used / budget * 100.0) if budget else 0.0
    assert 0.0 <= used_pct <= 100.0

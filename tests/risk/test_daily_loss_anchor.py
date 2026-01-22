"""Tests for daily loss anchoring behavior (max_daily_loss_pct)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.llm_strategist_runner import LLMStrategistBacktester


def _new_backtester_stub() -> LLMStrategistBacktester:
    # Bypass __init__; we only need the anchoring helper and its backing dict.
    bt = LLMStrategistBacktester.__new__(LLMStrategistBacktester)  # type: ignore
    bt.daily_loss_anchor_by_day = {}
    return bt


def test_anchor_not_reset_intraday() -> None:
    bt = _new_backtester_stub()
    bt._set_daily_loss_anchor("2021-01-01", 1000.0)
    bt._set_daily_loss_anchor("2021-01-01", 900.0)  # should be ignored
    assert bt.daily_loss_anchor_by_day["2021-01-01"] == 1000.0


def test_anchor_resets_on_new_day() -> None:
    bt = _new_backtester_stub()
    bt._set_daily_loss_anchor("2021-01-01", 1000.0)
    bt._set_daily_loss_anchor("2021-01-02", 2000.0)
    assert bt.daily_loss_anchor_by_day["2021-01-01"] == 1000.0
    assert bt.daily_loss_anchor_by_day["2021-01-02"] == 2000.0

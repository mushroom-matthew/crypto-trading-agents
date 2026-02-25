"""Tests for Runbook 57: Screener Timeframe Threading.

Verifies that indicator_timeframe flows from PaperTradingConfig → SessionState →
fetch_indicator_snapshots_activity → generate_strategy_plan_activity → LLM prompt.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.paper_trading import (
    PaperTradingConfig,
    SessionState,
    generate_strategy_plan_activity,
)


# ---------------------------------------------------------------------------
# Schema / field tests
# ---------------------------------------------------------------------------

def test_paper_trading_config_defaults_to_1h():
    """PaperTradingConfig.indicator_timeframe defaults to '1h'."""
    config = PaperTradingConfig(
        session_id="s1",
        symbols=["BTC-USD"],
    )
    assert config.indicator_timeframe == "1h"


def test_paper_trading_config_accepts_1m():
    """PaperTradingConfig.indicator_timeframe='1m' validates without error."""
    config = PaperTradingConfig(
        session_id="s1",
        symbols=["BTC-USD"],
        indicator_timeframe="1m",
    )
    assert config.indicator_timeframe == "1m"


def test_paper_trading_config_accepts_all_valid_timeframes():
    """PaperTradingConfig.indicator_timeframe accepts all documented valid values."""
    for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
        config = PaperTradingConfig(
            session_id="s1",
            symbols=["BTC-USD"],
            indicator_timeframe=tf,
        )
        assert config.indicator_timeframe == tf


def test_session_state_defaults_to_1h():
    """SessionState.indicator_timeframe defaults to '1h' (backwards compat)."""
    state = SessionState(
        session_id="s1",
        symbols=["BTC-USD"],
        strategy_prompt=None,
        plan_interval_hours=4.0,
    )
    assert state.indicator_timeframe == "1h"


def test_session_state_roundtrips_indicator_timeframe():
    """SessionState serializes and deserializes indicator_timeframe correctly."""
    state = SessionState(
        session_id="s1",
        symbols=["BTC-USD"],
        strategy_prompt=None,
        plan_interval_hours=4.0,
        indicator_timeframe="15m",
    )
    dumped = state.model_dump()
    restored = SessionState.model_validate(dumped)
    assert restored.indicator_timeframe == "15m"


# ---------------------------------------------------------------------------
# Activity tests — TIMEFRAME hint injection and snap_init
# ---------------------------------------------------------------------------

def _make_plan_stub():
    """Return a StrategyPlan-like stub (just needs model_dump() → dict)."""
    stub = MagicMock()
    stub.model_dump.return_value = {
        "plan_id": "test-plan-id",
        "triggers": [],
        "max_trades_per_day": 2,
    }
    return stub


@pytest.mark.asyncio
async def test_generate_plan_activity_injects_timeframe_hint():
    """generate_strategy_plan_activity injects TIMEFRAME: block when indicator_timeframe='1m'."""
    captured_prompts = []

    def fake_generate_plan(llm_input, prompt_template=None, run_id=None, plan_id=None):
        captured_prompts.append(prompt_template)
        return _make_plan_stub()

    fake_client = MagicMock()
    fake_client.generate_plan.side_effect = fake_generate_plan
    fake_client.last_generation_info = {}

    with patch("tools.paper_trading.LLMClient", return_value=fake_client):
        with patch("tools.paper_trading.PLAN_CACHE_DIR") as mock_cache_dir:
            mock_cache_dir.mkdir = MagicMock()
            mock_cache_file = MagicMock()
            mock_cache_file.exists.return_value = False
            mock_cache_file.write_text = MagicMock()
            mock_cache_dir.__truediv__ = MagicMock(return_value=mock_cache_file)

            await generate_strategy_plan_activity(
                symbols=["BTC-USD"],
                portfolio_state={"cash": 10000, "positions": {}, "total_equity": 10000},
                strategy_prompt="trade carefully",
                market_context={"BTC-USD": {"price": 50000.0}},
                indicator_timeframe="1m",
            )

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "TIMEFRAME:" in prompt
    assert "'1m'" in prompt


@pytest.mark.asyncio
async def test_generate_plan_activity_snap_init_uses_timeframe():
    """snap_init uses indicator_timeframe as the timeframe fallback, not hardcoded '1h'."""
    captured_assets = []

    def fake_generate_plan(llm_input, prompt_template=None, run_id=None, plan_id=None):
        captured_assets.extend(llm_input.assets)
        return _make_plan_stub()

    fake_client = MagicMock()
    fake_client.generate_plan.side_effect = fake_generate_plan
    fake_client.last_generation_info = {}

    with patch("tools.paper_trading.LLMClient", return_value=fake_client):
        with patch("tools.paper_trading.PLAN_CACHE_DIR") as mock_cache_dir:
            mock_cache_dir.mkdir = MagicMock()
            mock_cache_file = MagicMock()
            mock_cache_file.exists.return_value = False
            mock_cache_file.write_text = MagicMock()
            mock_cache_dir.__truediv__ = MagicMock(return_value=mock_cache_file)

            await generate_strategy_plan_activity(
                symbols=["ETH-USD"],
                portfolio_state={"cash": 5000, "positions": {}, "total_equity": 5000},
                strategy_prompt=None,
                # Minimal context — triggers the snap_init fallback path
                market_context={"ETH-USD": {"price": 3000.0}},
                indicator_timeframe="5m",
            )

    # The minimal fallback path creates an IndicatorSnapshot with the given timeframe
    assert len(captured_assets) == 1
    asset = captured_assets[0]
    assert asset.indicators[0].timeframe == "5m"


@pytest.mark.asyncio
async def test_generate_plan_activity_defaults_to_1h_when_no_timeframe():
    """When indicator_timeframe is None, snap_init falls back to '1h'."""
    captured_assets = []

    def fake_generate_plan(llm_input, prompt_template=None, run_id=None, plan_id=None):
        captured_assets.extend(llm_input.assets)
        return _make_plan_stub()

    fake_client = MagicMock()
    fake_client.generate_plan.side_effect = fake_generate_plan
    fake_client.last_generation_info = {}

    with patch("tools.paper_trading.LLMClient", return_value=fake_client):
        with patch("tools.paper_trading.PLAN_CACHE_DIR") as mock_cache_dir:
            mock_cache_dir.mkdir = MagicMock()
            mock_cache_file = MagicMock()
            mock_cache_file.exists.return_value = False
            mock_cache_file.write_text = MagicMock()
            mock_cache_dir.__truediv__ = MagicMock(return_value=mock_cache_file)

            await generate_strategy_plan_activity(
                symbols=["BTC-USD"],
                portfolio_state={"cash": 10000, "positions": {}, "total_equity": 10000},
                strategy_prompt=None,
                market_context={"BTC-USD": {"price": 50000.0}},
                indicator_timeframe=None,  # explicit None → should fall back to 1h
            )

    assert len(captured_assets) == 1
    asset = captured_assets[0]
    assert asset.indicators[0].timeframe == "1h"

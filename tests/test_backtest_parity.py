"""Tests for Runbook 67: Backtest-Paper Trading Parity.

Verifies that:
- StrategistBacktestResult carries all R67 parity fields.
- LLMStrategistBacktester initialises R67 state fields.
- _build_results_payload() surfaces R67 counters at the top level.
- _build_results_payload() handles missing keys gracefully (defaults to 0).
- episode_records are passed through from llm_data to the payload.
- _r67_build_episode_on_close produces a valid episode dict without crashing.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_min_result(**overrides: Any):
    """Return a minimal StrategistBacktestResult instance."""
    from backtesting.llm_strategist_runner import StrategistBacktestResult

    defaults: Dict[str, Any] = dict(
        equity_curve=pd.Series(dtype=float),
        fills=pd.DataFrame(),
        plan_log=[],
        summary={},
        llm_costs={},
        final_cash=10000.0,
        final_positions={},
        daily_reports=[],
        bar_decisions={},
    )
    defaults.update(overrides)
    return StrategistBacktestResult(**defaults)


def _build_payload(llm_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call persistence._build_results_payload with a minimal results dict."""
    from backtesting.persistence import _build_results_payload

    results = {
        "final_equity": 10000.0,
        "equity_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 1.0,
        "llm_data": llm_data,
    }
    return _build_results_payload(results)


# ---------------------------------------------------------------------------
# 1. StrategistBacktestResult — R67 parity fields exist
# ---------------------------------------------------------------------------


class TestStrategyBacktestResultParityFields:
    def test_episode_records_field_exists(self) -> None:
        field_names = {f.name for f in dc_fields(_make_min_result().__class__)}
        assert "episode_records" in field_names

    def test_exit_binding_mismatch_blocked_field_exists(self) -> None:
        field_names = {f.name for f in dc_fields(_make_min_result().__class__)}
        assert "exit_binding_mismatch_blocked" in field_names

    def test_validation_rejected_count_field_exists(self) -> None:
        field_names = {f.name for f in dc_fields(_make_min_result().__class__)}
        assert "validation_rejected_count" in field_names

    def test_policy_loop_skip_count_field_exists(self) -> None:
        field_names = {f.name for f in dc_fields(_make_min_result().__class__)}
        assert "policy_loop_skip_count" in field_names

    def test_episode_records_defaults_to_empty_list(self) -> None:
        result = _make_min_result()
        assert result.episode_records == []

    def test_exit_binding_mismatch_blocked_defaults_to_zero(self) -> None:
        result = _make_min_result()
        assert result.exit_binding_mismatch_blocked == 0

    def test_validation_rejected_count_defaults_to_zero(self) -> None:
        result = _make_min_result()
        assert result.validation_rejected_count == 0

    def test_policy_loop_skip_count_defaults_to_zero(self) -> None:
        result = _make_min_result()
        assert result.policy_loop_skip_count == 0

    def test_episode_records_accepts_list_of_dicts(self) -> None:
        ep = {"symbol": "BTC-USD", "outcome": "win"}
        result = _make_min_result(episode_records=[ep])
        assert result.episode_records == [ep]

    def test_counter_fields_accept_nonzero_values(self) -> None:
        result = _make_min_result(
            exit_binding_mismatch_blocked=3,
            validation_rejected_count=2,
            policy_loop_skip_count=7,
        )
        assert result.exit_binding_mismatch_blocked == 3
        assert result.validation_rejected_count == 2
        assert result.policy_loop_skip_count == 7


# ---------------------------------------------------------------------------
# 2. LLMStrategistBacktester — R67 state fields initialised
# ---------------------------------------------------------------------------


class TestBacktesterParityStateFields:
    def _make_backtester(self):
        from backtesting.llm_strategist_runner import LLMStrategistBacktester
        from backtesting.llm_shim import make_strategist_shim_transport
        from agents.strategies.llm_client import LLMClient
        from pathlib import Path

        llm_client = LLMClient(
            transport=make_strategist_shim_transport(),
            model="shim",
            allow_fallback=False,
        )
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)
        bt = LLMStrategistBacktester(
            pairs=["BTC-USD"],
            start=start,
            end=end,
            initial_cash=10000.0,
            fee_rate=0.001,
            llm_client=llm_client,
            cache_dir=Path("/tmp/r67-parity-test"),
            llm_calls_per_day=2,
        )
        return bt

    def test_episode_records_initialised(self) -> None:
        bt = self._make_backtester()
        assert hasattr(bt, "_episode_records")
        assert isinstance(bt._episode_records, list)
        assert bt._episode_records == []

    def test_validation_rejected_count_initialised(self) -> None:
        bt = self._make_backtester()
        assert hasattr(bt, "_validation_rejected_count")
        assert bt._validation_rejected_count == 0

    def test_policy_loop_skip_count_initialised(self) -> None:
        bt = self._make_backtester()
        assert hasattr(bt, "_policy_loop_skip_count")
        assert bt._policy_loop_skip_count == 0

    def test_exit_binding_mismatch_blocked_initialised(self) -> None:
        bt = self._make_backtester()
        assert hasattr(bt, "_exit_binding_mismatch_blocked")
        assert bt._exit_binding_mismatch_blocked == 0

    def test_position_originating_plans_initialised(self) -> None:
        bt = self._make_backtester()
        assert hasattr(bt, "_position_originating_plans")
        assert isinstance(bt._position_originating_plans, dict)

    def test_amt_state_initialised(self) -> None:
        bt = self._make_backtester()
        assert hasattr(bt, "_amt_state")
        assert isinstance(bt._amt_state, dict)


# ---------------------------------------------------------------------------
# 3. _build_results_payload — R67 counters surfaced at top level
# ---------------------------------------------------------------------------


class TestBuildResultsPayloadParityFields:
    def test_episode_count_surfaced_from_llm_data(self) -> None:
        payload = _build_payload({
            "episode_records": [{"ep": 1}, {"ep": 2}],
        })
        assert payload["episode_count"] == 2

    def test_episode_count_zero_when_empty_list(self) -> None:
        payload = _build_payload({"episode_records": []})
        assert payload["episode_count"] == 0

    def test_episode_count_zero_when_key_missing(self) -> None:
        payload = _build_payload({})
        assert payload["episode_count"] == 0

    def test_exit_binding_mismatch_blocked_surfaced(self) -> None:
        payload = _build_payload({"exit_binding_mismatch_blocked": 5})
        assert payload["exit_binding_mismatch_blocked"] == 5

    def test_exit_binding_mismatch_blocked_defaults_zero(self) -> None:
        payload = _build_payload({})
        assert payload["exit_binding_mismatch_blocked"] == 0

    def test_validation_rejected_count_surfaced(self) -> None:
        payload = _build_payload({"validation_rejected_count": 3})
        assert payload["validation_rejected_count"] == 3

    def test_validation_rejected_count_defaults_zero(self) -> None:
        payload = _build_payload({})
        assert payload["validation_rejected_count"] == 0

    def test_policy_loop_skip_count_surfaced(self) -> None:
        payload = _build_payload({"policy_loop_skip_count": 12})
        assert payload["policy_loop_skip_count"] == 12

    def test_policy_loop_skip_count_defaults_zero(self) -> None:
        payload = _build_payload({})
        assert payload["policy_loop_skip_count"] == 0

    def test_preexisting_episode_count_not_overwritten(self) -> None:
        """If caller already set episode_count, don't clobber it."""
        from backtesting.persistence import _build_results_payload

        results = {
            "episode_count": 99,
            "llm_data": {"episode_records": [{"ep": 1}]},
        }
        payload = _build_results_payload(results)
        assert payload["episode_count"] == 99

    def test_preexisting_policy_loop_skip_count_not_overwritten(self) -> None:
        from backtesting.persistence import _build_results_payload

        results = {
            "policy_loop_skip_count": 42,
            "llm_data": {"policy_loop_skip_count": 7},
        }
        payload = _build_results_payload(results)
        assert payload["policy_loop_skip_count"] == 42


# ---------------------------------------------------------------------------
# 4. activities.py llm_data wiring
# ---------------------------------------------------------------------------


class TestActivitiesLlmDataWiring:
    """Verify that result.episode_records etc. are threaded into llm_data dict.

    We test the logic by inspecting the source code structure (import-time)
    rather than running a full backtest simulation.
    """

    def test_episode_records_key_present_in_llm_data(self) -> None:
        """Grep-style check that activities.py references episode_records in llm_data."""
        import ast
        from pathlib import Path

        source = (Path(__file__).parent.parent / "backtesting" / "activities.py").read_text()
        # The key 'episode_records' must appear in the file
        assert "episode_records" in source, (
            "backtesting/activities.py must include episode_records in llm_data (R67 Step 10)"
        )

    def test_exit_binding_mismatch_blocked_in_activities(self) -> None:
        from pathlib import Path

        source = (Path(__file__).parent.parent / "backtesting" / "activities.py").read_text()
        assert "exit_binding_mismatch_blocked" in source

    def test_validation_rejected_count_in_activities(self) -> None:
        from pathlib import Path

        source = (Path(__file__).parent.parent / "backtesting" / "activities.py").read_text()
        assert "validation_rejected_count" in source

    def test_policy_loop_skip_count_in_activities(self) -> None:
        from pathlib import Path

        source = (Path(__file__).parent.parent / "backtesting" / "activities.py").read_text()
        assert "policy_loop_skip_count" in source


# ---------------------------------------------------------------------------
# 5. _r67_build_episode_on_close — helper produces episode dict
# ---------------------------------------------------------------------------


def _make_real_backtester():
    """Create a minimal LLMStrategistBacktester for episode helper tests."""
    from backtesting.llm_strategist_runner import LLMStrategistBacktester
    from backtesting.llm_shim import make_strategist_shim_transport
    from agents.strategies.llm_client import LLMClient
    from pathlib import Path

    llm_client = LLMClient(
        transport=make_strategist_shim_transport(),
        model="shim",
        allow_fallback=False,
    )
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 2, tzinfo=timezone.utc)
    return LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=start,
        end=end,
        initial_cash=10000.0,
        fee_rate=0.001,
        llm_client=llm_client,
        cache_dir=Path("/tmp/r67-parity-test"),
        llm_calls_per_day=2,
    )


def _set_position_meta(bt: Any, symbol: str, meta: Dict[str, Any]) -> None:
    """Helper to set position_meta on the backtester's portfolio."""
    bt.portfolio.position_meta[symbol] = meta


def _make_order_ns(
    symbol: str = "BTC-USD",
    side: str = "sell",
    price: float = 52000.0,
    ts: Optional[datetime] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        side=side,
        price=price,
        quantity=0.1,
        reason="below_stop",
        timestamp=ts or datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# 5b. episode_source field and source-weight multiplier
# ---------------------------------------------------------------------------


class TestEpisodeSourceDistinction:
    """Verify episode_source field and retrieval weighting by source."""

    def test_episode_memory_record_has_episode_source(self) -> None:
        from schemas.episode_memory import EpisodeMemoryRecord

        rec = EpisodeMemoryRecord(
            episode_id="ep-1",
            symbol="BTC-USD",
            outcome_class="win",
        )
        assert rec.episode_source == "live"

    def test_episode_source_backtest(self) -> None:
        from schemas.episode_memory import EpisodeMemoryRecord

        rec = EpisodeMemoryRecord(
            episode_id="ep-2",
            symbol="BTC-USD",
            outcome_class="win",
            episode_source="backtest",
        )
        assert rec.episode_source == "backtest"

    def test_episode_source_paper(self) -> None:
        from schemas.episode_memory import EpisodeMemoryRecord

        rec = EpisodeMemoryRecord(
            episode_id="ep-3",
            symbol="BTC-USD",
            outcome_class="loss",
            episode_source="paper",
        )
        assert rec.episode_source == "paper"

    def test_backtest_episode_tagged_correctly(self) -> None:
        """Backtest runner sets episode_source='backtest' on built records."""
        bt = _make_real_backtester()
        _set_position_meta(bt, "BTC-USD", {
            "entry_price": 50000.0,
            "opened_at": datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            "stop_price_abs": 48000.0,
            "target_price_abs": 55000.0,
        })
        bt._r67_build_episode_on_close(_make_order_ns(price=52000.0), pre_qty=0.1)
        assert len(bt._episode_records) == 1
        ep = bt._episode_records[0]
        assert ep.get("episode_source") == "backtest"

    def test_source_weight_multipliers_in_retrieval_request(self) -> None:
        from schemas.episode_memory import MemoryRetrievalRequest, DEFAULT_SOURCE_WEIGHT_MULTIPLIERS

        req = MemoryRetrievalRequest(
            symbol="BTC-USD",
            regime_fingerprint={"rsi": 0.5},
        )
        assert req.source_weight_multipliers == DEFAULT_SOURCE_WEIGHT_MULTIPLIERS

    def test_backtest_score_lower_than_live(self) -> None:
        """A backtest episode scores lower than an otherwise identical live episode."""
        from schemas.episode_memory import EpisodeMemoryRecord, MemoryRetrievalRequest
        from services.memory_retrieval_service import _score
        from datetime import timezone

        request = MemoryRetrievalRequest(
            symbol="BTC-USD",
            regime_fingerprint={"rsi": 0.5, "trend": 0.8},
        )
        common = dict(
            episode_id="ep",
            symbol="BTC-USD",
            outcome_class="win",
            regime_fingerprint={"rsi": 0.5, "trend": 0.8},
            exit_ts=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        live_ep = EpisodeMemoryRecord(**common, episode_source="live")
        paper_ep = EpisodeMemoryRecord(**common, episode_source="paper")
        backtest_ep = EpisodeMemoryRecord(**common, episode_source="backtest")

        s_live = _score(live_ep, request)
        s_paper = _score(paper_ep, request)
        s_backtest = _score(backtest_ep, request)

        assert s_live > s_paper > s_backtest, (
            f"Expected live ({s_live:.4f}) > paper ({s_paper:.4f}) > backtest ({s_backtest:.4f})"
        )

    def test_source_weights_customisable(self) -> None:
        """Caller can equalise weights to treat all sources identically."""
        from schemas.episode_memory import EpisodeMemoryRecord, MemoryRetrievalRequest
        from services.memory_retrieval_service import _score
        from datetime import timezone

        request = MemoryRetrievalRequest(
            symbol="BTC-USD",
            regime_fingerprint={"rsi": 0.5},
            source_weight_multipliers={"live": 1.0, "paper": 1.0, "backtest": 1.0},
        )
        common = dict(
            episode_id="ep",
            symbol="BTC-USD",
            outcome_class="win",
            regime_fingerprint={"rsi": 0.5},
            exit_ts=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        s_live = _score(EpisodeMemoryRecord(**common, episode_source="live"), request)
        s_backtest = _score(EpisodeMemoryRecord(**common, episode_source="backtest"), request)
        assert abs(s_live - s_backtest) < 1e-9


class TestBuildEpisodeOnClose:
    """Tests for _r67_build_episode_on_close using the actual production method."""

    _LONG_META: Dict[str, Any] = {
        "entry_price": 50000.0,
        "opened_at": datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
        "stop_price_abs": 48000.0,
        "target_price_abs": 55000.0,
        "playbook_id": "rsi_extremes",
        "timeframe": "1h",
    }

    def test_episode_appended_on_close(self) -> None:
        bt = _make_real_backtester()
        _set_position_meta(bt, "BTC-USD", dict(self._LONG_META))
        bt._r67_build_episode_on_close(_make_order_ns(price=52000.0), pre_qty=0.1)
        assert len(bt._episode_records) == 1

    def test_episode_dict_has_symbol(self) -> None:
        bt = _make_real_backtester()
        _set_position_meta(bt, "BTC-USD", dict(self._LONG_META))
        bt._r67_build_episode_on_close(_make_order_ns(price=52000.0), pre_qty=0.1)
        ep = bt._episode_records[0]
        assert ep.get("symbol") == "BTC-USD"

    def test_episode_dict_has_outcome(self) -> None:
        bt = _make_real_backtester()
        _set_position_meta(bt, "BTC-USD", dict(self._LONG_META))
        bt._r67_build_episode_on_close(_make_order_ns(price=52000.0), pre_qty=0.1)
        ep = bt._episode_records[0]
        assert "outcome_class" in ep

    def test_profitable_trade_is_win(self) -> None:
        """Exit at 52000 > entry 50000 → win."""
        bt = _make_real_backtester()
        _set_position_meta(bt, "BTC-USD", dict(self._LONG_META))
        bt._r67_build_episode_on_close(_make_order_ns(price=52000.0), pre_qty=0.1)
        ep = bt._episode_records[0]
        assert ep["outcome_class"] == "win"

    def test_losing_trade_is_loss(self) -> None:
        """Exit at 47000 < entry 50000 → loss."""
        bt = _make_real_backtester()
        _set_position_meta(bt, "BTC-USD", dict(self._LONG_META))
        bt._r67_build_episode_on_close(_make_order_ns(price=47000.0), pre_qty=0.1)
        ep = bt._episode_records[0]
        assert ep["outcome_class"] == "loss"

    def test_no_crash_when_entry_price_missing(self) -> None:
        """Missing entry_price → skip (non-fatal), no episode appended."""
        bt = _make_real_backtester()
        _set_position_meta(bt, "BTC-USD", {})  # no entry_price
        bt._r67_build_episode_on_close(_make_order_ns(), pre_qty=0.1)
        assert len(bt._episode_records) == 0  # skipped gracefully

    def test_no_crash_when_meta_missing_entirely(self) -> None:
        """No portfolio meta at all → skip, no exception."""
        bt = _make_real_backtester()
        # Don't set any meta
        bt._r67_build_episode_on_close(_make_order_ns(), pre_qty=0.1)
        assert len(bt._episode_records) == 0

    def test_multiple_episodes_accumulate(self) -> None:
        bt = _make_real_backtester()
        _set_position_meta(bt, "BTC-USD", dict(self._LONG_META))
        bt._r67_build_episode_on_close(
            _make_order_ns(price=52000.0, ts=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)),
            pre_qty=0.1,
        )
        _set_position_meta(bt, "BTC-USD", {**self._LONG_META, "entry_price": 49000.0})
        bt._r67_build_episode_on_close(
            _make_order_ns(price=48000.0, ts=datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc)),
            pre_qty=0.1,
        )
        assert len(bt._episode_records) == 2


# ---------------------------------------------------------------------------
# TestShimTriggerValidity — verifies shim plans pass StrategyPlan validation
# ---------------------------------------------------------------------------


class TestShimTriggerValidity:
    """Regression test: shim triggers must pass TriggerCondition stop validation.

    Entry triggers (direction=long/short) require either stop_anchor_type or
    stop_loss_pct > 0.  This test catches any future regression where shim
    templates are changed without adding the required stop definition.
    """

    def test_shim_plan_is_valid_strategy_plan(self) -> None:
        """build_strategist_shim_plan should produce a plan that parses as StrategyPlan."""
        import json
        from backtesting.llm_shim import build_strategist_shim_plan
        from schemas.llm_strategist import StrategyPlan

        payload = json.dumps({
            "assets": [{"symbol": "BTC-USD"}],
            "global_context": {"regime": "bull"},
            "risk_params": {"max_position_risk_pct": 2.0},
        })
        plan_dict = build_strategist_shim_plan(payload)
        # Must not raise — previously failed with ValidationError (no stop defined)
        plan = StrategyPlan.model_validate(plan_dict)
        assert plan is not None

    def test_all_entry_triggers_have_stops(self) -> None:
        """Every long/short trigger in the shim has stop_loss_pct > 0."""
        import json
        from backtesting.llm_shim import build_strategist_shim_plan

        payload = json.dumps({"assets": [{"symbol": "BTC-USD"}]})
        plan_dict = build_strategist_shim_plan(payload)
        entry_triggers = [
            t for t in plan_dict.get("triggers", [])
            if t.get("direction") in {"long", "short"}
        ]
        assert entry_triggers, "Shim must produce at least one entry trigger"
        for t in entry_triggers:
            has_pct = (t.get("stop_loss_pct") or 0.0) > 0
            has_anchor = bool(t.get("stop_anchor_type"))
            assert has_pct or has_anchor, (
                f"Trigger '{t['id']}' (direction={t['direction']}) has no stop defined"
            )

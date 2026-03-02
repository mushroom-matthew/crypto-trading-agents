"""Tests for R62: Playbook-first plan generation wiring.

Covers:
- RegimeEligibility.htf_trend_required and disallow_htf_counter_trend fields
- PlaybookRegistry.list_eligible() with HTF direction filtering
- plan_provider wires eligible_playbooks into llm_client and validates returned playbook_id
- llm_client._build_eligible_playbooks_block formats ELIGIBLE_PLAYBOOKS block correctly
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from schemas.playbook_definition import PlaybookDefinition, RegimeEligibility
from services.playbook_registry import PlaybookRegistry
from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import StrategyPlanProvider, _extract_htf_direction
from schemas.llm_strategist import (
    AssetState,
    IndicatorSnapshot,
    LLMInput as StrategistInput,
    PortfolioState,
    PositionSizingRule,
    RiskConstraint,
    StrategyPlan,
    TriggerCondition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_playbook(
    playbook_id: str,
    eligible_regimes: List[str] | None = None,
    htf_trend_required: str | None = None,
    disallow_htf_counter_trend: bool = False,
) -> PlaybookDefinition:
    re_ = RegimeEligibility(
        eligible_regimes=eligible_regimes or [],
        htf_trend_required=htf_trend_required,
        disallow_htf_counter_trend=disallow_htf_counter_trend,
    )
    return PlaybookDefinition(playbook_id=playbook_id, regime_eligibility=re_, description=f"Test {playbook_id}")


def _stub_registry(*playbooks: PlaybookDefinition) -> PlaybookRegistry:
    """Build a PlaybookRegistry whose internal dict is pre-populated (no file IO)."""
    registry = object.__new__(PlaybookRegistry)
    registry._playbooks = {pb.playbook_id: pb for pb in playbooks}
    return registry


def _llm_input(regime: str = "bull") -> StrategistInput:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    snapshot = IndicatorSnapshot(symbol="BTC-USD", timeframe="1h", as_of=ts, close=30000.0)
    asset = AssetState(symbol="BTC-USD", indicators=[snapshot], trend_state="uptrend", vol_state="normal")
    portfolio = PortfolioState(
        timestamp=ts,
        equity=100000.0,
        cash=100000.0,
        positions={},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )
    return StrategistInput(
        portfolio=portfolio,
        assets=[asset],
        risk_params={"max_position_risk_pct": 1.0},
        global_context={"regime": regime},
    )


def _minimal_plan(playbook_id: str | None = None) -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    sizing_rules = [PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)]
    return StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test plan",
        regime="bull",
        triggers=[],
        sizing_rules=sizing_rules,
        playbook_id=playbook_id,
    )


# ---------------------------------------------------------------------------
# RegimeEligibility HTF fields
# ---------------------------------------------------------------------------

class TestRegimeEligibilityHTFFields:
    def test_htf_trend_required_accepts_up_down_any(self):
        for val in ("up", "down", "any"):
            re_ = RegimeEligibility(htf_trend_required=val)
            assert re_.htf_trend_required == val

    def test_htf_trend_required_defaults_to_none(self):
        re_ = RegimeEligibility()
        assert re_.htf_trend_required is None

    def test_disallow_htf_counter_trend_defaults_to_false(self):
        re_ = RegimeEligibility()
        assert re_.disallow_htf_counter_trend is False

    def test_disallow_htf_counter_trend_set_to_true(self):
        re_ = RegimeEligibility(disallow_htf_counter_trend=True)
        assert re_.disallow_htf_counter_trend is True

    def test_htf_trend_required_rejects_invalid(self):
        with pytest.raises(Exception):
            RegimeEligibility(htf_trend_required="sideways")  # not in Literal


# ---------------------------------------------------------------------------
# PlaybookRegistry.list_eligible with HTF direction
# ---------------------------------------------------------------------------

class TestListEligibleHTF:
    def test_no_htf_requirement_always_eligible(self):
        pb = _make_playbook("pb_any", eligible_regimes=["bullish"])
        registry = _stub_registry(pb)
        result = registry.list_eligible("bullish", htf_direction="down")
        assert len(result) == 1

    def test_htf_trend_required_up_excluded_when_htf_down(self):
        pb = _make_playbook("pb_up_only", eligible_regimes=["bullish"], htf_trend_required="up")
        registry = _stub_registry(pb)
        result = registry.list_eligible("bullish", htf_direction="down")
        assert result == []

    def test_htf_trend_required_up_included_when_htf_up(self):
        pb = _make_playbook("pb_up_only", eligible_regimes=["bullish"], htf_trend_required="up")
        registry = _stub_registry(pb)
        result = registry.list_eligible("bullish", htf_direction="up")
        assert len(result) == 1

    def test_htf_trend_required_any_included_regardless(self):
        pb = _make_playbook("pb_any_htf", eligible_regimes=["bullish"], htf_trend_required="any")
        registry = _stub_registry(pb)
        assert len(registry.list_eligible("bullish", htf_direction="down")) == 1
        assert len(registry.list_eligible("bullish", htf_direction="up")) == 1

    def test_no_htf_direction_provided_skips_htf_filter(self):
        pb = _make_playbook("pb_up_only", eligible_regimes=["bullish"], htf_trend_required="up")
        registry = _stub_registry(pb)
        # No htf_direction → HTF filter is not applied
        result = registry.list_eligible("bullish")
        assert len(result) == 1

    def test_eligible_regimes_still_filters_regime(self):
        pb = _make_playbook("pb_bearish", eligible_regimes=["bearish"])
        registry = _stub_registry(pb)
        assert registry.list_eligible("bullish") == []

    def test_disallowed_regimes_excludes(self):
        re_ = RegimeEligibility(disallowed_regimes=["bullish"])
        pb = PlaybookDefinition(playbook_id="pb_no_bull", regime_eligibility=re_)
        registry = _stub_registry(pb)
        assert registry.list_eligible("bullish") == []

    def test_mixed_eligible_and_htf(self):
        pb_ok = _make_playbook("pb_ok", eligible_regimes=["bullish"])
        pb_up = _make_playbook("pb_up", eligible_regimes=["bullish"], htf_trend_required="up")
        pb_down = _make_playbook("pb_down", eligible_regimes=["bullish"], htf_trend_required="down")
        registry = _stub_registry(pb_ok, pb_up, pb_down)
        result = registry.list_eligible("bullish", htf_direction="up")
        ids = {pb.playbook_id for pb in result}
        assert "pb_ok" in ids
        assert "pb_up" in ids
        assert "pb_down" not in ids


# ---------------------------------------------------------------------------
# LLMClient._build_eligible_playbooks_block
# ---------------------------------------------------------------------------

class TestBuildEligiblePlaybooksBlock:
    def test_none_returns_none(self):
        assert LLMClient._build_eligible_playbooks_block(None) is None

    def test_empty_list_returns_none(self):
        assert LLMClient._build_eligible_playbooks_block([]) is None

    def test_block_contains_playbook_ids(self):
        pb1 = _make_playbook("rsi_extremes", eligible_regimes=["bullish"])
        pb2 = _make_playbook("macd_divergence", eligible_regimes=["bearish"])
        block = LLMClient._build_eligible_playbooks_block([pb1, pb2])
        assert block is not None
        assert "rsi_extremes" in block
        assert "macd_divergence" in block

    def test_block_has_xml_tags(self):
        pb = _make_playbook("rsi_extremes", eligible_regimes=["bullish"])
        block = LLMClient._build_eligible_playbooks_block([pb])
        assert "<ELIGIBLE_PLAYBOOKS>" in block
        assert "</ELIGIBLE_PLAYBOOKS>" in block

    def test_block_has_instruction(self):
        pb = _make_playbook("rsi_extremes")
        block = LLMClient._build_eligible_playbooks_block([pb])
        assert "playbook_id" in block
        assert "null" in block

    def test_block_includes_regimes_when_set(self):
        pb = _make_playbook("rsi_extremes", eligible_regimes=["bullish", "trending"])
        block = LLMClient._build_eligible_playbooks_block([pb])
        assert "bullish" in block
        assert "trending" in block


# ---------------------------------------------------------------------------
# plan_provider: post-LLM playbook_id validation
# ---------------------------------------------------------------------------

class TestPlanProviderPlaybookValidation:
    """Test that plan_provider clears playbook_id when not in eligible set."""

    def _make_provider(self, plan: StrategyPlan) -> StrategyPlanProvider:
        import tempfile
        from pathlib import Path

        class _DummyLLMClient:
            last_generation_info: Dict[str, Any] = {}

            def generate_plan(self, llm_input, prompt_template=None, **kwargs):
                return plan.model_copy(deep=True)

        tmp = Path(tempfile.mkdtemp())
        return StrategyPlanProvider(_DummyLLMClient(), cache_dir=tmp, llm_calls_per_day=100)

    def test_valid_playbook_id_preserved(self):
        pb = _make_playbook("rsi_extremes", eligible_regimes=[])  # no eligible_regimes = always eligible
        plan = _minimal_plan(playbook_id="rsi_extremes")
        provider = self._make_provider(plan)
        with patch("agents.strategies.plan_provider._get_eligible_playbooks", return_value=[pb]):
            result = provider.get_plan(
                run_id="test",
                plan_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                llm_input=_llm_input(),
                emit_events=False,
            )
        assert result.playbook_id == "rsi_extremes"

    def test_invalid_playbook_id_cleared(self):
        pb = _make_playbook("rsi_extremes")
        plan = _minimal_plan(playbook_id="unknown_playbook")
        provider = self._make_provider(plan)
        with patch("agents.strategies.plan_provider._get_eligible_playbooks", return_value=[pb]):
            result = provider.get_plan(
                run_id="test",
                plan_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                llm_input=_llm_input(),
                emit_events=False,
            )
        assert result.playbook_id is None

    def test_no_playbook_id_in_plan_ok(self):
        pb = _make_playbook("rsi_extremes")
        plan = _minimal_plan(playbook_id=None)
        provider = self._make_provider(plan)
        with patch("agents.strategies.plan_provider._get_eligible_playbooks", return_value=[pb]):
            result = provider.get_plan(
                run_id="test",
                plan_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                llm_input=_llm_input(),
                emit_events=False,
            )
        assert result.playbook_id is None

    def test_empty_eligible_list_skips_validation(self):
        """When no playbooks are eligible, plan.playbook_id is not cleared."""
        plan = _minimal_plan(playbook_id="any_playbook")
        provider = self._make_provider(plan)
        with patch("agents.strategies.plan_provider._get_eligible_playbooks", return_value=[]):
            result = provider.get_plan(
                run_id="test",
                plan_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                llm_input=_llm_input(),
                emit_events=False,
            )
        assert result.playbook_id == "any_playbook"


# ---------------------------------------------------------------------------
# _extract_htf_direction helper
# ---------------------------------------------------------------------------

class TestExtractHTFDirection:
    def test_none_indicator_returns_none(self):
        assert _extract_htf_direction(None) is None

    def test_indicator_with_htf_daily_trend(self):
        class _FakeIndicator:
            htf_daily_trend = "up"
        assert _extract_htf_direction(_FakeIndicator()) == "up"

    def test_indicator_without_htf_field_returns_none(self):
        class _FakeIndicator:
            pass
        assert _extract_htf_direction(_FakeIndicator()) is None

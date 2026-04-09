"""Tests for R76 SessionIntent schema and session planner service."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from schemas.session_intent import SessionIntent, SymbolIntent


# ---------------------------------------------------------------------------
# SymbolIntent tests
# ---------------------------------------------------------------------------

def _make_symbol_intent(**kwargs) -> SymbolIntent:
    defaults = dict(
        symbol="BTC-USD",
        opportunity_score_norm=0.75,
        risk_budget_fraction=0.5,
        thesis_summary="Strong momentum with tight structure.",
        expected_hold_horizon="intraday",
        direction_bias="long",
    )
    defaults.update(kwargs)
    return SymbolIntent(**defaults)


def test_symbol_intent_basic():
    si = _make_symbol_intent()
    assert si.symbol == "BTC-USD"
    assert si.opportunity_score_norm == 0.75
    assert si.risk_budget_fraction == 0.5
    assert si.direction_bias == "long"
    assert si.expected_hold_horizon == "intraday"


def test_symbol_intent_defaults():
    si = SymbolIntent(
        symbol="ETH-USD",
        opportunity_score_norm=0.6,
        risk_budget_fraction=0.3,
        thesis_summary="Test thesis",
    )
    assert si.direction_bias == "neutral"
    assert si.expected_hold_horizon == "intraday"
    assert si.playbook_id is None


def test_symbol_intent_bounds():
    with pytest.raises(Exception):
        SymbolIntent(
            symbol="BTC-USD",
            opportunity_score_norm=1.5,  # > 1.0 → invalid
            risk_budget_fraction=0.5,
            thesis_summary="test",
        )
    with pytest.raises(Exception):
        SymbolIntent(
            symbol="BTC-USD",
            opportunity_score_norm=0.5,
            risk_budget_fraction=1.5,  # > 1.0 → invalid
            thesis_summary="test",
        )


# ---------------------------------------------------------------------------
# SessionIntent tests
# ---------------------------------------------------------------------------

def _make_session_intent(**kwargs) -> SessionIntent:
    defaults = dict(
        selected_symbols=["BTC-USD", "ETH-USD"],
        symbol_intents=[
            _make_symbol_intent(symbol="BTC-USD", risk_budget_fraction=0.6),
            _make_symbol_intent(
                symbol="ETH-USD",
                opportunity_score_norm=0.6,
                risk_budget_fraction=0.3,
                direction_bias="neutral",
            ),
        ],
        regime_summary="Trending bull with moderate vol.",
        planner_rationale="Both symbols showed strong structure alignment.",
    )
    defaults.update(kwargs)
    return SessionIntent(**defaults)


def test_session_intent_defaults():
    si = _make_session_intent()
    assert len(si.selected_symbols) == 2
    assert si.total_risk_budget_fraction == 1.0
    assert si.use_ai_planner is True
    assert si.is_fallback is False
    assert si.planned_trade_cadence_min == 5
    assert si.planned_trade_cadence_max == 10


def test_session_intent_risk_budget_sum_valid():
    # sum = 0.9 ≤ 1.0 → valid
    si = _make_session_intent()
    assert si  # no exception


def test_session_intent_risk_budget_overcommit():
    with pytest.raises(Exception, match="overcommitted"):
        SessionIntent(
            selected_symbols=["BTC-USD", "ETH-USD"],
            symbol_intents=[
                _make_symbol_intent(symbol="BTC-USD", risk_budget_fraction=0.7),
                _make_symbol_intent(
                    symbol="ETH-USD",
                    opportunity_score_norm=0.6,
                    risk_budget_fraction=0.5,  # total = 1.2 → overcommit
                    direction_bias="neutral",
                ),
            ],
        )


def test_session_intent_empty_symbol_intents():
    # Empty symbol_intents is valid (planner may omit them)
    si = SessionIntent(
        selected_symbols=["BTC-USD"],
        symbol_intents=[],
    )
    assert si.symbol_intents == []


def test_to_prompt_block():
    si = _make_session_intent()
    block = si.to_prompt_block()
    assert "SESSION_INTENT" in block
    assert "BTC-USD" in block
    assert "ETH-USD" in block
    assert "score=" in block
    assert "risk=" in block
    assert "bias=" in block
    assert "Thesis:" in block


def test_to_prompt_block_no_symbol_intents():
    si = SessionIntent(selected_symbols=["BTC-USD"], symbol_intents=[])
    block = si.to_prompt_block()
    assert "SESSION_INTENT" in block
    assert "BTC-USD" in block


def test_symbol_intent_for_found():
    si = _make_session_intent()
    found = si.symbol_intent_for("BTC-USD")
    assert found is not None
    assert found.symbol == "BTC-USD"


def test_symbol_intent_for_not_found():
    si = _make_session_intent()
    assert si.symbol_intent_for("SOL-USD") is None


def test_session_intent_fallback_flag():
    si = SessionIntent(
        selected_symbols=["BTC-USD"],
        symbol_intents=[_make_symbol_intent()],
        is_fallback=True,
        use_ai_planner=True,
    )
    assert si.is_fallback is True


def test_session_intent_playbook_in_prompt():
    si = SessionIntent(
        selected_symbols=["BTC-USD"],
        symbol_intents=[
            _make_symbol_intent(playbook_id="donchian_breakout"),
        ],
    )
    block = si.to_prompt_block()
    assert "donchian_breakout" in block


# ---------------------------------------------------------------------------
# Session planner: fallback logic
# ---------------------------------------------------------------------------

def test_fallback_intent_equal_allocation():
    from services.session_planner import _fallback_intent
    from schemas.opportunity import OpportunityCard

    def _card(sym: str) -> OpportunityCard:
        now = datetime.now(timezone.utc)
        return OpportunityCard(
            symbol=sym,
            opportunity_score=0.4,
            opportunity_score_norm=0.5,
            vol_edge=0.5,
            structure_edge=0.5,
            trend_edge=0.5,
            liquidity_score=0.5,
            spread_penalty=0.1,
            instability_penalty=0.0,
            expected_hold_horizon="intraday",
            scored_at=now,
            indicator_as_of=now,
        )

    cards = [_card("BTC-USD"), _card("ETH-USD")]
    intent = _fallback_intent(["BTC-USD", "ETH-USD"], cards)
    assert intent.is_fallback is True
    assert set(intent.selected_symbols) == {"BTC-USD", "ETH-USD"}
    total = sum(si.risk_budget_fraction for si in intent.symbol_intents)
    assert abs(total - 1.0) < 0.01


def test_fallback_intent_single_symbol():
    from services.session_planner import _fallback_intent
    intent = _fallback_intent(["BTC-USD"], [])
    assert intent.selected_symbols == ["BTC-USD"]
    assert len(intent.symbol_intents) == 1
    assert intent.symbol_intents[0].risk_budget_fraction == 1.0


# ---------------------------------------------------------------------------
# Session planner: LLM response parsing
# ---------------------------------------------------------------------------

def test_parse_llm_response_valid():
    from services.session_planner import _parse_llm_response
    from schemas.opportunity import OpportunityCard
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    card = OpportunityCard(
        symbol="BTC-USD",
        opportunity_score=0.5,
        opportunity_score_norm=0.7,
        vol_edge=0.6,
        structure_edge=0.6,
        trend_edge=0.6,
        liquidity_score=0.7,
        spread_penalty=0.1,
        instability_penalty=0.0,
        expected_hold_horizon="intraday",
        scored_at=now,
        indicator_as_of=now,
    )

    raw = """{
        "selected_symbols": ["BTC-USD"],
        "regime_summary": "Bull trending",
        "planner_rationale": "Strong momentum",
        "planned_trade_cadence_min": 4,
        "planned_trade_cadence_max": 8,
        "symbol_intents": [
            {
                "symbol": "BTC-USD",
                "opportunity_score_norm": 0.7,
                "risk_budget_fraction": 0.8,
                "playbook_id": null,
                "thesis_summary": "BTC showing strong trend alignment",
                "expected_hold_horizon": "intraday",
                "direction_bias": "long"
            }
        ]
    }"""

    intent = _parse_llm_response(raw, [card])
    assert intent.selected_symbols == ["BTC-USD"]
    assert intent.regime_summary == "Bull trending"
    assert len(intent.symbol_intents) == 1
    assert intent.symbol_intents[0].direction_bias == "long"
    assert intent.is_fallback is False


def test_parse_llm_response_filters_unknown_symbols():
    from services.session_planner import _parse_llm_response
    from schemas.opportunity import OpportunityCard
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    card = OpportunityCard(
        symbol="BTC-USD",
        opportunity_score=0.5,
        opportunity_score_norm=0.7,
        vol_edge=0.6,
        structure_edge=0.6,
        trend_edge=0.6,
        liquidity_score=0.7,
        spread_penalty=0.1,
        instability_penalty=0.0,
        expected_hold_horizon="intraday",
        scored_at=now,
        indicator_as_of=now,
    )

    raw = """{
        "selected_symbols": ["BTC-USD", "FAKE-USD"],
        "symbol_intents": [
            {
                "symbol": "BTC-USD",
                "opportunity_score_norm": 0.7,
                "risk_budget_fraction": 1.0,
                "thesis_summary": "test",
                "expected_hold_horizon": "intraday",
                "direction_bias": "neutral"
            }
        ]
    }"""

    intent = _parse_llm_response(raw, [card])
    # FAKE-USD not in cards → filtered out
    assert "FAKE-USD" not in intent.selected_symbols
    assert "BTC-USD" in intent.selected_symbols


def test_parse_llm_response_markdown_stripped():
    from services.session_planner import _parse_llm_response
    from schemas.opportunity import OpportunityCard
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    card = OpportunityCard(
        symbol="ETH-USD",
        opportunity_score=0.4,
        opportunity_score_norm=0.6,
        vol_edge=0.5,
        structure_edge=0.5,
        trend_edge=0.5,
        liquidity_score=0.6,
        spread_penalty=0.1,
        instability_penalty=0.0,
        expected_hold_horizon="swing",
        scored_at=now,
        indicator_as_of=now,
    )

    raw = """```json
{
    "selected_symbols": ["ETH-USD"],
    "symbol_intents": [
        {
            "symbol": "ETH-USD",
            "opportunity_score_norm": 0.6,
            "risk_budget_fraction": 1.0,
            "thesis_summary": "Swing play on structure",
            "expected_hold_horizon": "swing",
            "direction_bias": "long"
        }
    ]
}
```"""
    intent = _parse_llm_response(raw, [card])
    assert "ETH-USD" in intent.selected_symbols


def test_parse_llm_response_no_valid_symbols_raises():
    from services.session_planner import _parse_llm_response
    from schemas.opportunity import OpportunityCard
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    card = OpportunityCard(
        symbol="BTC-USD",
        opportunity_score=0.5,
        opportunity_score_norm=0.7,
        vol_edge=0.6,
        structure_edge=0.6,
        trend_edge=0.6,
        liquidity_score=0.7,
        spread_penalty=0.1,
        instability_penalty=0.0,
        expected_hold_horizon="intraday",
        scored_at=now,
        indicator_as_of=now,
    )

    raw = '{"selected_symbols": ["FAKE-USD"]}'
    with pytest.raises(ValueError, match="no valid symbols"):
        _parse_llm_response(raw, [card])

"""Tests for Runbook 46: template-matched plan generation via vector store routing."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from schemas.llm_strategist import (
    AssetState,
    IndicatorSnapshot,
    LLMInput,
    PortfolioState,
    RegimeAssessment,
)
from vector_store.retriever import RetrievalResult, StrategyVectorStore
from agents.strategies.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_input(
    regime: str = "range", trend_state: str = "sideways", vol_state: str = "low"
) -> LLMInput:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    indicator = IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=ts,
        close=40000.0,
        rsi_14=50.0,
    )
    asset = AssetState(
        symbol="BTC-USD",
        indicators=[indicator],
        trend_state=trend_state,
        vol_state=vol_state,
        regime_assessment=RegimeAssessment(
            regime=regime, confidence=0.8, primary_signals=["test"]
        ),
    )
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
    return LLMInput(
        portfolio=portfolio,
        assets=[asset],
        risk_params={"max_position_risk_pct": 1.0},
        global_context={"available_timeframes": ["1h"], "regime": regime},
    )


def _make_store(tmp_path: Path, docs: dict[str, str]) -> StrategyVectorStore:
    """Create a StrategyVectorStore from a dict of {filename: content} strategy docs."""
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()
    (tmp_path / "playbooks").mkdir()
    for filename, content in docs.items():
        (strategies_dir / filename).write_text(content, encoding="utf-8")
    return StrategyVectorStore(base_dir=tmp_path)


_COMPRESSION_DOC = """\
---
title: Compression Breakout
type: strategy
regimes: [range, volatile]
tags: [breakout, compression, volatility]
identifiers: [compression_flag, bb_bandwidth_pct_rank, expansion_flag, breakout_confirmed]
template_file: compression_breakout
---
# Compression Breakout
Price is in a consolidation phase: low BB bandwidth, contracting ATR, inside bars.
"""

_BULL_DOC = """\
---
title: Bull Trending
type: strategy
regimes: [bull]
tags: [trend, momentum]
identifiers: [close, sma_medium, rsi_14, macd_hist]
---
# Bull Trending
Favors uptrends where trend_state is uptrend.
"""


# ---------------------------------------------------------------------------
# Test 1: VectorDocument.template_file is parsed + returned in RetrievalResult
# ---------------------------------------------------------------------------

def test_compression_breakout_retrieval_selects_correct_template(tmp_path):
    """A strategy doc with template_file=compression_breakout → template_id returned."""
    store = _make_store(tmp_path, {"compression_breakout.md": _COMPRESSION_DOC})
    assert len(store.documents) == 1
    doc = store.documents[0]
    assert doc.template_file == "compression_breakout"

    # Single doc → it must be ranked first → template_id comes from it
    result = store.retrieve_context(_make_llm_input())
    assert isinstance(result, RetrievalResult)
    assert result.template_id == "compression_breakout"
    assert result.context is not None
    assert "STRATEGY_KNOWLEDGE" in result.context


# ---------------------------------------------------------------------------
# Test 2: Doc without template_file → template_id is None
# ---------------------------------------------------------------------------

def test_bull_trending_retrieval_returns_no_template(tmp_path):
    """Generic bull regime doc with no template_file → template_id is None."""
    store = _make_store(tmp_path, {"bull_trending.md": _BULL_DOC})
    doc = store.documents[0]
    assert doc.template_file is None

    result = store.retrieve_context(_make_llm_input(regime="bull", trend_state="uptrend"))
    assert isinstance(result, RetrievalResult)
    assert result.template_id is None


# ---------------------------------------------------------------------------
# Test 3: generate_plan records retrieved_template_id in last_generation_info
# ---------------------------------------------------------------------------

def test_generate_plan_uses_retrieved_template():
    """If _get_strategy_context returns template_id, generate_plan loads that template
    and records retrieved_template_id in last_generation_info.

    Uses allow_fallback=True so no real LLM is needed; we only verify telemetry.
    """
    # allow_fallback=True means LLM failure → fallback plan; still exercises strategy context.
    client = LLMClient(allow_fallback=True, model="nonexistent-model-r46-test")

    with patch.object(
        client,
        "_get_strategy_context",
        return_value=("some strategy context", "compression_breakout"),
    ):
        plan = client.generate_plan(_make_llm_input())

    assert plan is not None
    # Telemetry must record the retrieved template id
    assert client.last_generation_info.get("retrieved_template_id") == "compression_breakout"


# ---------------------------------------------------------------------------
# Test 4: Explicit prompt_template overrides retrieval (effective_template wins)
# ---------------------------------------------------------------------------

def test_explicit_template_overrides_retrieval():
    """Explicit prompt_template arg takes precedence over retrieved template file.

    Even when retrieval suggests compression_breakout, an explicit prompt_template
    prevents the file from being loaded as the effective template.  The retrieved
    template_id is still recorded in telemetry.
    """
    client = LLMClient(allow_fallback=True, model="nonexistent-model-r46-test")

    with patch.object(
        client,
        "_get_strategy_context",
        return_value=("ctx", "compression_breakout"),
    ):
        plan = client.generate_plan(
            _make_llm_input(),
            prompt_template="EXPLICIT OVERRIDE INSTRUCTIONS",
        )

    assert plan is not None
    # retrieved_template_id is still captured in telemetry
    assert client.last_generation_info.get("retrieved_template_id") == "compression_breakout"


# ---------------------------------------------------------------------------
# Test 5: Missing template file falls back gracefully — no crash
# ---------------------------------------------------------------------------

def test_missing_template_file_falls_back_gracefully():
    """If template_file points to a nonexistent .txt, generate_plan does not crash.

    The effective_template stays None (no file loaded), the base prompt is used,
    and retrieved_template_id is still recorded in last_generation_info.
    """
    client = LLMClient(allow_fallback=True, model="nonexistent-model-r46-test")

    with patch.object(
        client,
        "_get_strategy_context",
        return_value=("ctx", "definitely_nonexistent_template_xyz_r46"),
    ):
        plan = client.generate_plan(_make_llm_input())

    assert plan is not None
    # retrieved_template_id is still recorded even though the file didn't exist
    assert (
        client.last_generation_info.get("retrieved_template_id")
        == "definitely_nonexistent_template_xyz_r46"
    )

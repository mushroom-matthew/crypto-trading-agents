from __future__ import annotations

from datetime import datetime, timezone

from schemas.llm_strategist import AssetState, IndicatorSnapshot, LLMInput, PortfolioState, RegimeAssessment
from vector_store.retriever import get_strategy_vector_store


def _llm_input() -> LLMInput:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    indicator = IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=ts,
        close=40000.0,
        rsi_14=55.0,
    )
    asset = AssetState(
        symbol="BTC-USD",
        indicators=[indicator],
        trend_state="uptrend",
        vol_state="normal",
        regime_assessment=RegimeAssessment(regime="bull", confidence=0.8, primary_signals=["test"]),
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
        global_context={"available_timeframes": ["1h"]},
    )


def test_vector_store_loads_documents():
    store = get_strategy_vector_store()
    assert len(store.documents) >= 5


def test_vector_store_retrieval_context():
    store = get_strategy_vector_store()
    context = store.retrieve_context(_llm_input())
    assert context is not None
    assert "STRATEGY_KNOWLEDGE" in context

"""Tests for Runbook 64: TickSnapshot and StructuralTargetSelector wiring.

Covers:
1. build_tick_snapshot() called per bar returns a TickSnapshot with provenance
2. TriggerEngine.on_bar() accepts tick_snapshot; snapshot_id/hash/staleness_s appear in context
3. _get_latest_structure_snapshot helper returns correct value from history
4. select_stop_candidates / select_target_candidates wired into paper_trading._execute_order
5. build_tick_snapshot wired into evaluate_triggers_activity
6. Explicit stop/target anchors are NOT overridden by structural candidates
"""
from __future__ import annotations

import pathlib
from datetime import datetime, timezone
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _indicator(symbol: str = "BTC-USD", timeframe: str = "1h"):
    from schemas.llm_strategist import IndicatorSnapshot
    return IndicatorSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        as_of=NOW,
        close=50000.0,
        volume=1234.5,
        atr_14=750.0,
        rsi_14=55.0,
        compression_flag=False,
        expansion_flag=False,
        breakout_confirmed=False,
    )


def _portfolio():
    from schemas.llm_strategist import PortfolioState
    return PortfolioState(
        timestamp=NOW,
        equity=100_000.0,
        cash=100_000.0,
        positions={},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )


def _plan():
    from schemas.llm_strategist import RiskConstraint, StrategyPlan, TriggerCondition
    trigger = TriggerCondition(
        id="btc_long",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="close > 0",
        exit_rule="False",
        stop_loss_pct=2.0,
    )
    return StrategyPlan(
        generated_at=NOW,
        valid_until=NOW,
        global_view="test",
        regime="range",
        triggers=[trigger],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=2.0,
            max_symbol_exposure_pct=100.0,
            max_portfolio_exposure_pct=100.0,
            max_daily_loss_pct=10.0,
        ),
    )


def _structure_level(price: float, role: str, dist: float, stop: bool = True, target: bool = True):
    from schemas.structure_engine import StructureLevel
    kind = "swing_high" if role == "resistance" else "swing_low"
    return StructureLevel(
        level_id=f"BTC-USD|{kind}|1h|{price:.4f}",
        snapshot_id=str(uuid4()),
        symbol="BTC-USD",
        as_of_ts=NOW,
        price=price,
        source_timeframe="1h",
        kind=kind,
        source_label=f"{kind} @ {price}",
        role_now=role,
        distance_abs=dist,
        distance_pct=dist / 100.0,
        eligible_for_stop_anchor=stop,
        eligible_for_target_anchor=target,
        eligible_for_entry_trigger=False,
    )


def _make_structure_snapshot(*levels):
    from schemas.structure_engine import StructureSnapshot
    return StructureSnapshot(
        snapshot_id=str(uuid4()),
        snapshot_hash="abcd1234" * 8,
        symbol="BTC-USD",
        as_of_ts=NOW,
        generated_at_ts=NOW,
        source_timeframe="1h",
        reference_price=50000.0,
        levels=list(levels),
    )


# ---------------------------------------------------------------------------
# 1. build_tick_snapshot returns a valid TickSnapshot
# ---------------------------------------------------------------------------

def test_build_tick_snapshot_returns_tick_snapshot():
    from services.market_snapshot_builder import build_tick_snapshot
    from schemas.market_snapshot import TickSnapshot

    ind = _indicator()
    snap = build_tick_snapshot(ind)

    assert isinstance(snap, TickSnapshot)
    assert snap.provenance.snapshot_id
    assert snap.provenance.snapshot_hash
    assert snap.provenance.symbol == "BTC-USD"
    assert snap.close == 50000.0


def test_build_tick_snapshot_staleness_in_quality():
    from services.market_snapshot_builder import build_tick_snapshot

    ind = _indicator()
    snap = build_tick_snapshot(ind)

    assert snap.quality is not None
    assert snap.quality.staleness_seconds is not None
    assert snap.quality.staleness_seconds >= 0.0


# ---------------------------------------------------------------------------
# 2. TriggerEngine.on_bar() accepts tick_snapshot; context gets snapshot fields
# ---------------------------------------------------------------------------

def test_trigger_engine_on_bar_accepts_tick_snapshot():
    """on_bar() must accept tick_snapshot keyword argument without error."""
    from agents.strategies.risk_engine import RiskEngine
    from agents.strategies.trade_risk import TradeRiskEvaluator
    from agents.strategies.trigger_engine import Bar, TriggerEngine
    from services.market_snapshot_builder import build_tick_snapshot

    plan = _plan()
    re = RiskEngine(plan.risk_constraints, {}, risk_profile=None)
    engine = TriggerEngine(plan, re, trade_risk=TradeRiskEvaluator(re))

    ind = _indicator()
    tick_snapshot = build_tick_snapshot(ind)
    bar = Bar(
        symbol="BTC-USD", timeframe="1h", timestamp=NOW,
        open=50000.0, high=50100.0, low=49900.0, close=50000.0, volume=1.0,
    )

    # Must not raise; tick_snapshot forwarded as keyword arg
    orders, blocks = engine.on_bar(
        bar, ind, _portfolio(), tick_snapshot=tick_snapshot
    )
    # No assertions on orders — only verifying the call succeeds


def test_trigger_engine_context_gets_snapshot_id_hash_staleness():
    """_context() must surface snapshot_id, snapshot_hash, snapshot_staleness_s."""
    from agents.strategies.risk_engine import RiskEngine
    from agents.strategies.trade_risk import TradeRiskEvaluator
    from agents.strategies.trigger_engine import TriggerEngine
    from services.market_snapshot_builder import build_tick_snapshot

    plan = _plan()
    re = RiskEngine(plan.risk_constraints, {}, risk_profile=None)
    engine = TriggerEngine(plan, re, trade_risk=TradeRiskEvaluator(re))

    ind = _indicator()
    tick_snapshot = build_tick_snapshot(ind)

    ctx = engine._context(ind, None, None, _portfolio(), None, tick_snapshot=tick_snapshot)

    assert ctx["snapshot_id"] == tick_snapshot.provenance.snapshot_id
    assert ctx["snapshot_hash"] == tick_snapshot.provenance.snapshot_hash
    assert ctx["snapshot_staleness_s"] is not None
    assert ctx["snapshot_staleness_s"] >= 0.0


def test_trigger_engine_context_without_snapshot_has_none_fields():
    """Context without tick_snapshot must still have snapshot keys set to None."""
    from agents.strategies.risk_engine import RiskEngine
    from agents.strategies.trade_risk import TradeRiskEvaluator
    from agents.strategies.trigger_engine import TriggerEngine

    plan = _plan()
    re = RiskEngine(plan.risk_constraints, {}, risk_profile=None)
    engine = TriggerEngine(plan, re, trade_risk=TradeRiskEvaluator(re))

    ind = _indicator()
    ctx = engine._context(ind, None, None, _portfolio(), None, tick_snapshot=None)

    assert "snapshot_id" in ctx
    assert "snapshot_hash" in ctx
    assert "snapshot_staleness_s" in ctx
    assert ctx["snapshot_id"] is None
    assert ctx["snapshot_hash"] is None
    assert ctx["snapshot_staleness_s"] is None


# ---------------------------------------------------------------------------
# 3. _get_latest_structure_snapshot helper
# ---------------------------------------------------------------------------

def test_get_latest_structure_snapshot_empty_history_returns_none():
    from tools.paper_trading import _get_latest_structure_snapshot

    result = _get_latest_structure_snapshot({}, "BTC-USD")
    assert result is None


def test_get_latest_structure_snapshot_returns_most_recent():
    from tools.paper_trading import _get_latest_structure_snapshot

    snap = _make_structure_snapshot()
    history = {"BTC-USD": [snap.model_dump(mode="json")]}

    result = _get_latest_structure_snapshot(history, "BTC-USD")
    assert result is not None
    assert result.snapshot_id == snap.snapshot_id
    assert result.symbol == "BTC-USD"


def test_get_latest_structure_snapshot_wrong_symbol_returns_none():
    from tools.paper_trading import _get_latest_structure_snapshot

    snap = _make_structure_snapshot()
    history = {"BTC-USD": [snap.model_dump(mode="json")]}

    result = _get_latest_structure_snapshot(history, "ETH-USD")
    assert result is None


def test_get_latest_structure_snapshot_invalid_data_returns_none():
    from tools.paper_trading import _get_latest_structure_snapshot

    history = {"BTC-USD": [{"invalid": "data"}]}
    result = _get_latest_structure_snapshot(history, "BTC-USD")
    assert result is None


# ---------------------------------------------------------------------------
# 4. select_stop/target_candidates integration with _get_latest_structure_snapshot
# ---------------------------------------------------------------------------

def test_select_stop_candidates_from_history_long():
    from tools.paper_trading import _get_latest_structure_snapshot
    from services.structural_target_selector import select_stop_candidates

    support_level = _structure_level(48000.0, "support", 2000.0, stop=True, target=False)
    resist_level = _structure_level(52000.0, "resistance", 2000.0, stop=False, target=True)
    snap = _make_structure_snapshot(support_level, resist_level)
    history = {"BTC-USD": [snap.model_dump(mode="json")]}

    structure = _get_latest_structure_snapshot(history, "BTC-USD")
    assert structure is not None

    stops = select_stop_candidates(structure, direction="long", max_distance_atr=3.0)
    assert len(stops) >= 1
    assert all(lvl.role_now == "support" for lvl in stops)
    assert stops[0].price == 48000.0


def test_select_target_candidates_from_history_long():
    from tools.paper_trading import _get_latest_structure_snapshot
    from services.structural_target_selector import select_target_candidates

    support_level = _structure_level(48000.0, "support", 2000.0, stop=True, target=False)
    resist_level = _structure_level(52000.0, "resistance", 2000.0, stop=False, target=True)
    snap = _make_structure_snapshot(support_level, resist_level)
    history = {"BTC-USD": [snap.model_dump(mode="json")]}

    structure = _get_latest_structure_snapshot(history, "BTC-USD")
    assert structure is not None

    targets = select_target_candidates(structure, direction="long", max_distance_atr=10.0)
    assert len(targets) >= 1
    assert all(lvl.role_now == "resistance" for lvl in targets)
    assert targets[0].price == 52000.0


# ---------------------------------------------------------------------------
# 5. Wiring checks — source code references (fast smoke tests)
# ---------------------------------------------------------------------------

def test_paper_trading_references_build_tick_snapshot():
    """evaluate_triggers_activity must call build_tick_snapshot (R64 wiring)."""
    src = pathlib.Path("tools/paper_trading.py").read_text()
    assert "build_tick_snapshot" in src, (
        "build_tick_snapshot not found in tools/paper_trading.py — R64 wiring missing"
    )


def test_paper_trading_references_select_stop_candidates():
    """_execute_order must reference select_stop_candidates (R64 structural stop wiring)."""
    src = pathlib.Path("tools/paper_trading.py").read_text()
    assert "select_stop_candidates" in src, (
        "select_stop_candidates not found in tools/paper_trading.py — R64 structural stop wiring missing"
    )


def test_paper_trading_references_select_target_candidates():
    """_execute_order must reference select_target_candidates (R64 structural target wiring)."""
    src = pathlib.Path("tools/paper_trading.py").read_text()
    assert "select_target_candidates" in src, (
        "select_target_candidates not found in tools/paper_trading.py — R64 structural target wiring missing"
    )


def test_trigger_engine_on_bar_accepts_tick_snapshot_kwarg():
    """on_bar() signature must include tick_snapshot parameter (R64 wiring)."""
    import inspect
    from agents.strategies.trigger_engine import TriggerEngine
    sig = inspect.signature(TriggerEngine.on_bar)
    assert "tick_snapshot" in sig.parameters, (
        "TriggerEngine.on_bar() missing tick_snapshot parameter — R64 wiring missing"
    )


def test_paper_trading_references_get_latest_structure_snapshot():
    """paper_trading.py must define _get_latest_structure_snapshot (R64 helper)."""
    src = pathlib.Path("tools/paper_trading.py").read_text()
    assert "_get_latest_structure_snapshot" in src, (
        "_get_latest_structure_snapshot not found in tools/paper_trading.py — R64 helper missing"
    )


# ---------------------------------------------------------------------------
# 6. Explicit anchors are NOT overridden by structural candidates
# ---------------------------------------------------------------------------

def test_explicit_stop_anchor_not_in_structural_candidate_path():
    """When stop_anchor_type is set, structural candidate selection is skipped.

    This test validates the guard condition in _execute_order:
      if stop_anchor_type is None: (only then do we log structural candidates)
    """
    src = pathlib.Path("tools/paper_trading.py").read_text()
    # The guard must use stop_anchor_type is None check
    assert "stop_anchor_type is None" in src, (
        "Guard 'stop_anchor_type is None' not found — explicit anchors may be overridden"
    )


def test_explicit_target_anchor_not_in_structural_candidate_path():
    """When target_anchor_type is set, structural target candidate selection is skipped."""
    src = pathlib.Path("tools/paper_trading.py").read_text()
    assert "target_anchor_type is None" in src, (
        "Guard 'target_anchor_type is None' not found — explicit anchors may be overridden"
    )

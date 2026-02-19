"""Unit tests for Runbook 45: Adaptive Trade Management (R-multiple state machine).

Tests cover:
- PositionRiskState R-tracking field defaults
- _apply_stop_adjustment: long and short, advance-only, wick buffer, event emission
- _advance_trade_state: R1/R2/R3 rungs, partial exits, MFE/MAE update, rung catch
- Trade state progression: EARLY → MATURE → EXTENDED → TRAIL
- Trigger engine _context(): current_R / mfe_r / mae_r / trade_state / position_fraction /
  r1_reached / r2_reached / r3_reached identifiers
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.llm_strategist_runner import (
    LLMStrategistBacktester,
    PositionRiskState,
    TradeManagementConfig,
)
from agents.strategies.trigger_engine import Bar, Order, TriggerEngine
from agents.strategies.risk_engine import RiskEngine
from schemas.llm_strategist import (
    IndicatorSnapshot,
    PositionSizingRule,
    RiskConstraint,
    StrategyPlan,
    TriggerCondition,
)


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

def _ts() -> datetime:
    return datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)


def _state(
    entry_price: float = 50_000.0,
    direction: str = "long",
    initial_risk_abs: float | None = 1_000.0,
    stop_price: float | None = 49_000.0,
    **kwargs,
) -> PositionRiskState:
    """Build a minimal PositionRiskState."""
    return PositionRiskState(
        symbol="BTC-USD",
        timeframe="1h",
        entry_ts=_ts(),
        entry_price=entry_price,
        direction=direction,
        initial_risk_abs=initial_risk_abs,
        stop_price=stop_price,
        **kwargs,
    )


def _bar(close: float = 50_000.0, timeframe: str = "1h", symbol: str = "BTC-USD") -> Bar:
    """Build a minimal Bar."""
    return Bar(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=_ts(),
        open=close - 100,
        high=close + 100,
        low=close - 100,
        close=close,
        volume=1_000.0,
    )


class _FakeBacktester:
    """Duck-typed mock for unit-testing R-management methods without full run()."""

    def __init__(self, config: TradeManagementConfig | None = None, base_tf: str = "1h"):
        self.portfolio = SimpleNamespace(position_meta={}, positions={})
        self.position_risk_state: dict = {}
        self.trade_mgmt_config = config or TradeManagementConfig()
        self._trade_mgmt_events: list = []
        self.base_timeframe = base_tf

    # Bind LLMStrategistBacktester methods to this mock object
    _apply_stop_adjustment = LLMStrategistBacktester._apply_stop_adjustment
    _advance_trade_state = LLMStrategistBacktester._advance_trade_state


def _plan_with_triggers(triggers: list[TriggerCondition]) -> StrategyPlan:
    """Build a minimal StrategyPlan for TriggerEngine instantiation."""
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rc = RiskConstraint(
        max_position_risk_pct=5.0,
        max_symbol_exposure_pct=50.0,
        max_portfolio_exposure_pct=80.0,
        max_daily_loss_pct=3.0,
    )
    return StrategyPlan(
        generated_at=now,
        valid_until=now,
        global_view="test",
        regime="range",
        triggers=triggers,
        risk_constraints=rc,
        sizing_rules=[
            PositionSizingRule(
                symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0
            )
        ],
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long", "short"],
        allowed_trigger_categories=["trend_continuation"],
    )


def _engine() -> TriggerEngine:
    """Build a minimal TriggerEngine with no triggers for _context() testing."""
    plan = _plan_with_triggers([
        TriggerCondition(
            id="t1",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="False",
            exit_rule="False",
            category="trend_continuation",
        )
    ])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    return TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)


def _indicator(close: float = 50_000.0) -> IndicatorSnapshot:
    """Build a minimal IndicatorSnapshot."""
    return IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=_ts(),
        close=close,
        low=close - 500,
        high=close + 500,
        atr_14=800.0,
    )


# ---------------------------------------------------------------------------
# 1–2. PositionRiskState R-tracking field defaults
# ---------------------------------------------------------------------------

def test_position_risk_state_r_tracking_field_defaults():
    """New R-tracking fields must default to safe/neutral values."""
    s = _state(initial_risk_abs=None, stop_price=None)
    assert s.initial_risk_abs is None
    assert s.position_fraction == 1.0
    assert s.trade_state == "EARLY"
    assert s.mfe_r == 0.0
    assert s.mae_r == 0.0


def test_position_risk_state_trigger_flags_default_false():
    """r1/r2/r3 flags must default to False."""
    s = _state()
    assert s.r1_triggered is False
    assert s.r2_triggered is False
    assert s.r3_triggered is False


# ---------------------------------------------------------------------------
# 3–8. _apply_stop_adjustment — long direction
# ---------------------------------------------------------------------------

def test_apply_stop_adj_moves_stop_to_neg_r_for_long():
    """R1 adjustment moves stop to entry - 0.25 * initial_risk for longs."""
    bt = _FakeBacktester()
    s = _state(entry_price=50_000.0, direction="long", initial_risk_abs=1_000.0, stop_price=48_500.0)
    bt.portfolio.position_meta["BTC-USD"] = {"entry_side": "long"}
    bt._apply_stop_adjustment(s, "BTC-USD", -0.25, "r1_mature", 1.0, _ts(), bt.trade_mgmt_config)
    # candidate = 50000 + (-0.25 * 1000) = 49750; > 48500 so advance-only passes
    assert s.stop_price == pytest.approx(49_750.0)


def test_apply_stop_adj_advance_only_long_rejects_backward_move():
    """Stop must not move backward (down) for longs."""
    bt = _FakeBacktester()
    s = _state(entry_price=50_000.0, direction="long", initial_risk_abs=1_000.0, stop_price=50_000.0)
    bt.portfolio.position_meta["BTC-USD"] = {"entry_side": "long"}
    # candidate = 50000 - 0.25 * 1000 = 49750, current = 50000 → reject
    bt._apply_stop_adjustment(s, "BTC-USD", -0.25, "r1_mature", 1.0, _ts(), bt.trade_mgmt_config)
    assert s.stop_price == pytest.approx(50_000.0)  # unchanged
    assert len(bt._trade_mgmt_events) == 0


def test_apply_stop_adj_emits_stop_adjustment_event():
    """A successful adjustment emits one StopAdjustmentEvent into _trade_mgmt_events."""
    bt = _FakeBacktester()
    s = _state(entry_price=50_000.0, direction="long", initial_risk_abs=1_000.0, stop_price=48_000.0)
    bt.portfolio.position_meta["BTC-USD"] = {"entry_side": "long"}
    bt._apply_stop_adjustment(s, "BTC-USD", -0.25, "r1_mature", 1.2, _ts(), bt.trade_mgmt_config)
    assert len(bt._trade_mgmt_events) == 1
    ev = bt._trade_mgmt_events[0]
    assert ev["rung"] == "r1_mature"
    assert ev["current_R"] == pytest.approx(1.2)
    assert ev["new_stop"] == pytest.approx(49_750.0)
    assert ev["engine_version"] == "45.0.0"


def test_apply_stop_adj_updates_position_meta_stop_price_abs():
    """After adjustment stop_price_abs in position_meta must reflect new stop."""
    bt = _FakeBacktester()
    s = _state(entry_price=50_000.0, direction="long", initial_risk_abs=1_000.0, stop_price=47_000.0)
    bt.portfolio.position_meta["BTC-USD"] = {"entry_side": "long"}
    bt._apply_stop_adjustment(s, "BTC-USD", -0.25, "r1_mature", 1.0, _ts(), bt.trade_mgmt_config)
    assert bt.portfolio.position_meta["BTC-USD"]["stop_price_abs"] == pytest.approx(49_750.0)


def test_apply_stop_adj_wick_buffer_reduces_candidate_for_long():
    """wick_buffer_r nudges stop further below candidate (cushion away from price)."""
    cfg = TradeManagementConfig(wick_buffer_r=0.1)  # 10% of initial risk as buffer
    bt = _FakeBacktester(config=cfg)
    s = _state(entry_price=50_000.0, direction="long", initial_risk_abs=1_000.0, stop_price=47_000.0)
    bt.portfolio.position_meta["BTC-USD"] = {"entry_side": "long"}
    bt._apply_stop_adjustment(s, "BTC-USD", -0.25, "r1_mature", 1.0, _ts(), cfg)
    # candidate = 50000 - 250 - buffer(0.1 * 1000 = 100) = 49650
    assert s.stop_price == pytest.approx(49_650.0)


def test_apply_stop_adj_no_change_if_initial_risk_none():
    """If initial_risk_abs is None, adjustment is a no-op."""
    bt = _FakeBacktester()
    s = _state(initial_risk_abs=None, stop_price=48_000.0)
    bt.portfolio.position_meta["BTC-USD"] = {"entry_side": "long"}
    bt._apply_stop_adjustment(s, "BTC-USD", -0.25, "r1_mature", 1.0, _ts(), bt.trade_mgmt_config)
    assert s.stop_price == pytest.approx(48_000.0)
    assert len(bt._trade_mgmt_events) == 0


# ---------------------------------------------------------------------------
# 9–10. _apply_stop_adjustment — short direction
# ---------------------------------------------------------------------------

def test_apply_stop_adj_moves_stop_for_short():
    """R1 adjustment for short: stop → entry + 0.25 * initial_risk (above entry)."""
    bt = _FakeBacktester()
    s = _state(entry_price=50_000.0, direction="short", initial_risk_abs=1_000.0, stop_price=52_000.0)
    bt.portfolio.position_meta["BTC-USD"] = {"entry_side": "short"}
    bt._apply_stop_adjustment(s, "BTC-USD", -0.25, "r1_mature", 1.0, _ts(), bt.trade_mgmt_config)
    # candidate = 50000 - (-0.25 * 1000) = 50250; < 52000 → advance-only passes for shorts
    assert s.stop_price == pytest.approx(50_250.0)


def test_apply_stop_adj_advance_only_short_rejects_backward_move():
    """Stop must not move backward (up) for shorts."""
    bt = _FakeBacktester()
    s = _state(entry_price=50_000.0, direction="short", initial_risk_abs=1_000.0, stop_price=50_000.0)
    bt.portfolio.position_meta["BTC-USD"] = {"entry_side": "short"}
    # candidate = 50250 > current 50000 → reject (would move stop against direction)
    bt._apply_stop_adjustment(s, "BTC-USD", -0.25, "r1_mature", 1.0, _ts(), bt.trade_mgmt_config)
    assert s.stop_price == pytest.approx(50_000.0)
    assert len(bt._trade_mgmt_events) == 0


# ---------------------------------------------------------------------------
# 11–14. _advance_trade_state: rung triggers (long)
# ---------------------------------------------------------------------------

def _setup_long_position(bt: _FakeBacktester, entry: float = 50_000.0, ir: float = 1_000.0, qty: float = 0.1):
    """Populate bt with an open long position."""
    s = _state(entry_price=entry, direction="long", initial_risk_abs=ir, stop_price=entry - ir)
    bt.position_risk_state["BTC-USD"] = s
    bt.portfolio.position_meta["BTC-USD"] = {"entry_side": "long", "stop_price_abs": entry - ir}
    bt.portfolio.positions["BTC-USD"] = qty
    return s


def test_advance_trade_state_no_fire_below_r1():
    """No rung fires when current_R < 1.0."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt)
    # close = 50000 + 0.5 * 1000 = 50500 → current_R = 0.5
    partials = bt._advance_trade_state(_bar(close=50_500.0))
    assert partials == []
    assert s.r1_triggered is False
    assert s.trade_state == "EARLY"
    assert len(bt._trade_mgmt_events) == 0


def test_advance_trade_state_r1_fires_at_threshold():
    """R1 fires at current_R >= 1.0 and advances stop."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt, entry=50_000.0, ir=1_000.0)
    # close = 51000 → current_R = 1.0
    partials = bt._advance_trade_state(_bar(close=51_000.0))
    assert partials == []  # R1 is stop-only, no partial
    assert s.r1_triggered is True
    assert s.trade_state == "MATURE"
    # Stop should move to entry - 0.25 * ir = 49750
    assert s.stop_price == pytest.approx(49_750.0)


def test_advance_trade_state_r2_fires_at_threshold():
    """R2 fires at current_R >= 2.0, emits partial order and moves stop."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt, entry=50_000.0, ir=1_000.0, qty=1.0)
    s.r1_triggered = True  # R1 already fired
    # close = 52000 → current_R = 2.0
    partials = bt._advance_trade_state(_bar(close=52_000.0))
    assert len(partials) == 1
    assert partials[0].reason == "r2_partial_exit"
    assert partials[0].quantity == pytest.approx(0.5)  # 50% of 1.0
    assert s.r2_triggered is True
    assert s.trade_state == "EXTENDED"
    # Stop → entry + 0.5 * ir = 50500
    assert s.stop_price == pytest.approx(50_500.0)


def test_advance_trade_state_r3_fires_at_threshold():
    """R3 fires at current_R >= 3.0, moves stop to +1.5R."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt, entry=50_000.0, ir=1_000.0)
    s.r1_triggered = True
    s.r2_triggered = True
    # close = 53000 → current_R = 3.0
    partials = bt._advance_trade_state(_bar(close=53_000.0))
    assert partials == []  # R3 is stop-only
    assert s.r3_triggered is True
    assert s.trade_state == "TRAIL"
    # Stop → entry + 1.5 * ir = 51500
    assert s.stop_price == pytest.approx(51_500.0)


# ---------------------------------------------------------------------------
# 15–16. Partial exit order and event
# ---------------------------------------------------------------------------

def test_advance_trade_state_r2_emits_partial_exit_order_fields():
    """Partial exit order has correct side, intent, exit_fraction for longs."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt, entry=50_000.0, ir=1_000.0, qty=2.0)
    s.r1_triggered = True
    partials = bt._advance_trade_state(_bar(close=52_000.0))
    assert len(partials) == 1
    o = partials[0]
    assert o.side == "sell"
    assert o.intent == "exit"
    assert o.exit_fraction == pytest.approx(0.5)
    assert o.price == pytest.approx(52_000.0)


def test_advance_trade_state_r2_emits_partial_exit_event():
    """R2 partial exit emits a PartialExitEvent into _trade_mgmt_events."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt, entry=50_000.0, ir=1_000.0, qty=1.0)
    s.r1_triggered = True
    bt._advance_trade_state(_bar(close=52_000.0))
    partial_events = [e for e in bt._trade_mgmt_events if "fraction_exited" in e]
    assert len(partial_events) == 1
    ev = partial_events[0]
    assert ev["rung"] == "r2_extended"
    assert ev["fraction_exited"] == pytest.approx(0.5)
    assert ev["exit_R"] == pytest.approx(2.0)
    assert ev["initial_risk_abs"] == pytest.approx(1_000.0)
    assert ev["position_fraction_before"] == pytest.approx(1.0)
    assert ev["position_fraction_after"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 17. R1 not re-triggered
# ---------------------------------------------------------------------------

def test_advance_trade_state_r1_not_re_triggered():
    """R1 does not fire twice once r1_triggered is True."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt, entry=50_000.0, ir=1_000.0)
    s.r1_triggered = True
    s.trade_state = "MATURE"
    old_stop = s.stop_price
    bt._advance_trade_state(_bar(close=51_000.0))  # still at R1 level
    assert s.r1_triggered is True
    assert len([e for e in bt._trade_mgmt_events if e.get("rung") == "r1_mature"]) == 0


# ---------------------------------------------------------------------------
# 18–19. MFE / MAE tracking
# ---------------------------------------------------------------------------

def test_advance_trade_state_updates_mfe_r():
    """mfe_r is updated to the peak current_R seen."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt, entry=50_000.0, ir=1_000.0)
    bt._advance_trade_state(_bar(close=51_500.0))  # current_R = 1.5
    bt._advance_trade_state(_bar(close=51_200.0))  # current_R = 1.2 (pullback)
    assert s.mfe_r == pytest.approx(1.5)


def test_advance_trade_state_updates_mae_r():
    """mae_r tracks the most adverse excursion (lowest current_R)."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt, entry=50_000.0, ir=1_000.0)
    bt._advance_trade_state(_bar(close=49_500.0))  # current_R = -0.5
    bt._advance_trade_state(_bar(close=50_000.0))  # current_R = 0.0
    assert s.mae_r == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# 20. Rung catch (multi-rung gap)
# ---------------------------------------------------------------------------

def test_advance_trade_state_rung_catch_on_gap():
    """If price gaps from EARLY to R2 in one bar, R1 and R2 both fire (rung_catch=True)."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt, entry=50_000.0, ir=1_000.0, qty=1.0)
    # current_R = 2.1 → both R1 and R2 fire simultaneously
    bt._advance_trade_state(_bar(close=52_100.0))
    assert s.r1_triggered is True
    assert s.r2_triggered is True
    # Both stop events should have rung_catch=True
    stop_events = [e for e in bt._trade_mgmt_events if "new_stop" in e]
    assert all(e["rung_catch"] is True for e in stop_events)


# ---------------------------------------------------------------------------
# 21. position_meta current_R update
# ---------------------------------------------------------------------------

def test_advance_trade_state_updates_meta_current_R():
    """_advance_trade_state must write current_R into position_meta for trigger engine."""
    bt = _FakeBacktester()
    _setup_long_position(bt, entry=50_000.0, ir=1_000.0)
    bt._advance_trade_state(_bar(close=51_500.0))  # current_R = 1.5
    assert bt.portfolio.position_meta["BTC-USD"]["current_R"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 22. Disabled config
# ---------------------------------------------------------------------------

def test_advance_trade_state_returns_empty_if_disabled():
    """If TradeManagementConfig.enabled is False, _advance_trade_state is a no-op."""
    cfg = TradeManagementConfig(enabled=False)
    bt = _FakeBacktester(config=cfg)
    _setup_long_position(bt, entry=50_000.0, ir=1_000.0)
    partials = bt._advance_trade_state(_bar(close=53_000.0))
    assert partials == []
    assert len(bt._trade_mgmt_events) == 0


# ---------------------------------------------------------------------------
# 23–24. Short direction via _advance_trade_state
# ---------------------------------------------------------------------------

def _setup_short_position(bt: _FakeBacktester, entry: float = 50_000.0, ir: float = 1_000.0, qty: float = -0.1):
    """Populate bt with an open short position."""
    s = PositionRiskState(
        symbol="BTC-USD",
        timeframe="1h",
        entry_ts=_ts(),
        entry_price=entry,
        direction="short",
        initial_risk_abs=ir,
        stop_price=entry + ir,
    )
    bt.position_risk_state["BTC-USD"] = s
    bt.portfolio.position_meta["BTC-USD"] = {"entry_side": "short", "stop_price_abs": entry + ir}
    bt.portfolio.positions["BTC-USD"] = qty
    return s


def test_advance_trade_state_r1_fires_short():
    """R1 fires for shorts when price falls 1R below entry."""
    bt = _FakeBacktester()
    s = _setup_short_position(bt, entry=50_000.0, ir=1_000.0)
    # current_R (short) = (entry - close) / ir = (50000 - 49000) / 1000 = 1.0
    bt._advance_trade_state(_bar(close=49_000.0))
    assert s.r1_triggered is True
    assert s.trade_state == "MATURE"
    # Stop for short: entry - (-0.25 * ir) = 50000 + 250 = 50250; < 51000 → advance-only ok
    assert s.stop_price == pytest.approx(50_250.0)


def test_advance_trade_state_r2_fires_short_partial():
    """R2 for shorts emits a buy partial exit order."""
    bt = _FakeBacktester()
    s = _setup_short_position(bt, entry=50_000.0, ir=1_000.0, qty=-1.0)
    s.r1_triggered = True
    # current_R = (50000 - 48000) / 1000 = 2.0
    partials = bt._advance_trade_state(_bar(close=48_000.0))
    assert len(partials) == 1
    assert partials[0].side == "buy"  # buy to cover for shorts
    assert partials[0].quantity == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 25–27. Trade state progression
# ---------------------------------------------------------------------------

def test_trade_state_early_to_mature():
    """EARLY → MATURE when R1 fires."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt)
    assert s.trade_state == "EARLY"
    bt._advance_trade_state(_bar(close=51_000.0))
    assert s.trade_state == "MATURE"


def test_trade_state_mature_to_extended():
    """MATURE → EXTENDED when R2 fires (after R1 already fired)."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt, qty=1.0)
    s.r1_triggered = True
    s.trade_state = "MATURE"
    bt._advance_trade_state(_bar(close=52_000.0))
    assert s.trade_state == "EXTENDED"


def test_trade_state_extended_to_trail():
    """EXTENDED → TRAIL when R3 fires."""
    bt = _FakeBacktester()
    s = _setup_long_position(bt)
    s.r1_triggered = True
    s.r2_triggered = True
    s.trade_state = "EXTENDED"
    bt._advance_trade_state(_bar(close=53_000.0))
    assert s.trade_state == "TRAIL"


# ---------------------------------------------------------------------------
# 28–34. Trigger engine _context() R-tracking identifiers
# ---------------------------------------------------------------------------

def _ctx_with_meta(meta: dict, close: float = 50_000.0) -> dict:
    """Build a trigger engine context with the given position_meta."""
    from schemas.llm_strategist import PortfolioState as LLMPortfolioState
    eng = _engine()
    ind = _indicator(close=close)
    portfolio = LLMPortfolioState(
        timestamp=_ts(),
        equity=100_000.0,
        cash=90_000.0,
        positions={"BTC-USD": 0.1},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.5,
        profit_factor_30d=1.0,
    )
    return eng._context(ind, asset_state=None, portfolio=portfolio, position_meta={"BTC-USD": meta})


def test_r_tracking_defaults_in_context_when_no_meta():
    """R-tracking fields default to 0.0 / 'EARLY' / False when meta has no R data."""
    eng = _engine()
    ind = _indicator()
    ctx = eng._context(ind, asset_state=None)
    assert ctx["current_R"] == pytest.approx(0.0)
    assert ctx["mfe_r"] == pytest.approx(0.0)
    assert ctx["mae_r"] == pytest.approx(0.0)
    assert ctx["trade_state"] == "EARLY"
    assert ctx["position_fraction"] == pytest.approx(1.0)
    assert ctx["r1_reached"] is False
    assert ctx["r2_reached"] is False
    assert ctx["r3_reached"] is False


def test_r_tracking_reads_current_R_from_meta():
    """current_R in context comes from position_meta['current_R']."""
    ctx = _ctx_with_meta({"current_R": 1.5, "entry_side": "long"})
    assert ctx["current_R"] == pytest.approx(1.5)


def test_r_tracking_reads_mfe_mae_from_meta():
    """mfe_r and mae_r in context come from position_meta."""
    ctx = _ctx_with_meta({"current_R": 1.5, "mfe_r": 1.8, "mae_r": -0.3, "entry_side": "long"})
    assert ctx["mfe_r"] == pytest.approx(1.8)
    assert ctx["mae_r"] == pytest.approx(-0.3)


def test_r1_reached_true_when_current_R_ge_1():
    """r1_reached is True when current_R >= 1.0."""
    ctx = _ctx_with_meta({"current_R": 1.0, "entry_side": "long"})
    assert ctx["r1_reached"] is True


def test_r2_reached_true_when_current_R_ge_2():
    """r2_reached is True when current_R >= 2.0."""
    ctx = _ctx_with_meta({"current_R": 2.1, "entry_side": "long"})
    assert ctx["r2_reached"] is True
    assert ctx["r1_reached"] is True


def test_r3_reached_true_when_current_R_ge_3():
    """r3_reached is True when current_R >= 3.0."""
    ctx = _ctx_with_meta({"current_R": 3.0, "entry_side": "long"})
    assert ctx["r3_reached"] is True


def test_trade_state_in_context():
    """trade_state identifier mirrors position_meta value."""
    ctx = _ctx_with_meta({"current_R": 1.5, "trade_state": "MATURE", "entry_side": "long"})
    assert ctx["trade_state"] == "MATURE"


def test_position_fraction_in_context():
    """position_fraction identifier mirrors position_meta value."""
    ctx = _ctx_with_meta({"current_R": 2.0, "position_fraction": 0.5, "entry_side": "long"})
    assert ctx["position_fraction"] == pytest.approx(0.5)

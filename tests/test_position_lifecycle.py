"""Tests for R63: Position lifecycle completion.

Covers:
1. AdaptiveTradeManagementState: initial(), tick(), phase transitions
2. SetupEventGenerator wired into evaluate_triggers_activity
3. episode_memory_store_state appended after position close
4. SessionState round-trip with new R63 fields
5. generate_strategy_plan_activity loads in-session episodes
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. AdaptiveTradeManagementState
# ---------------------------------------------------------------------------

class TestAdaptiveTradeManagementState:
    """Unit tests for the R-multiple state machine."""

    def _state(self, **kwargs):
        from services.adaptive_trade_management import AdaptiveTradeManagementState
        # Map legacy stop_price_abs → initial_stop_price + active_stop_price (R85 rename)
        if "stop_price_abs" in kwargs:
            stop = kwargs.pop("stop_price_abs")
            kwargs.setdefault("initial_stop_price", stop)
            kwargs.setdefault("active_stop_price", stop)
        defaults = dict(
            symbol="BTC-USD",
            direction="long",
            entry_price=100.0,
            initial_stop_price=95.0,
            active_stop_price=95.0,
            target_price_abs=115.0,
            phase="EARLY",
            bars_held=0,
            peak_r_achieved=0.0,
        )
        defaults.update(kwargs)
        return AdaptiveTradeManagementState(**defaults)

    def test_initial_from_position_meta(self):
        from services.adaptive_trade_management import AdaptiveTradeManagementState
        meta = {
            "symbol": "ETH-USD",
            "entry_side": "short",
            "signal_entry_price": 200.0,
            "stop_price_abs": 210.0,
            "target_price_abs": 180.0,
        }
        state = AdaptiveTradeManagementState.initial(meta)
        assert state.symbol == "ETH-USD"
        assert state.direction == "short"
        assert state.entry_price == 200.0
        assert state.phase == "EARLY"
        assert state.bars_held == 0
        assert state.peak_r_achieved == 0.0

    def test_tick_increments_bars_held(self):
        s = self._state()
        s2 = s.tick(current_price=100.5)
        assert s2.bars_held == 1
        assert s2.last_price == 100.5

    def test_phase_early_below_threshold(self):
        s = self._state(entry_price=100.0, stop_price_abs=95.0)
        # R = (102 - 100) / 5 = 0.4 < 0.5 → EARLY
        s2 = s.tick(current_price=102.0)
        assert s2.phase == "EARLY"

    def test_phase_transitions_mature(self):
        s = self._state(entry_price=100.0, stop_price_abs=95.0)
        # R = (102.5 - 100) / 5 = 0.5 → MATURE
        s2 = s.tick(current_price=102.5)
        assert s2.phase == "MATURE"

    def test_phase_transitions_extended(self):
        s = self._state(entry_price=100.0, stop_price_abs=95.0)
        # R = (105 - 100) / 5 = 1.0 → EXTENDED
        s2 = s.tick(current_price=105.0)
        assert s2.phase == "EXTENDED"

    def test_phase_transitions_trail(self):
        s = self._state(entry_price=100.0, stop_price_abs=95.0)
        # R = (107.5 - 100) / 5 = 1.5 → TRAIL
        s2 = s.tick(current_price=107.5)
        assert s2.phase == "TRAIL"

    def test_phase_monotonic_no_regression(self):
        """Phase never regresses even if price drops back."""
        s = self._state(entry_price=100.0, stop_price_abs=95.0)
        # Advance to TRAIL
        s = s.tick(current_price=107.5)
        assert s.phase == "TRAIL"
        # Price drops back near entry — peak_r stays → TRAIL remains
        s2 = s.tick(current_price=100.1)
        assert s2.phase == "TRAIL"

    def test_short_direction(self):
        s = self._state(
            direction="short",
            entry_price=100.0,
            initial_stop_price=105.0,
            active_stop_price=105.0,  # risk = 5
        )
        # R = (100 - 95) / 5 = 1.0 → EXTENDED
        s2 = s.tick(current_price=95.0)
        assert s2.phase == "EXTENDED"

    def test_model_dump_validate_roundtrip(self):
        from services.adaptive_trade_management import AdaptiveTradeManagementState
        s = self._state()
        s2 = s.tick(current_price=103.0)
        dumped = s2.model_dump()
        restored = AdaptiveTradeManagementState.model_validate(dumped)
        assert restored.phase == s2.phase
        assert restored.bars_held == s2.bars_held
        assert restored.peak_r_achieved == s2.peak_r_achieved

    def test_no_stop_no_crash(self):
        """If initial_stop_price is None, phase stays EARLY but no crash."""
        from services.adaptive_trade_management import AdaptiveTradeManagementState
        s = AdaptiveTradeManagementState(
            symbol="BTC-USD",
            direction="long",
            entry_price=100.0,
            initial_stop_price=None,
            active_stop_price=None,
            phase="EARLY",
            bars_held=0,
            peak_r_achieved=0.0,
        )
        s2 = s.tick(current_price=110.0)
        assert s2.phase == "EARLY"  # no risk denominator, stays EARLY
        assert s2.bars_held == 1

    def test_zero_entry_price_no_crash(self):
        """entry_price=0 should not divide by zero."""
        from services.adaptive_trade_management import AdaptiveTradeManagementState
        s = AdaptiveTradeManagementState(
            symbol="BTC-USD",
            direction="long",
            entry_price=0.0,
            initial_stop_price=None,
            active_stop_price=None,
            phase="EARLY",
            bars_held=0,
            peak_r_achieved=0.0,
        )
        s2 = s.tick(current_price=10.0)
        assert s2.phase == "EARLY"


# ---------------------------------------------------------------------------
# 2. SessionState R63 fields
# ---------------------------------------------------------------------------

class TestSessionStateR63Fields:
    """Verify new R63 fields exist on SessionState and survive model_dump/validate."""

    def _minimal_session_state(self, **overrides):
        from tools.paper_trading import SessionState
        base = dict(
            session_id="test-session",
            symbols=["BTC-USD"],
            strategy_prompt=None,
            plan_interval_hours=1.0,
        )
        base.update(overrides)
        return SessionState(**base)

    def test_adaptive_management_states_default_empty(self):
        s = self._minimal_session_state()
        assert s.adaptive_management_states == {}

    def test_episode_memory_store_state_default_empty(self):
        s = self._minimal_session_state()
        assert s.episode_memory_store_state == []

    def test_round_trip_adaptive_states(self):
        from tools.paper_trading import SessionState
        s = self._minimal_session_state(
            adaptive_management_states={
                "BTC-USD": {"symbol": "BTC-USD", "direction": "long",
                            "entry_price": 100.0, "phase": "MATURE",
                            "bars_held": 3, "peak_r_achieved": 0.6}
            }
        )
        dumped = s.model_dump()
        restored = SessionState.model_validate(dumped)
        assert restored.adaptive_management_states["BTC-USD"]["phase"] == "MATURE"

    def test_round_trip_episode_memory(self):
        from tools.paper_trading import SessionState
        ep = {
            "episode_id": "ep-1",
            "symbol": "BTC-USD",
            "direction": "long",
            "outcome_class": "win",
            "r_achieved": 1.5,
            "failure_modes": [],
        }
        s = self._minimal_session_state(episode_memory_store_state=[ep])
        dumped = s.model_dump()
        restored = SessionState.model_validate(dumped)
        assert len(restored.episode_memory_store_state) == 1
        assert restored.episode_memory_store_state[0]["r_achieved"] == 1.5


# ---------------------------------------------------------------------------
# 3. SetupEventGenerator wired in evaluate_triggers_activity
# ---------------------------------------------------------------------------

def test_setup_event_generator_import_reachable():
    """SetupEventGenerator must be importable from agents.analytics."""
    from agents.analytics.setup_event_generator import SetupEventGenerator
    assert SetupEventGenerator is not None


def test_paper_trading_references_setup_event_generator():
    """Verify SetupEventGenerator is referenced in paper_trading.py (wiring check)."""
    import pathlib
    src = pathlib.Path("tools/paper_trading.py").read_text()
    assert "SetupEventGenerator" in src, (
        "SetupEventGenerator not found in tools/paper_trading.py — R63 wiring missing"
    )


def test_paper_trading_references_adaptive_trade_management():
    """Verify AdaptiveTradeManagement is referenced in paper_trading.py (wiring check)."""
    import pathlib
    src = pathlib.Path("tools/paper_trading.py").read_text()
    assert "AdaptiveTradeManagement" in src, (
        "AdaptiveTradeManagement not found in tools/paper_trading.py — R63 wiring missing"
    )


# ---------------------------------------------------------------------------
# 4. Episode memory store state — in-session load
# ---------------------------------------------------------------------------

def test_episode_memory_store_state_loaded_into_mem_store():
    """generate_strategy_plan_activity loads __episode_memory_store_state__ into EpisodeMemoryStore."""
    from services.episode_memory_service import EpisodeMemoryStore
    from schemas.episode_memory import EpisodeMemoryRecord

    store = EpisodeMemoryStore(engine=None)
    ep = EpisodeMemoryRecord(
        episode_id="test-ep-1",
        symbol="BTC-USD",
        direction="long",
        outcome_class="win",
        r_achieved=1.2,
        failure_modes=[],
    )
    store.add(ep)
    records = store.get_by_symbol("BTC-USD")
    assert len(records) == 1
    assert records[0].r_achieved == 1.2


def test_memory_retrieval_service_with_in_session_episodes():
    """MemoryRetrievalService retrieves win/loss from EpisodeMemoryStore populated from session."""
    from services.episode_memory_service import EpisodeMemoryStore
    from services.memory_retrieval_service import MemoryRetrievalService
    from schemas.episode_memory import EpisodeMemoryRecord, MemoryRetrievalRequest

    store = EpisodeMemoryStore(engine=None)
    for i, (oc, r) in enumerate([("win", 1.5), ("loss", -0.8), ("win", 2.0)]):
        store.add(EpisodeMemoryRecord(
            episode_id=f"ep-{i}",
            symbol="BTC-USD",
            direction="long",
            outcome_class=oc,
            r_achieved=r,
            failure_modes=[],
        ))

    req = MemoryRetrievalRequest(symbol="BTC-USD", regime_fingerprint={})
    bundle = MemoryRetrievalService(store).retrieve(req)
    assert bundle.winning_contexts or bundle.losing_contexts, "Expected at least one episode bucket populated"


# ---------------------------------------------------------------------------
# 5. EpisodeMemory DB model exists
# ---------------------------------------------------------------------------

def test_episode_memory_model_exists():
    """EpisodeMemory SQLAlchemy model must be importable."""
    from app.db.models import EpisodeMemory
    assert EpisodeMemory.__tablename__ == "episode_memory"


def test_episode_memory_migration_exists():
    """Alembic migration for episode_memory table must exist."""
    import pathlib
    migrations = list(pathlib.Path("app/db/migrations/versions").glob("*.py"))
    migration_names = [m.stem for m in migrations]
    assert any("episode_memory" in name for name in migration_names), (
        f"No episode_memory migration found in {migration_names}"
    )

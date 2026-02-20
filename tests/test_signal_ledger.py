"""Tests for Signal Ledger (Runbook 43).

Tests use in-memory SQLite to avoid requiring a real Postgres DB.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy as sa

from schemas.signal_event import SignalEvent
from services.signal_ledger_service import SignalLedgerService, compute_regime_snapshot_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sqlite_engine():
    """Create an in-memory SQLite engine with the signal_ledger table."""
    engine = sa.create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS signal_ledger (
                signal_id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                engine_version TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                trigger_id TEXT NOT NULL,
                regime_snapshot_hash TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                stop_anchor_type TEXT,
                target_anchor_type TEXT,
                risk_r_multiple REAL NOT NULL,
                expected_hold_bars INTEGER NOT NULL,
                valid_until TEXT NOT NULL,
                thesis TEXT,
                screener_rank INTEGER,
                confidence TEXT,
                outcome TEXT,
                outcome_ts TEXT,
                r_achieved REAL,
                mfe_pct REAL,
                mae_pct REAL,
                fill_price REAL,
                fill_ts TEXT,
                fill_latency_ms INTEGER,
                slippage_bps REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """))
    return engine


def _make_signal(**overrides) -> SignalEvent:
    """Build a minimal valid SignalEvent."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        engine_version="1.0.0",
        ts=now,
        valid_until=now + timedelta(hours=48),
        timeframe="1h",
        symbol="BTC-USD",
        direction="long",
        trigger_id="TRIG_001",
        strategy_type="compression_breakout",
        regime_snapshot_hash="abc123",
        entry_price=50000.0,
        stop_price_abs=49000.0,
        target_price_abs=52000.0,
        risk_r_multiple=2.0,
        expected_hold_bars=12,
        thesis="BTC breaking compression range with volume.",
    )
    defaults.update(overrides)
    return SignalEvent(**defaults)


# ---------------------------------------------------------------------------
# schema tests
# ---------------------------------------------------------------------------

class TestSignalEventSchema:
    def test_signal_event_schema_required_fields(self):
        """SignalEvent validates with all required fields."""
        signal = _make_signal()
        assert signal.signal_id  # uuid generated
        assert signal.engine_version == "1.0.0"
        assert signal.direction == "long"

    def test_extra_fields_forbidden(self):
        """extra='forbid' rejects unknown fields."""
        with pytest.raises(Exception):
            SignalEvent(
                engine_version="1.0.0",
                ts=datetime.now(timezone.utc),
                valid_until=datetime.now(timezone.utc) + timedelta(hours=48),
                timeframe="1h",
                symbol="BTC-USD",
                direction="long",
                trigger_id="T1",
                strategy_type="trend",
                regime_snapshot_hash="abc",
                entry_price=50000.0,
                stop_price_abs=49000.0,
                target_price_abs=52000.0,
                risk_r_multiple=2.0,
                expected_hold_bars=12,
                thesis="test",
                unknown_field="bad",  # should be rejected
            )

    def test_optional_fields_default_none(self):
        """Optional fields default to None."""
        signal = _make_signal()
        assert signal.screener_rank is None
        assert signal.confidence is None
        assert signal.stop_anchor_type is None
        assert signal.target_anchor_type is None
        assert signal.setup_event_id is None
        assert signal.model_score is None

    def test_feature_schema_version_default(self):
        """feature_schema_version defaults to '1.2.0'."""
        signal = _make_signal()
        assert signal.feature_schema_version == "1.2.0"


# ---------------------------------------------------------------------------
# compute_regime_snapshot_hash
# ---------------------------------------------------------------------------

class TestRegimeSnapshotHash:
    def test_hash_is_deterministic(self):
        """Same dict produces same hash regardless of key insertion order."""
        snap = {"rsi_14": 55.2, "atr_14": 100.5, "close": 50000.0}
        snap_reordered = {"close": 50000.0, "atr_14": 100.5, "rsi_14": 55.2}

        h1 = compute_regime_snapshot_hash(snap)
        h2 = compute_regime_snapshot_hash(snap_reordered)
        assert h1 == h2

    def test_hash_changes_on_value_change(self):
        """Different values produce different hashes."""
        snap1 = {"rsi_14": 55.2, "close": 50000.0}
        snap2 = {"rsi_14": 56.0, "close": 50000.0}
        assert compute_regime_snapshot_hash(snap1) != compute_regime_snapshot_hash(snap2)

    def test_hash_is_sha256_hex(self):
        """Hash is a 64-char hex string (SHA-256)."""
        snap = {"rsi_14": 55.2}
        h = compute_regime_snapshot_hash(snap)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# SignalLedgerService — insert_signal
# ---------------------------------------------------------------------------

class TestSignalLedgerServiceInsert:
    def test_insert_signal_idempotent(self):
        """Inserting the same signal_id twice does not raise (ON CONFLICT DO NOTHING)."""
        engine = _make_sqlite_engine()
        svc = SignalLedgerService(engine=engine)
        signal = _make_signal()
        svc.insert_signal(signal)
        svc.insert_signal(signal)  # second insert should not raise

        with engine.connect() as conn:
            count = conn.execute(sa.text(
                "SELECT COUNT(*) FROM signal_ledger WHERE signal_id = :sid"
            ), {"sid": signal.signal_id}).scalar()
        assert count == 1

    def test_insert_signal_persists_fields(self):
        """Inserted signal has correct field values."""
        engine = _make_sqlite_engine()
        svc = SignalLedgerService(engine=engine)
        signal = _make_signal(symbol="ETH-USD", direction="short", risk_r_multiple=3.5)
        svc.insert_signal(signal)

        with engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT * FROM signal_ledger WHERE signal_id = :sid"
            ), {"sid": signal.signal_id}).fetchone()

        assert row is not None
        assert row.symbol == "ETH-USD"
        assert row.direction == "short"
        assert float(row.risk_r_multiple) == pytest.approx(3.5, rel=1e-3)

    def test_insert_with_no_engine_is_no_op(self):
        """Service with no engine silently skips insert."""
        svc = SignalLedgerService(engine=None)
        # patch env so no engine is created
        with patch.dict("os.environ", {"SIGNAL_LEDGER_ENABLED": "0", "DB_DSN": ""}):
            import importlib
            import services.signal_ledger_service as mod
            orig = mod._SIGNAL_LEDGER_ENABLED
            mod._SIGNAL_LEDGER_ENABLED = False
            svc2 = SignalLedgerService(engine=None)
            svc2.insert_signal(_make_signal())  # should not raise
            mod._SIGNAL_LEDGER_ENABLED = orig


# ---------------------------------------------------------------------------
# SignalLedgerService — record_fill / slippage
# ---------------------------------------------------------------------------

class TestSignalLedgerRecordFill:
    def _insert_and_get_service(self):
        engine = _make_sqlite_engine()
        svc = SignalLedgerService(engine=engine)
        signal = _make_signal()
        svc.insert_signal(signal)
        return svc, signal, engine

    def test_record_fill_computes_slippage_bps_positive(self):
        """Positive slippage (fill above signal price) computed correctly."""
        svc, signal, engine = self._insert_and_get_service()
        signal_entry = signal.entry_price  # 50000.0
        fill_price = 50100.0  # 0.2% above = 20 bps
        now = datetime.now(timezone.utc)

        svc.record_fill(
            signal_id=signal.signal_id,
            fill_price=fill_price,
            fill_ts=now + timedelta(seconds=5),
            signal_ts=now,
            signal_entry_price=signal_entry,
        )

        with engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT slippage_bps, fill_latency_ms FROM signal_ledger WHERE signal_id = :sid"
            ), {"sid": signal.signal_id}).fetchone()

        expected_bps = (fill_price - signal_entry) / signal_entry * 10_000
        assert row is not None
        assert float(row.slippage_bps) == pytest.approx(expected_bps, rel=1e-3)

    def test_record_fill_computes_slippage_bps_negative(self):
        """Negative slippage (fill below signal price) computed correctly."""
        svc, signal, engine = self._insert_and_get_service()
        signal_entry = signal.entry_price  # 50000.0
        fill_price = 49900.0  # fill below signal price
        now = datetime.now(timezone.utc)

        svc.record_fill(
            signal_id=signal.signal_id,
            fill_price=fill_price,
            fill_ts=now,
            signal_ts=now,
            signal_entry_price=signal_entry,
        )

        expected_bps = (fill_price - signal_entry) / signal_entry * 10_000
        assert expected_bps < 0

        with engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT slippage_bps FROM signal_ledger WHERE signal_id = :sid"
            ), {"sid": signal.signal_id}).fetchone()
        assert float(row.slippage_bps) == pytest.approx(expected_bps, rel=1e-3)

    def test_slippage_warning_threshold(self, caplog):
        """Warning logged when |slippage_bps| > 30."""
        svc, signal, engine = self._insert_and_get_service()
        signal_entry = 50000.0
        fill_price = 50200.0  # 0.4% above = 40 bps > 30 threshold
        now = datetime.now(timezone.utc)

        with caplog.at_level(logging.WARNING, logger="services.signal_ledger_service"):
            svc.record_fill(
                signal_id=signal.signal_id,
                fill_price=fill_price,
                fill_ts=now,
                signal_ts=now,
                signal_entry_price=signal_entry,
            )

        assert any("SLIPPAGE WARNING" in r.message for r in caplog.records)

    def test_no_slippage_warning_below_threshold(self, caplog):
        """No warning when slippage is within threshold."""
        svc, signal, engine = self._insert_and_get_service()
        signal_entry = 50000.0
        fill_price = 50050.0  # 0.1% = 10 bps < 30 threshold
        now = datetime.now(timezone.utc)

        with caplog.at_level(logging.WARNING, logger="services.signal_ledger_service"):
            svc.record_fill(
                signal_id=signal.signal_id,
                fill_price=fill_price,
                fill_ts=now,
                signal_ts=now,
                signal_entry_price=signal_entry,
            )

        assert not any("SLIPPAGE WARNING" in r.message for r in caplog.records)

    def test_fill_latency_computed(self):
        """fill_latency_ms = (fill_ts - signal_ts) * 1000."""
        svc, signal, engine = self._insert_and_get_service()
        base = datetime.now(timezone.utc)
        signal_ts = base
        fill_ts = base + timedelta(seconds=2)

        svc.record_fill(
            signal_id=signal.signal_id,
            fill_price=50000.0,
            fill_ts=fill_ts,
            signal_ts=signal_ts,
            signal_entry_price=50000.0,
        )

        with engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT fill_latency_ms FROM signal_ledger WHERE signal_id = :sid"
            ), {"sid": signal.signal_id}).fetchone()
        assert row.fill_latency_ms == 2000


# ---------------------------------------------------------------------------
# SignalLedgerService — resolve_signal (reconciler outcomes)
# ---------------------------------------------------------------------------

class TestSignalLedgerResolve:
    def _insert_and_get(self):
        engine = _make_sqlite_engine()
        svc = SignalLedgerService(engine=engine)
        signal = _make_signal()
        svc.insert_signal(signal)
        return svc, signal, engine

    def test_resolve_target_hit(self):
        """resolve_signal writes outcome=target_hit and r_achieved."""
        svc, signal, engine = self._insert_and_get()
        now = datetime.now(timezone.utc)
        svc.resolve_signal(
            signal_id=signal.signal_id,
            outcome="target_hit",
            outcome_ts=now,
            r_achieved=2.0,
            mfe_pct=4.0,
            mae_pct=-0.5,
        )
        with engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT outcome, r_achieved FROM signal_ledger WHERE signal_id = :sid"
            ), {"sid": signal.signal_id}).fetchone()
        assert row.outcome == "target_hit"
        assert float(row.r_achieved) == pytest.approx(2.0, rel=1e-3)

    def test_resolve_stop_hit(self):
        """resolve_signal writes outcome=stop_hit with negative r_achieved."""
        svc, signal, engine = self._insert_and_get()
        now = datetime.now(timezone.utc)
        svc.resolve_signal(
            signal_id=signal.signal_id,
            outcome="stop_hit",
            outcome_ts=now,
            r_achieved=-1.0,
            mfe_pct=0.5,
            mae_pct=-2.0,
        )
        with engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT outcome, r_achieved FROM signal_ledger WHERE signal_id = :sid"
            ), {"sid": signal.signal_id}).fetchone()
        assert row.outcome == "stop_hit"
        assert float(row.r_achieved) < 0

    def test_resolve_expired(self):
        """resolve_signal writes outcome=expired."""
        svc, signal, engine = self._insert_and_get()
        now = datetime.now(timezone.utc)
        svc.resolve_signal(
            signal_id=signal.signal_id,
            outcome="expired",
            outcome_ts=now,
            r_achieved=0.0,
            mfe_pct=0.0,
            mae_pct=0.0,
        )
        with engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT outcome FROM signal_ledger WHERE signal_id = :sid"
            ), {"sid": signal.signal_id}).fetchone()
        assert row.outcome == "expired"

    def test_resolve_is_idempotent(self):
        """resolve_signal WHERE outcome IS NULL prevents double-resolution."""
        svc, signal, engine = self._insert_and_get()
        now = datetime.now(timezone.utc)
        svc.resolve_signal(signal.signal_id, "target_hit", now, 2.0, 4.0, -0.5)
        svc.resolve_signal(signal.signal_id, "stop_hit", now, -1.0, 0.1, -2.0)  # should be no-op
        with engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT outcome FROM signal_ledger WHERE signal_id = :sid"
            ), {"sid": signal.signal_id}).fetchone()
        assert row.outcome == "target_hit"  # original preserved


# ---------------------------------------------------------------------------
# Reconciler: outcome detection logic (unit, without DB)
# ---------------------------------------------------------------------------

class TestSignalOutcomeReconcilerLogic:
    """Test reconciler's _resolve_one logic using mocked OHLCV."""

    def _make_reconciler(self, engine):
        from services.signal_outcome_reconciler import SignalOutcomeReconciler
        svc = SignalLedgerService(engine=engine)
        return SignalOutcomeReconciler(ledger_service=svc, engine=engine), svc

    def _make_row(self, direction="long", entry=50000.0, stop=49000.0, target=52000.0):
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            signal_id="test-sig-001",
            ts=now - timedelta(hours=2),
            valid_until=now + timedelta(hours=46),
            symbol="BTC-USD",
            direction=direction,
            timeframe="1h",
            entry_price=entry,
            stop_price=stop,
            target_price=target,
        )

    def test_reconciler_target_hit_long(self):
        """Reconciler resolves target_hit when high >= target."""
        import pandas as pd

        engine = _make_sqlite_engine()
        rec, svc = self._make_reconciler(engine)

        signal = _make_signal()
        svc.insert_signal(signal)

        row = self._make_row(direction="long", entry=50000.0, stop=49000.0, target=52000.0)
        row.signal_id = signal.signal_id

        # Create OHLCV where bar 2 crosses target
        idx = [
            datetime.now(timezone.utc) - timedelta(hours=2),
            datetime.now(timezone.utc) - timedelta(hours=1),
        ]
        df = pd.DataFrame({
            "open":  [50000.0, 51000.0],
            "high":  [51000.0, 52500.0],  # bar 2 crosses target 52000
            "low":   [49500.0, 50500.0],
            "close": [50500.0, 52200.0],
        }, index=idx)

        with patch("backtesting.dataset.load_ohlcv", return_value=df):
            outcome = rec._resolve_one(row)

        assert outcome == "target_hit"

    def test_reconciler_stop_hit_long(self):
        """Reconciler resolves stop_hit when low <= stop."""
        import pandas as pd

        engine = _make_sqlite_engine()
        rec, svc = self._make_reconciler(engine)
        signal = _make_signal()
        svc.insert_signal(signal)

        row = self._make_row(direction="long", entry=50000.0, stop=49000.0, target=52000.0)
        row.signal_id = signal.signal_id

        idx = [
            datetime.now(timezone.utc) - timedelta(hours=2),
            datetime.now(timezone.utc) - timedelta(hours=1),
        ]
        df = pd.DataFrame({
            "open":  [50000.0, 49500.0],
            "high":  [50200.0, 49600.0],
            "low":   [49800.0, 48800.0],  # bar 2 crosses stop 49000
            "close": [49900.0, 49000.0],
        }, index=idx)

        with patch("backtesting.dataset.load_ohlcv", return_value=df):
            outcome = rec._resolve_one(row)

        assert outcome == "stop_hit"

    def test_reconciler_expired(self):
        """Reconciler marks expired when valid_until passes without hit."""
        import pandas as pd

        engine = _make_sqlite_engine()
        rec, svc = self._make_reconciler(engine)
        signal = _make_signal()
        svc.insert_signal(signal)

        now = datetime.now(timezone.utc)
        # Signal already past valid_until
        row = SimpleNamespace(
            signal_id=signal.signal_id,
            ts=now - timedelta(hours=50),
            valid_until=now - timedelta(hours=2),  # already expired
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_price=50000.0,
            stop_price=49000.0,
            target_price=52000.0,
        )

        idx = [now - timedelta(hours=3)]
        df = pd.DataFrame({
            "open": [50000.0], "high": [50500.0], "low": [49800.0], "close": [50200.0]
        }, index=idx)

        with patch("backtesting.dataset.load_ohlcv", return_value=df):
            outcome = rec._resolve_one(row)

        assert outcome == "expired"

    def test_reconciler_short_direction(self):
        """Short signal: target is below entry, stop is above entry."""
        import pandas as pd

        engine = _make_sqlite_engine()
        rec, svc = self._make_reconciler(engine)
        signal = _make_signal(
            direction="short",
            entry_price=50000.0,
            stop_price_abs=51000.0,  # stop above for shorts
            target_price_abs=48000.0,  # target below for shorts
        )
        svc.insert_signal(signal)

        row = self._make_row(direction="short", entry=50000.0, stop=51000.0, target=48000.0)
        row.signal_id = signal.signal_id

        idx = [
            datetime.now(timezone.utc) - timedelta(hours=2),
            datetime.now(timezone.utc) - timedelta(hours=1),
        ]
        df = pd.DataFrame({
            "open":  [50000.0, 49000.0],
            "high":  [50200.0, 49200.0],
            "low":   [49800.0, 47800.0],  # bar 2 crosses target 48000 (low <= target)
            "close": [49900.0, 48200.0],
        }, index=idx)

        with patch("backtesting.dataset.load_ohlcv", return_value=df):
            outcome = rec._resolve_one(row)

        assert outcome == "target_hit"


# ---------------------------------------------------------------------------
# Capital gates
# ---------------------------------------------------------------------------

class TestCapitalGates:
    def test_capital_gates_insufficient_signals(self):
        """gate_pass=False when fewer than 40 resolved signals."""
        engine = _make_sqlite_engine()
        svc = SignalLedgerService(engine=engine)

        # Insert only 5 resolved signals
        for i in range(5):
            signal = _make_signal(risk_r_multiple=2.0)
            svc.insert_signal(signal)
            svc.resolve_signal(signal.signal_id, "target_hit",
                               datetime.now(timezone.utc), 2.0, 4.0, -0.5)

        gates = svc.evaluate_capital_gates()
        assert gates["gate_pass"] is False
        assert gates["resolved_count"] == 5
        assert gates["conditions"]["resolved_count_ge_40"] is False

    def test_capital_gates_negative_expectancy(self):
        """gate_pass=False when mean_r <= 0."""
        engine = _make_sqlite_engine()
        svc = SignalLedgerService(engine=engine)

        # Insert 50 resolved signals all with negative r
        for i in range(50):
            signal = _make_signal()
            svc.insert_signal(signal)
            svc.resolve_signal(signal.signal_id, "stop_hit",
                               datetime.now(timezone.utc), -1.0, 0.5, -2.0)

        gates = svc.evaluate_capital_gates()
        assert gates["gate_pass"] is False
        assert gates["conditions"]["expectancy_positive"] is False

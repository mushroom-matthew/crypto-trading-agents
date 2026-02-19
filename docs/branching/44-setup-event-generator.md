# Branch: setup-event-generator

## Priority: P1 (Data plane prerequisite for model training)

## Purpose

The system can detect setups and place trades, but it cannot answer: "Was this a good setup before the outcome was known?" Without a frozen feature record at the moment of detection — not at fill time, not at outcome — there is no way to train a model that scores setups at decision time.

This runbook adds the **Setup Event Generator**: a per-symbol state machine that detects compression and break-attempt state transitions, freezes the full feature snapshot at each transition, and writes a `SetupEvent` to a durable ledger. A companion **SetupOutcomeReconciler** labels each event after TTL: did price reach 1R before the stop? What was MFE/MAE?

It also defines the **model integration contract**: a narrow `ModelScorePacket` returned by a `ModelScorer` interface. The risk engine consumes the packet for two purposes only:

1. **Hard gate**: `p_false_breakout > 0.40` blocks the entry
2. **Sizing multiplier**: `clamp(0.5 + 1.0 * (quality - 0.5), 0.5, 1.25)`

The default `NullModelScorer` returns `None` — no model trained yet means no effect on sizing or gating. This design is intentionally narrow: the model scores setups, the LLM writes plans, the engine executes and risks. No overlap.

This runbook depends on:
- **Runbook 42** (level-anchored stops) — provides stable `stop_price_abs` and `target_price_abs` at fill time. Without immutable stop/target anchors, outcome labels drift with rolling indicators and become meaningless for model training.
- **Runbook 43** (signal ledger) — `SignalEvent` links back to `SetupEvent` via `setup_event_id`, enabling join between setup context and execution outcome.

## Design Decisions Locked In

| Decision | Value |
|---|---|
| Model output contract | `ModelScorePacket`: `model_quality_score (0–1)`, optional `p_cont_1R`, `p_false_breakout`, `p_atr_expand` |
| Hard gate threshold | `p_false_breakout > 0.40` → block entry |
| Sizing formula | `clamp(0.5 + 1.0 * (quality - 0.5), 0.5, 1.25)` |
| Default scorer | `NullModelScorer` — returns `None`, zero effect on sizing/gating |
| Score at CompressionCandidate | Yes — for watchlist ranking (not gating) |
| Score at BreakAttempt | Yes — for gating + sizing multiplier |
| First model type | LightGBM/XGBoost on frozen features (no transformer until ≥5000 labeled events) |
| Label integrity | Labels derived from `stop_price_abs` / `target_price_abs` (Runbook 42). Do not use rolling indicators as label anchors. |
| Version surfaces | `FEATURE_SCHEMA_VERSION`, `ENGINE_SEMVER`, `strategy_template_version` — all stored in `SetupEvent` |

## Scope

1. **`schemas/feature_version.py`** — new: `FEATURE_SCHEMA_VERSION` constant + bump protocol
2. **`schemas/setup_event.py`** — new: `SetupEvent`, `SetupState`, session context fields
3. **`schemas/model_score.py`** — new: `ModelScorePacket` (the model's return type)
4. **`services/model_scorer.py`** — new: `ModelScorer` abstract base + `NullModelScorer` (default)
5. **`agents/analytics/setup_event_generator.py`** — new: `SetupEventGenerator` per-symbol state machine
6. **`backtesting/llm_strategist_runner.py`** — wire: call generator per bar, apply gating + sizing multiplier at entry
7. **`schemas/signal_event.py`** — extend: add `setup_event_id`, `feature_schema_version`, `strategy_template_version`, `model_score` to `SignalEvent`
8. **`app/db/migrations/`** — new Alembic migration: `setup_event_ledger` table
9. **`services/setup_outcome_reconciler.py`** — new: labels open `SetupEvent` rows after TTL using OHLCV
10. **`app/cli/main.py`** — add: `setup-events export-parquet` CLI command
11. **`tests/test_setup_event_generator.py`** — new: unit + integration tests

## Out of Scope

- Training the model itself (that is a Jupyter notebook / script, not a runbook)
- Model serving infrastructure (HTTP inference endpoint, ONNX export)
- Bar-by-bar model calls for the full universe (event-conditioned only)
- Automated capital stage promotion from setup performance (human-approved)
- Replay of historical events from existing fill records (backfill is a future migration)
- Multi-leg setup tracking (one `SetupEvent` per state transition, scale-ins are separate)

## Key Files

- `schemas/feature_version.py` (new)
- `schemas/setup_event.py` (new)
- `schemas/model_score.py` (new)
- `services/model_scorer.py` (new)
- `agents/analytics/setup_event_generator.py` (new)
- `backtesting/llm_strategist_runner.py` (modify)
- `schemas/signal_event.py` (modify)
- `app/db/migrations/versions/XXXX_add_setup_event_ledger.py` (new)
- `services/setup_outcome_reconciler.py` (new)
- `tests/test_setup_event_generator.py` (new)

---

## Implementation Steps

### Step 1: Feature schema versioning — `schemas/feature_version.py`

```python
"""Feature schema version constants.

Bump FEATURE_SCHEMA_VERSION whenever IndicatorSnapshot fields change:
  - Fields added: increment minor (1.2.0 → 1.3.0)
  - Fields removed or renamed: increment major (1.3.0 → 2.0.0)

This constant is stored in every SetupEvent and SignalEvent row.
Training data MUST be stratified by feature_schema_version before fitting.
Never pool events across major versions.
"""

FEATURE_SCHEMA_VERSION = "1.2.0"
# History:
#   1.0.0 — initial IndicatorSnapshot (base indicators through scalper fields)
#   1.1.0 — added Fibonacci + expansion/contraction ratios
#   1.2.0 — added 15 candlestick fields (R38) + 13 htf_* fields (R41)
```

And engine versioning — add to `backtesting/constants.py` (create if not exists):
```python
import os

# Semver of the strategy engine. Bump on each merged runbook that changes
# strategy behavior. Training data MUST be stratified by ENGINE_SEMVER.
ENGINE_SEMVER: str = os.environ.get("ENGINE_SEMVER", "0.5.0")
```

### Step 2: Define `SetupEvent` in `schemas/setup_event.py`

```python
"""
SetupEvent schema — frozen research record at a setup state transition.

DISCLAIMER: Setup events are research observations, not personalized investment
advice. They carry no sizing. Subscribers apply their own risk rules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


SetupState = Literal[
    "compression_candidate",  # Compression detected; watching for break
    "break_attempt",          # Price crossing range edge with impulse/volume
    "confirmed",              # Price reached 1R target before stop
    "failed",                 # Price hit stop before 1R target
    "ttl_expired",            # TTL elapsed without resolution
]


class SessionContext(BaseModel):
    """Lightweight session embedding for domain/time conditioning."""

    model_config = {"extra": "forbid"}

    session_type: str = Field(
        description=(
            "Session identifier: 'crypto_utc_day', 'crypto_asia', "
            "'crypto_london', 'crypto_us', 'nyse_rth', 'nyse_pm', 'nyse_ah'."
        ),
    )
    time_in_session_sin: float = Field(
        description="Sine of fractional time-in-session (cyclic encoding).",
    )
    time_in_session_cos: float = Field(
        description="Cosine of fractional time-in-session (cyclic encoding).",
    )
    minutes_to_session_close: Optional[float] = Field(
        default=None,
        description="Minutes until session ends. None for continuous markets.",
    )
    is_weekend: bool = Field(
        description="True if bar is in a weekend period (crypto only).",
    )
    asset_class: str = Field(
        default="crypto",
        description="Asset class domain token: 'crypto', 'equity', 'fx'.",
    )


class SetupEvent(BaseModel):
    """Frozen record of a setup state transition.

    One row per state transition. Multiple rows share the same
    `setup_chain_id` if they are the same setup lifecycle
    (compression → break → confirmed/failed).

    DISCLAIMER: Research observation only. Not personalized investment advice.
    """

    model_config = {"extra": "forbid"}

    # --- Identity ---
    setup_event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this individual state transition record.",
    )
    setup_chain_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description=(
            "Shared ID for all events in the same setup lifecycle "
            "(compression → break → outcome). Set at CompressionCandidate; "
            "inherited by subsequent transitions."
        ),
    )
    state: SetupState = Field(
        description="State machine state at the time of this event.",
    )

    # --- Provenance / version ---
    engine_semver: str = Field(
        description="Engine semver at emission time (from ENGINE_SEMVER constant).",
    )
    feature_schema_version: str = Field(
        description="IndicatorSnapshot schema version (from FEATURE_SCHEMA_VERSION).",
    )
    strategy_template_version: Optional[str] = Field(
        default=None,
        description=(
            "Active strategy template name (e.g., 'compression_breakout_v1', "
            "'htf_cascade_v1'). Null if no template active."
        ),
    )

    # --- Time / instrument ---
    ts: datetime = Field(
        description="UTC bar timestamp when this state transition was detected.",
    )
    symbol: str
    timeframe: str

    # --- Session context (for model domain conditioning) ---
    session: SessionContext = Field(
        description="Session and domain tokens for model conditioning.",
    )

    # --- Frozen feature snapshot ---
    feature_snapshot: Dict[str, Any] = Field(
        description=(
            "Full IndicatorSnapshot serialized to dict at the moment of this "
            "transition. Keys include all indicator fields including candlestick "
            "and htf_* fields. Values are float | bool | str | None. "
            "Do NOT update this field after creation — immutability is what "
            "makes supervised labels meaningful."
        ),
    )
    feature_snapshot_hash: str = Field(
        description=(
            "SHA-256 hex digest of feature_snapshot JSON (sorted keys). "
            "Proves the features were recorded at decision time, not retrofitted."
        ),
    )

    # --- Range context (locked at compression detection) ---
    compression_range_high: Optional[float] = Field(
        default=None,
        description=(
            "donchian_upper_short at time of CompressionCandidate detection. "
            "Used as BreakAttempt criterion — break is defined relative to THIS "
            "range, not the rolling Donchian at break time. Inherited at all "
            "subsequent states in the same chain."
        ),
    )
    compression_range_low: Optional[float] = Field(
        default=None,
        description="donchian_lower_short at time of CompressionCandidate detection.",
    )
    compression_atr_at_detection: Optional[float] = Field(
        default=None,
        description="atr_14 at time of CompressionCandidate detection (for R sizing).",
    )

    # --- Model score (populated if model available at this state) ---
    model_quality_score: Optional[float] = Field(
        default=None,
        description="Model setup quality score (0–1). None if model not available.",
    )
    p_cont_1R: Optional[float] = Field(
        default=None,
        description="Model-estimated probability of hitting 1R before stop.",
    )
    p_false_breakout: Optional[float] = Field(
        default=None,
        description=(
            "Model-estimated probability of false breakout "
            "(return to range and hit stop within TTL). "
            "If > 0.40 at BreakAttempt, entry is BLOCKED."
        ),
    )
    p_atr_expand: Optional[float] = Field(
        default=None,
        description="Model-estimated probability of ATR expanding in next 20 bars.",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Model artifact version that produced the score.",
    )

    # --- Outcome (filled by SetupOutcomeReconciler after TTL) ---
    outcome: Optional[Literal["hit_1r", "hit_stop", "ttl_expired"]] = Field(
        default=None,
        description=(
            "Outcome after TTL: hit the 1R target, hit the stop, or expired. "
            "Null until reconciled. This is the primary training label."
        ),
    )
    outcome_ts: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the outcome was resolved.",
    )
    bars_to_outcome: Optional[int] = Field(
        default=None,
        description="Number of bars from BreakAttempt to outcome.",
    )
    mfe_pct: Optional[float] = Field(
        default=None,
        description=(
            "Maximum favorable excursion as % of entry price "
            "(best price reached in the direction of the setup)."
        ),
    )
    mae_pct: Optional[float] = Field(
        default=None,
        description=(
            "Maximum adverse excursion as % of entry price "
            "(worst price reached against the direction of the setup)."
        ),
    )
    r_achieved: Optional[float] = Field(
        default=None,
        description=(
            "R-multiple achieved at outcome: "
            "(outcome_price - entry) / (entry - stop_price). "
            "Positive = favorable, negative = loss."
        ),
    )
    ttl_bars: int = Field(
        default=48,
        description=(
            "Number of bars from BreakAttempt detection before outcome is "
            "forced to 'ttl_expired'. Default: 48 bars (48h on 1h timeframe)."
        ),
    )

    # --- Link to execution ---
    signal_event_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of the SignalEvent that fired from this setup, if any. "
            "Null if setup expired without a trigger firing."
        ),
    )
```

### Step 3: `ModelScorePacket` and `ModelScorer` interface — `schemas/model_score.py` + `services/model_scorer.py`

**`schemas/model_score.py`:**
```python
"""Model integration contract.

The model returns a ModelScorePacket. The risk engine consumes it for:
  1. Hard gate:  p_false_breakout > 0.40 → block entry
  2. Sizing:     size_multiplier = clamp(0.5 + 1.0 * (quality - 0.5), 0.5, 1.25)

When model_quality_score is None, the engine behaves as if no model exists:
  - size_multiplier = 1.0 (no effect)
  - hard gate = False (not blocked)
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ModelScorePacket(BaseModel):
    """Output contract from ModelScorer.score(). Consumed by risk engine."""

    model_config = {"extra": "forbid"}

    model_quality_score: Optional[float] = Field(
        default=None,
        description="Overall setup quality (0–1). None if model not available.",
    )
    p_cont_1R: Optional[float] = Field(
        default=None,
        description="P(price hits 1R before stop within TTL).",
    )
    p_false_breakout: Optional[float] = Field(
        default=None,
        description=(
            "P(price returns inside compression range and hits stop within TTL). "
            "Hard gate threshold: > 0.40 blocks entry."
        ),
    )
    p_atr_expand: Optional[float] = Field(
        default=None,
        description="P(ATR expands > 1.5x over next 20 bars).",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Model artifact version (e.g., 'lgbm-v1.0.2').",
    )
    calibration_bucket: Optional[str] = Field(
        default=None,
        description=(
            "Calibration group for this prediction (e.g., 'crypto_largecap', "
            "'crypto_midcap'). Used to stratify reliability analysis."
        ),
    )
```

**`services/model_scorer.py`:**
```python
"""ModelScorer interface and default NullModelScorer.

The ModelScorer is the only interface between the model training pipeline
and the live/backtest execution engine. Keep it narrow.

Implementations:
  NullModelScorer  — always returns None scores (default, no model trained)
  LightGBMScorer   — loads a serialized LightGBM model from disk (future)
  XGBoostScorer    — loads a serialized XGBoost model from disk (future)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

from schemas.model_score import ModelScorePacket


class ModelScorer(ABC):
    """Abstract base: score a setup given its frozen feature context."""

    @abstractmethod
    def score(self, feature_snapshot: Dict[str, Any]) -> ModelScorePacket:
        """Score a setup from its frozen feature snapshot.

        Args:
            feature_snapshot: SetupEvent.feature_snapshot dict (serialized
                IndicatorSnapshot). Keys are feature names, values are
                float | bool | None.

        Returns:
            ModelScorePacket with model_quality_score and optional sub-scores.
            All fields None if model unavailable or input is insufficient.
        """
        ...

    def is_entry_blocked(self, packet: ModelScorePacket) -> bool:
        """Hard gate: block entry if p_false_breakout > 0.40."""
        if packet.p_false_breakout is None:
            return False
        return packet.p_false_breakout > 0.40

    def size_multiplier(self, packet: ModelScorePacket) -> float:
        """Sizing multiplier from model_quality_score.

        Returns 1.0 (no effect) when model_quality_score is None.
        Clamps to [0.5, 1.25] to prevent extreme sizing.
        """
        if packet.model_quality_score is None:
            return 1.0
        q = packet.model_quality_score
        return max(0.5, min(1.25, 0.5 + 1.0 * (q - 0.5)))


class NullModelScorer(ModelScorer):
    """Default scorer — no trained model.

    Always returns a ModelScorePacket with all None scores.
    size_multiplier returns 1.0, is_entry_blocked returns False.
    This is the correct default: no model means no effect.
    """

    def score(self, feature_snapshot: Dict[str, Any]) -> ModelScorePacket:
        return ModelScorePacket()
```

### Step 4: `SetupEventGenerator` — `agents/analytics/setup_event_generator.py`

The generator maintains a per-symbol state machine. Called once per bar from the backtest runner. Emits `SetupEvent` instances on state transitions.

```python
"""Per-symbol state machine that detects compression → break attempt transitions
and emits SetupEvents for the training data pipeline.

State lifecycle:
    IDLE
      ↓  _is_compression_candidate()
    COMPRESSION_CANDIDATE  (emits SetupEvent, freezes range)
      ↓  _is_break_attempt()  (using locked range from candidate state)
    BREAK_ATTEMPT          (emits SetupEvent, triggers model scoring)
      ↓  outcome reconciler fills later
    CONFIRMED | FAILED | TTL_EXPIRED
      ↓  → back to IDLE (or back to COMPRESSION_CANDIDATE if still compressed)

Detection criteria (intentionally simple — tune after first 200 labeled events):

CompressionCandidate (all required):
  - is_inside_bar == 1.0  (current range inside prior bar)
  - candle_body_pct < 0.40  (weak directional conviction)
  - vol_state in ('low', 'normal')  (not already in volatility expansion)
  - htf_price_vs_daily_mid: abs value < 0.75  (near daily midpoint, not at structural extreme)
    (condition skipped if htf_price_vs_daily_mid is None — insufficient history)

BreakAttempt (from COMPRESSION_CANDIDATE state):
  - close > compression_range_high * 1.001  (upside break, 0.1% noise filter)
    OR close < compression_range_low * 0.999  (downside break)
  - is_impulse_candle == 1.0 OR vol_burst == 1  (impulse confirmation)
  - Within compression_ttl_bars of CompressionCandidate (default: 24 bars)
    Prevents stale setups from triggering on unrelated moves.

TTL expiry from BreakAttempt: outcome_ttl_bars (default: 48 bars).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from schemas.feature_version import FEATURE_SCHEMA_VERSION
from schemas.setup_event import SessionContext, SetupEvent, SetupState
from schemas.llm_strategist import IndicatorSnapshot
from services.model_scorer import ModelScorer, NullModelScorer


# Compression detection thresholds (tune after ≥200 labeled events)
_COMPRESSION_BODY_PCT_MAX = 0.40
_COMPRESSION_HTF_MID_ABS_MAX = 0.75
_BREAK_NOISE_FILTER = 0.001   # 0.1% — prevents false breaks from spread noise
_COMPRESSION_TTL_BARS = 24    # Max bars to wait for break after compression
_OUTCOME_TTL_BARS = 48        # Max bars to wait for outcome after break attempt


def _hash_snapshot(snapshot_dict: dict) -> str:
    canonical = json.dumps(snapshot_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _session_context_for(ts: datetime, symbol: str) -> SessionContext:
    """Compute session context tokens for a bar timestamp.

    For crypto, defines four synthetic liquidity windows:
      Asia:   00:00–08:00 UTC
      London: 08:00–13:00 UTC
      US:     13:00–20:00 UTC
      Close:  20:00–24:00 UTC
    """
    hour = ts.hour + ts.minute / 60.0
    session_boundaries = [
        ("crypto_asia",   0.0,  8.0),
        ("crypto_london", 8.0, 13.0),
        ("crypto_us",    13.0, 20.0),
        ("crypto_close", 20.0, 24.0),
    ]
    session_type = "crypto_utc_day"
    session_start, session_end = 0.0, 24.0
    for name, start, end in session_boundaries:
        if start <= hour < end:
            session_type = name
            session_start, session_end = start, end
            break

    session_len = session_end - session_start
    frac = (hour - session_start) / session_len if session_len > 0 else 0.0
    angle = 2 * math.pi * frac
    is_weekend = ts.weekday() >= 5  # Saturday=5, Sunday=6

    return SessionContext(
        session_type=session_type,
        time_in_session_sin=math.sin(angle),
        time_in_session_cos=math.cos(angle),
        minutes_to_session_close=max(0.0, (session_end - hour) * 60),
        is_weekend=is_weekend,
        asset_class="crypto",
    )


@dataclass
class _SymbolState:
    """In-memory state machine state for one symbol."""
    state: SetupState = "idle"  # type: ignore[assignment]  (idle is not in the Literal)
    setup_chain_id: str = ""
    compression_range_high: Optional[float] = None
    compression_range_low: Optional[float] = None
    compression_atr: Optional[float] = None
    compression_ts: Optional[datetime] = None
    compression_bars_elapsed: int = 0
    break_ts: Optional[datetime] = None
    break_bars_elapsed: int = 0


class SetupEventGenerator:
    """Per-symbol state machine that detects and emits SetupEvents.

    Usage:
        gen = SetupEventGenerator(engine_semver="0.5.0", scorer=NullModelScorer())
        for bar_ts, snapshot in per_bar_loop:
            events = gen.on_bar(symbol, timeframe, bar_ts, snapshot)
            for evt in events:
                ledger_service.write(evt)
    """

    def __init__(
        self,
        engine_semver: str,
        scorer: ModelScorer | None = None,
        strategy_template_version: str | None = None,
        compression_ttl_bars: int = _COMPRESSION_TTL_BARS,
        outcome_ttl_bars: int = _OUTCOME_TTL_BARS,
    ) -> None:
        self._engine_semver = engine_semver
        self._scorer = scorer or NullModelScorer()
        self._template_version = strategy_template_version
        self._compression_ttl = compression_ttl_bars
        self._outcome_ttl = outcome_ttl_bars
        self._states: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        timeframe: str,
        bar_ts: datetime,
        snapshot: IndicatorSnapshot,
    ) -> List[SetupEvent]:
        """Process one bar. Returns 0 or more SetupEvents emitted this bar."""
        key = f"{symbol}:{timeframe}"
        if key not in self._states:
            self._states[key] = _SymbolState()
        state = self._states[key]
        events: List[SetupEvent] = []

        if state.state == "idle":
            if self._is_compression_candidate(snapshot):
                evt = self._emit_compression_candidate(symbol, timeframe, bar_ts, snapshot, state)
                events.append(evt)

        elif state.state == "compression_candidate":
            state.compression_bars_elapsed += 1
            if state.compression_bars_elapsed > self._compression_ttl:
                # Compression resolved without break — return to idle
                state.state = "idle"
            elif self._is_break_attempt(snapshot, state):
                evt = self._emit_break_attempt(symbol, timeframe, bar_ts, snapshot, state)
                events.append(evt)
            elif not self._is_still_compressed(snapshot):
                # Compression invalidated (e.g., large bar expanded range)
                state.state = "idle"

        elif state.state == "break_attempt":
            state.break_bars_elapsed += 1
            if state.break_bars_elapsed > self._outcome_ttl:
                # TTL: outcome_reconciler will label this after the fact
                state.state = "idle"

        # NOTE: 'confirmed', 'failed', 'ttl_expired' are terminal — only the
        # SetupOutcomeReconciler writes these outcomes to the ledger row.
        # The in-memory state machine resets to idle immediately after break_attempt
        # TTL regardless of reconciled outcome.

        return events

    def reset(self, symbol: str, timeframe: str) -> None:
        """Reset state machine for a symbol/timeframe (e.g., at day boundaries)."""
        key = f"{symbol}:{timeframe}"
        if key in self._states:
            self._states[key] = _SymbolState()

    # ------------------------------------------------------------------
    # Detection predicates
    # ------------------------------------------------------------------

    def _is_compression_candidate(self, snapshot: IndicatorSnapshot) -> bool:
        if snapshot.vol_state == "extreme":
            return False
        has_inside_bar = (snapshot.is_inside_bar or 0.0) >= 1.0
        low_conviction = (snapshot.candle_body_pct or 1.0) < _COMPRESSION_BODY_PCT_MAX
        htf_near_mid = True
        if snapshot.htf_price_vs_daily_mid is not None:
            htf_near_mid = abs(snapshot.htf_price_vs_daily_mid) < _COMPRESSION_HTF_MID_ABS_MAX
        return has_inside_bar and low_conviction and htf_near_mid

    def _is_break_attempt(self, snapshot: IndicatorSnapshot, state: _SymbolState) -> bool:
        if state.compression_range_high is None or state.compression_range_low is None:
            return False
        close = snapshot.close
        upside_break = close > state.compression_range_high * (1 + _BREAK_NOISE_FILTER)
        downside_break = close < state.compression_range_low * (1 - _BREAK_NOISE_FILTER)
        if not (upside_break or downside_break):
            return False
        impulse = (snapshot.is_impulse_candle or 0.0) >= 1.0
        vol_burst = bool(snapshot.vol_burst)
        return impulse or vol_burst

    def _is_still_compressed(self, snapshot: IndicatorSnapshot) -> bool:
        """Check if compression conditions still hold (prevents staying in state on expansion)."""
        if snapshot.vol_state == "extreme":
            return False
        # Impulse candle = compression broke — go back to idle
        if (snapshot.is_impulse_candle or 0.0) >= 1.0:
            return False
        return True

    # ------------------------------------------------------------------
    # Event factories
    # ------------------------------------------------------------------

    def _freeze_snapshot(self, snapshot: IndicatorSnapshot) -> tuple[dict, str]:
        snap_dict = snapshot.model_dump()
        # Convert non-serializable types to primitives
        for k, v in snap_dict.items():
            if isinstance(v, datetime):
                snap_dict[k] = v.isoformat()
        snap_hash = _hash_snapshot(snap_dict)
        return snap_dict, snap_hash

    def _emit_compression_candidate(
        self,
        symbol: str,
        timeframe: str,
        bar_ts: datetime,
        snapshot: IndicatorSnapshot,
        state: _SymbolState,
    ) -> SetupEvent:
        import uuid
        chain_id = str(uuid.uuid4())
        state.state = "compression_candidate"
        state.setup_chain_id = chain_id
        state.compression_range_high = snapshot.donchian_upper_short
        state.compression_range_low = snapshot.donchian_lower_short
        state.compression_atr = snapshot.atr_14
        state.compression_ts = bar_ts
        state.compression_bars_elapsed = 0

        snap_dict, snap_hash = self._freeze_snapshot(snapshot)
        score = self._scorer.score(snap_dict)

        return SetupEvent(
            setup_chain_id=chain_id,
            state="compression_candidate",
            engine_semver=self._engine_semver,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            strategy_template_version=self._template_version,
            ts=bar_ts,
            symbol=symbol,
            timeframe=timeframe,
            session=_session_context_for(bar_ts, symbol),
            feature_snapshot=snap_dict,
            feature_snapshot_hash=snap_hash,
            compression_range_high=state.compression_range_high,
            compression_range_low=state.compression_range_low,
            compression_atr_at_detection=state.compression_atr,
            model_quality_score=score.model_quality_score,
            p_cont_1R=score.p_cont_1R,
            p_false_breakout=score.p_false_breakout,
            p_atr_expand=score.p_atr_expand,
            model_version=score.model_version,
            ttl_bars=self._outcome_ttl,
        )

    def _emit_break_attempt(
        self,
        symbol: str,
        timeframe: str,
        bar_ts: datetime,
        snapshot: IndicatorSnapshot,
        state: _SymbolState,
    ) -> SetupEvent:
        state.state = "break_attempt"
        state.break_ts = bar_ts
        state.break_bars_elapsed = 0

        snap_dict, snap_hash = self._freeze_snapshot(snapshot)
        score = self._scorer.score(snap_dict)

        return SetupEvent(
            setup_chain_id=state.setup_chain_id,
            state="break_attempt",
            engine_semver=self._engine_semver,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            strategy_template_version=self._template_version,
            ts=bar_ts,
            symbol=symbol,
            timeframe=timeframe,
            session=_session_context_for(bar_ts, symbol),
            feature_snapshot=snap_dict,
            feature_snapshot_hash=snap_hash,
            compression_range_high=state.compression_range_high,
            compression_range_low=state.compression_range_low,
            compression_atr_at_detection=state.compression_atr,
            model_quality_score=score.model_quality_score,
            p_cont_1R=score.p_cont_1R,
            p_false_breakout=score.p_false_breakout,
            p_atr_expand=score.p_atr_expand,
            model_version=score.model_version,
            ttl_bars=self._outcome_ttl,
        )
```

### Step 5: Wire into `backtesting/llm_strategist_runner.py`

**Initialization** (in `__init__`):
```python
from backtesting.constants import ENGINE_SEMVER
from agents.analytics.setup_event_generator import SetupEventGenerator
from services.model_scorer import NullModelScorer

# After self.window_configs = ...
self._model_scorer = getattr(self, "_model_scorer", NullModelScorer())
self._setup_gen = SetupEventGenerator(
    engine_semver=ENGINE_SEMVER,
    scorer=self._model_scorer,
    strategy_template_version=None,  # set by caller if using a named template
)
self._setup_events: list = []  # collected for export / logging
```

**Per-bar setup event generation** (in the main loop, after `_indicator_snapshot`):
```python
snapshot = self._indicator_snapshot(pair, base_tf, timestamp)
if snapshot:
    new_events = self._setup_gen.on_bar(pair, base_tf, timestamp, snapshot)
    self._setup_events.extend(new_events)
```

**Model gating at entry** (in the entry evaluation path, before `_risk_budget_gate`):
```python
def _model_gate(self, symbol: str, snapshot: IndicatorSnapshot | None) -> tuple[bool, float]:
    """Return (is_blocked, size_multiplier) from model scorer.

    Returns (False, 1.0) when no model is trained (NullModelScorer).
    """
    if snapshot is None:
        return False, 1.0
    score = self._model_scorer.score(snapshot.model_dump())
    blocked = self._model_scorer.is_entry_blocked(score)
    mult = self._model_scorer.size_multiplier(score)
    if blocked:
        logger.info(
            "model gate BLOCKED entry for %s: p_false_breakout=%.3f > 0.40",
            symbol, score.p_false_breakout or 0.0,
        )
    return blocked, mult
```

Call before sizing:
```python
model_blocked, size_mult = self._model_gate(symbol, snapshot)
if model_blocked:
    return None  # entry blocked by model gate

# Apply size_multiplier to target_risk_pct before position sizing
effective_risk_pct = target_risk_pct * size_mult
```

**Expose setup events in backtest result**:
```python
# In StrategistBacktestResult or the run() return value:
result.setup_events = self._setup_events
```

### Step 6: Extend `SignalEvent` (Runbook 43 schema update)

Add three new optional fields to `SignalEvent` in `schemas/signal_event.py`:
```python
setup_event_id: Optional[str] = Field(
    default=None,
    description=(
        "ID of the SetupEvent (break_attempt state) that preceded this signal. "
        "Null if the trigger fired outside a tracked setup lifecycle."
    ),
)
feature_schema_version: str = Field(
    default="1.2.0",
    description="IndicatorSnapshot schema version at emission time.",
)
strategy_template_version: Optional[str] = Field(
    default=None,
    description="Active strategy template name (e.g., 'compression_breakout_v1').",
)
model_score: Optional[dict] = Field(
    default=None,
    description=(
        "Serialized ModelScorePacket at signal emission time. "
        "Null if NullModelScorer active."
    ),
)
```

### Step 7: DB migration — `setup_event_ledger` table

```sql
CREATE TABLE setup_event_ledger (
    setup_event_id       TEXT PRIMARY KEY,
    setup_chain_id       TEXT NOT NULL,
    state                TEXT NOT NULL,  -- compression_candidate | break_attempt | confirmed | failed | ttl_expired
    ts                   TIMESTAMPTZ NOT NULL,
    symbol               TEXT NOT NULL,
    timeframe            TEXT NOT NULL,
    engine_semver        TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    strategy_template_version TEXT,
    feature_snapshot     JSONB NOT NULL,  -- full IndicatorSnapshot dict
    feature_snapshot_hash TEXT NOT NULL,
    compression_range_high  NUMERIC(24,12),
    compression_range_low   NUMERIC(24,12),
    compression_atr_at_detection NUMERIC(24,12),
    session_type         TEXT NOT NULL,
    time_in_session_sin  REAL NOT NULL,
    time_in_session_cos  REAL NOT NULL,
    is_weekend           BOOLEAN NOT NULL,
    asset_class          TEXT NOT NULL DEFAULT 'crypto',
    -- Model scores (null until model is trained)
    model_quality_score  REAL,
    p_cont_1r            REAL,
    p_false_breakout     REAL,
    p_atr_expand         REAL,
    model_version        TEXT,
    -- Outcome (null until reconciled by SetupOutcomeReconciler)
    outcome              TEXT,           -- hit_1r | hit_stop | ttl_expired
    outcome_ts           TIMESTAMPTZ,
    bars_to_outcome      INTEGER,
    mfe_pct              REAL,
    mae_pct              REAL,
    r_achieved           REAL,
    ttl_bars             INTEGER NOT NULL DEFAULT 48,
    -- Links
    signal_event_id      TEXT            -- FK to signal_ledger.signal_id
);

CREATE INDEX idx_sel_symbol_ts  ON setup_event_ledger (symbol, ts);
CREATE INDEX idx_sel_chain      ON setup_event_ledger (setup_chain_id);
CREATE INDEX idx_sel_state      ON setup_event_ledger (state);
CREATE INDEX idx_sel_outcome    ON setup_event_ledger (outcome) WHERE outcome IS NULL;
CREATE INDEX idx_sel_fschema    ON setup_event_ledger (feature_schema_version);
```

> **Important**: Index on `feature_schema_version` is required. Training scripts MUST filter to a single version before fitting. Pooling across versions is invalid because feature semantics change.

### Step 8: `SetupOutcomeReconciler` — `services/setup_outcome_reconciler.py`

The reconciler scans `break_attempt` rows older than `ttl_bars` and labels them using subsequent OHLCV.

```python
"""
SetupOutcomeReconciler — labels open SetupEvent rows after TTL.

Outcome rules (evaluated in order, first match wins):

1. hit_1r: Within ttl_bars after break_attempt ts, price reaches entry + 1R
   (where 1R = abs(entry - stop), using compression_atr_at_detection as proxy
   if stop is not available). Favorable direction: same as break direction
   (up if close > compression_range_high, down otherwise).

2. hit_stop: Within ttl_bars, price reaches stop level
   (compression_range_low * 0.999 for upside breaks,
    compression_range_high * 1.001 for downside breaks).

3. ttl_expired: Neither 1 nor 2 occurred within ttl_bars.

Labels are immutable once written (no updates after first resolution).

MFE/MAE are computed as the best/worst price seen from break_ts to outcome_ts,
expressed as % of the entry price.
"""
```

Key methods:
```python
class SetupOutcomeReconciler:
    def reconcile_open_events(
        self,
        cutoff_ts: datetime,
        ohlcv_provider: Callable[[str, datetime, datetime, str], pd.DataFrame],
    ) -> int:
        """Reconcile all open break_attempt rows older than cutoff_ts.
        Returns count of rows labeled."""

    def _determine_outcome(
        self,
        event: SetupEvent,
        ohlcv: pd.DataFrame,
    ) -> tuple[str, float | None, float | None, float | None, int]:
        """Returns (outcome, mfe_pct, mae_pct, r_achieved, bars_to_outcome)."""
```

### Step 9: Parquet export CLI — `app/cli/main.py`

Add command group:
```
uv run python -m app.cli.main setup-events export-parquet \
  --output data/training/setup_events.parquet \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --feature-schema-version 1.2.0 \
  --state break_attempt \
  --require-outcome
```

Options:
- `--feature-schema-version`: filter to specific schema version (required for training)
- `--state`: which state to export (default: `break_attempt`)
- `--require-outcome`: only rows with non-null outcome (default: True for training)
- `--output`: parquet path

### Step 10: Tests — `tests/test_setup_event_generator.py`

Key test cases:
```python
def test_no_event_on_normal_bar():
    """Normal bar (large body, not inside bar) → no SetupEvent."""

def test_compression_candidate_detected():
    """Inside bar + low body + near daily mid → CompressionCandidate emitted."""

def test_break_attempt_emitted_after_compression():
    """Compression detected, then close breaks above range high with impulse → BreakAttempt."""

def test_no_break_without_impulse():
    """Close breaks range but no impulse candle AND no vol burst → stays in compression_candidate."""

def test_compression_ttl_resets_to_idle():
    """After compression_ttl_bars without break, state returns to idle."""

def test_feature_snapshot_is_immutable_hash():
    """feature_snapshot_hash matches SHA-256 of the dict."""

def test_null_model_scorer_does_not_block():
    """NullModelScorer → is_entry_blocked returns False, size_multiplier returns 1.0."""

def test_model_gate_blocks_on_high_p_false_breakout():
    """Score with p_false_breakout=0.55 → is_entry_blocked returns True."""

def test_model_size_multiplier_clamp():
    """quality=0.0 → mult=0.5, quality=1.0 → mult=1.25, quality=0.5 → mult=1.0."""

def test_session_context_crypto_us():
    """Bar at 15:00 UTC → session_type = 'crypto_us', is_weekend = False."""

def test_session_context_weekend():
    """Bar on Saturday → is_weekend = True."""
```

---

## Test Plan
```bash
# Unit: state machine + model gating
uv run pytest tests/test_setup_event_generator.py -vv

# Unit: ModelScorePacket + ModelScorer interface
uv run pytest tests/test_setup_event_generator.py -k "model" -vv

# Schema: SetupEvent serializes without Pydantic errors
uv run pytest tests/test_setup_event_generator.py -k "schema" -vv

# Integration: ensure existing tests not broken by runner changes
uv run pytest tests/risk/ tests/test_llm_strategist_runner.py -x --tb=short -q

# Full suite (baseline: 649 passing)
uv run pytest -q
```

## Acceptance Criteria

- [ ] `FEATURE_SCHEMA_VERSION` constant defined; updated to `"1.2.0"` (post R38+R41)
- [ ] `ENGINE_SEMVER` defined; stored in every `SetupEvent` row
- [ ] `SetupEventGenerator.on_bar()` emits `SetupEvent(state="compression_candidate")` on inside bar + low conviction + near daily mid
- [ ] `SetupEventGenerator.on_bar()` emits `SetupEvent(state="break_attempt")` when close breaks locked range with impulse/volume confirmation
- [ ] Compression TTL resets state to `idle` after `compression_ttl_bars` without break
- [ ] `feature_snapshot` field is immutable after creation; `feature_snapshot_hash` is verified by tests
- [ ] `NullModelScorer` returns `ModelScorePacket()` (all None); `size_multiplier` = 1.0; `is_entry_blocked` = False
- [ ] `ModelScorer.is_entry_blocked()` returns True when `p_false_breakout > 0.40`
- [ ] `ModelScorer.size_multiplier()` returns value in `[0.5, 1.25]` for any input
- [ ] Runner applies model gate before position sizing; logs entry blocks at INFO
- [ ] `setup_event_ledger` DB migration creates table with correct schema + all four indexes
- [ ] `SetupOutcomeReconciler` labels open rows with `hit_1r`, `hit_stop`, or `ttl_expired`
- [ ] Parquet export CLI command filters by `feature_schema_version` and emits one row per event
- [ ] `SignalEvent` updated with `setup_event_id`, `feature_schema_version`, `strategy_template_version`
- [ ] No existing tests broken

## Human Verification Evidence

```
TODO: After implementation:
1. Run a 30-day backtest on BTC-USD + ETH-USD 1h.
2. Inspect setup_events list on StrategistBacktestResult.
3. Verify: compression_candidate rows have is_inside_bar == 1 in feature_snapshot.
4. Verify: break_attempt rows have close > compression_range_high or < compression_range_low.
5. Verify: feature_snapshot_hash matches hashlib.sha256(json.dumps(snapshot, sort_keys=True)).hexdigest().
6. Export setup events to parquet. Load in pandas. Confirm no NaN in feature columns.
7. Fit LightGBMClassifier on hit_1r label. Confirm AUC > 0.52 on held-out data
   (better than random; more is not expected until ≥5000 labeled events).
```

## Change Log
| Date | Change | Author |
|---|---|---|
| 2026-02-18 | Runbook created from ML architecture design session | Claude |

## Worktree Setup
```bash
git fetch
git worktree add -b feat/setup-event-generator ../wt-setup-event-gen main

# Depends on:
# - Runbook 42 (level-anchored stops) merged — provides stop_price_abs/target_price_abs for label integrity
# - Runbook 43 (signal ledger) implemented — SignalEvent to extend with setup_event_id

# When finished (after merge)
git worktree remove ../wt-setup-event-gen
```

## Git Workflow
```bash
git checkout main
git pull
git checkout -b feat/setup-event-generator

# Implement Steps 1–11

git add \
  schemas/feature_version.py \
  schemas/setup_event.py \
  schemas/model_score.py \
  services/model_scorer.py \
  agents/analytics/setup_event_generator.py \
  agents/analytics/__init__.py \
  backtesting/llm_strategist_runner.py \
  backtesting/constants.py \
  schemas/signal_event.py \
  app/db/migrations/versions/XXXX_add_setup_event_ledger.py \
  services/setup_outcome_reconciler.py \
  app/cli/main.py \
  tests/test_setup_event_generator.py

uv run pytest tests/test_setup_event_generator.py -vv
git commit -m "Add Setup Event Generator, ModelScorer interface, and setup_event_ledger (Runbook 44)"
```

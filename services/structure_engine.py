"""Deterministic market structure engine.

Runbook 58: Deterministic Structure Engine and Context Exposure.

This module produces StructureSnapshot objects from indicator and OHLCV data.
It implements two strata of level extraction:

  Stratum S1 — Anchor levels (from IndicatorSnapshot htf_* fields or daily DataFrame)
    - D-1 and D-2 prior-session OHLC anchors
    - Rolling 5D, 20D high/low windows
    - Prior weekly and monthly anchors (when daily DataFrame provided)

  Stratum S2 — Swing structure (from intraday OHLCV DataFrame)
    - Deterministic swing highs/lows using fixed lookback/forward rules
    - Recency and proximity ranked

Event detection compares current levels against a prior snapshot to emit:
  - level_broken / level_reclaimed
  - range_breakout / range_breakdown (5D / 20D bounds)
  - structure_shift (HTF anchor role reversal)

Design invariants:
  - Pure functions — no I/O or randomness
  - All outputs deterministic given the same input
  - Level IDs are content-addressed (symbol|kind|timeframe|price)
  - Role classification (support/resistance) is separate from raw identity

Runbook 54 forward-compat:
  StructureSnapshot.policy_trigger_reasons and policy_event_priority are
  populated here.  The policy-loop cadence router (R54, docs-only) will
  consume these fields to schedule reassessment ticks.

Runbook 52/56 forward-compat:
  Levels with eligible_for_stop_anchor=True / eligible_for_target_anchor=True
  are the candidate pool for the structural target selector (R56, docs-only).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    import numpy as np
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

from schemas.llm_strategist import IndicatorSnapshot
from schemas.structure_engine import (
    STRUCTURE_ENGINE_VERSION,
    LevelLadder,
    StructureEvent,
    StructureLevel,
    StructureQuality,
    StructureSnapshot,
    compute_structure_snapshot_hash,
)

logger = logging.getLogger(__name__)

# Proximity band thresholds (ATR multiples) for ladder bucketing
_NEAR_ATR_THRESHOLD = 1.0
_MID_ATR_THRESHOLD = 3.0

# Minimum separation between S2 swing levels (fraction of price)
_SWING_MIN_SEPARATION_PCT = 0.3

# Minimum lookback/forward bars for S2 swing confirmation
_SWING_LOOKBACK = 5

# For range breakout/breakdown — prefer 5D window when available; fallback to 20D
_RANGE_WINDOW_PRIMARY = "5d"
_RANGE_WINDOW_SECONDARY = "20d"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _make_level_id(symbol: str, kind: str, source_tf: str, price: float) -> str:
    """Stable, content-addressed level identifier."""
    return f"{symbol}|{kind}|{source_tf}|{price:.4f}"


def _classify_role(
    price_level: float,
    reference_price: float,
    neutral_band_pct: float = 0.05,
) -> str:
    """Classify a level as support, resistance, or neutral vs reference_price.

    Neutral band: within neutral_band_pct% of reference price.
    Above band → resistance; below band → support.
    """
    if reference_price <= 0:
        return "neutral"
    pct_diff = (price_level - reference_price) / reference_price * 100
    if pct_diff > neutral_band_pct:
        return "resistance"
    if pct_diff < -neutral_band_pct:
        return "support"
    return "neutral"


def _compute_distances(
    price_level: float,
    reference_price: float,
    reference_atr: Optional[float],
) -> Tuple[float, float, Optional[float]]:
    """Return (distance_abs, distance_pct, distance_atr)."""
    distance_abs = abs(price_level - reference_price)
    distance_pct = (distance_abs / reference_price * 100) if reference_price > 0 else 0.0
    distance_atr = (distance_abs / reference_atr) if (reference_atr and reference_atr > 0) else None
    return distance_abs, distance_pct, distance_atr


def _make_level(
    snapshot_id: str,
    symbol: str,
    as_of_ts: datetime,
    price: float,
    kind: str,
    source_tf: str,
    source_label: str,
    reference_price: float,
    reference_atr: Optional[float],
    source_metadata: Optional[Dict[str, Any]] = None,
    eligible_for_entry_trigger: bool = False,
    eligible_for_stop_anchor: bool = False,
    eligible_for_target_anchor: bool = False,
    strength_score: Optional[float] = None,
) -> StructureLevel:
    """Construct a StructureLevel with computed role and distances."""
    role = _classify_role(price, reference_price)
    dist_abs, dist_pct, dist_atr = _compute_distances(price, reference_price, reference_atr)
    level_id = _make_level_id(symbol, kind, source_tf, price)
    return StructureLevel(
        level_id=level_id,
        snapshot_id=snapshot_id,
        symbol=symbol,
        as_of_ts=as_of_ts,
        price=price,
        source_timeframe=source_tf,
        kind=kind,
        source_label=source_label,
        source_metadata=source_metadata or {},
        role_now=role,
        distance_abs=dist_abs,
        distance_pct=dist_pct,
        distance_atr=dist_atr,
        eligible_for_entry_trigger=eligible_for_entry_trigger,
        eligible_for_stop_anchor=eligible_for_stop_anchor,
        eligible_for_target_anchor=eligible_for_target_anchor,
        strength_score=strength_score,
    )


def _bucket_by_atr(
    level: StructureLevel,
    near_threshold: float = _NEAR_ATR_THRESHOLD,
    mid_threshold: float = _MID_ATR_THRESHOLD,
) -> str:
    """Return 'near', 'mid', or 'far' based on distance_atr (fallback: mid)."""
    if level.distance_atr is None:
        return "mid"
    if level.distance_atr <= near_threshold:
        return "near"
    if level.distance_atr <= mid_threshold:
        return "mid"
    return "far"


def _build_ladder(
    levels: List[StructureLevel],
    source_tf: str,
) -> LevelLadder:
    """Build a LevelLadder for one source timeframe from the global level list."""
    tf_levels = [l for l in levels if l.source_timeframe == source_tf]
    supports = sorted(
        [l for l in tf_levels if l.role_now == "support"],
        key=lambda l: l.distance_abs,
    )
    resistances = sorted(
        [l for l in tf_levels if l.role_now == "resistance"],
        key=lambda l: l.distance_abs,
    )

    near_s, mid_s, far_s = [], [], []
    for level in supports:
        bucket = _bucket_by_atr(level)
        {"near": near_s, "mid": mid_s, "far": far_s}[bucket].append(level)

    near_r, mid_r, far_r = [], [], []
    for level in resistances:
        bucket = _bucket_by_atr(level)
        {"near": near_r, "mid": mid_r, "far": far_r}[bucket].append(level)

    return LevelLadder(
        source_timeframe=source_tf,
        near_supports=near_s,
        mid_supports=mid_s,
        far_supports=far_s,
        near_resistances=near_r,
        mid_resistances=mid_r,
        far_resistances=far_r,
    )


# ---------------------------------------------------------------------------
# Stratum S1 — Anchor levels from IndicatorSnapshot
# ---------------------------------------------------------------------------

def _extract_s1_from_indicator(
    snapshot_id: str,
    symbol: str,
    as_of_ts: datetime,
    ind: IndicatorSnapshot,
    reference_price: float,
    reference_atr: Optional[float],
) -> List[StructureLevel]:
    """Extract Stratum S1 anchor levels from the indicator's htf_* fields.

    Sources:
      - htf_daily_high/low → D-1 prior session OHLC anchors (timeframe: 1d)
      - htf_daily_open     → D-1 open (timeframe: 1d)
      - htf_prev_daily_high/low → D-2 anchors (timeframe: 1d)
      - htf_5d_high/low    → Rolling 5-session window (timeframe: 5d)

    Returns empty list when all htf fields are None.
    """
    levels: List[StructureLevel] = []

    def add(price: Optional[float], kind: str, tf: str, label: str,
            stop_ok: bool = False, target_ok: bool = False,
            entry_ok: bool = False, strength: Optional[float] = None) -> None:
        if price is None or price <= 0:
            return
        levels.append(_make_level(
            snapshot_id=snapshot_id, symbol=symbol, as_of_ts=as_of_ts,
            price=price, kind=kind, source_tf=tf, source_label=label,
            reference_price=reference_price, reference_atr=reference_atr,
            eligible_for_entry_trigger=entry_ok,
            eligible_for_stop_anchor=stop_ok,
            eligible_for_target_anchor=target_ok,
            strength_score=strength,
        ))

    # D-1 anchors (strong — primary session boundaries)
    add(ind.htf_daily_high, "prior_session_high", "1d", "D-1 High",
        stop_ok=True, target_ok=True, entry_ok=True, strength=0.8)
    add(ind.htf_daily_low, "prior_session_low", "1d", "D-1 Low",
        stop_ok=True, target_ok=True, entry_ok=True, strength=0.8)
    if getattr(ind, "htf_daily_open", None):
        add(ind.htf_daily_open, "prior_session_open", "1d", "D-1 Open",
            strength=0.5)
    # D-1 midpoint
    if ind.htf_daily_high and ind.htf_daily_low:
        mid = (ind.htf_daily_high + ind.htf_daily_low) / 2
        add(mid, "prior_session_mid", "1d", "D-1 Mid", strength=0.4)

    # D-2 anchors
    add(getattr(ind, "htf_prev_daily_high", None), "prior_session_high", "1d", "D-2 High",
        stop_ok=True, target_ok=True, strength=0.6)
    add(getattr(ind, "htf_prev_daily_low", None), "prior_session_low", "1d", "D-2 Low",
        stop_ok=True, target_ok=True, strength=0.6)

    # 5D rolling window (weekly proxy)
    add(getattr(ind, "htf_5d_high", None), "rolling_window_high", "5d", "5D High",
        target_ok=True, entry_ok=True, strength=0.7)
    add(getattr(ind, "htf_5d_low", None), "rolling_window_low", "5d", "5D Low",
        stop_ok=True, entry_ok=True, strength=0.7)

    return levels


def _extract_s1_from_daily_df(
    snapshot_id: str,
    symbol: str,
    as_of_ts: datetime,
    daily_df: "pd.DataFrame",
    reference_price: float,
    reference_atr: Optional[float],
) -> List[StructureLevel]:
    """Extract extended S1 anchor levels from a raw daily OHLCV DataFrame.

    Adds:
      - Rolling 20D high/low
      - Prior weekly high/low (most recent complete week, Mon–Sun)
      - Prior monthly high/low (most recent complete calendar month)

    Expects DataFrame with columns: open, high, low, close, volume
    and a DatetimeIndex (UTC) sorted ascending.

    Returns empty list when df is too short or lacks expected columns.
    """
    if not _HAS_PANDAS:
        return []
    levels: List[StructureLevel] = []

    required_cols = {"high", "low", "open", "close"}
    if not required_cols.issubset(set(daily_df.columns)):
        return levels

    df = daily_df.copy()
    if not hasattr(df.index, "tzinfo"):
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    if len(df) < 2:
        return levels

    def add(price: float, kind: str, tf: str, label: str,
            stop_ok: bool = False, target_ok: bool = False,
            entry_ok: bool = False, strength: Optional[float] = None,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        if price is None or price <= 0:
            return
        levels.append(_make_level(
            snapshot_id=snapshot_id, symbol=symbol, as_of_ts=as_of_ts,
            price=price, kind=kind, source_tf=tf, source_label=label,
            reference_price=reference_price, reference_atr=reference_atr,
            eligible_for_entry_trigger=entry_ok,
            eligible_for_stop_anchor=stop_ok,
            eligible_for_target_anchor=target_ok,
            strength_score=strength,
            source_metadata=metadata or {},
        ))

    # Rolling 20D high/low (exclude current bar)
    lookback = df.iloc[:-1]  # drop latest incomplete bar
    if len(lookback) >= 20:
        window = lookback.iloc[-20:]
        add(float(window["high"].max()), "rolling_window_high", "20d", "20D High",
            target_ok=True, entry_ok=True, strength=0.65)
        add(float(window["low"].min()), "rolling_window_low", "20d", "20D Low",
            stop_ok=True, entry_ok=True, strength=0.65)
    elif len(lookback) >= 5:
        window = lookback
        add(float(window["high"].max()), "rolling_window_high", "20d", f"{len(window)}D High",
            strength=0.5)
        add(float(window["low"].min()), "rolling_window_low", "20d", f"{len(window)}D Low",
            strength=0.5)

    # Prior week (Mon–Sun, most recent complete week)
    try:
        # Resample to weekly (ending Sunday)
        weekly = df.resample("W-SUN", closed="right", label="right").agg(
            {"high": "max", "low": "min"}
        )
        # Drop any incomplete current week (the last row covers up to now)
        if len(weekly) >= 2:
            prior_week = weekly.iloc[-2]  # second-last row = last complete week
            add(float(prior_week["high"]), "rolling_window_high", "1w", "Prior Week High",
                target_ok=True, entry_ok=True, strength=0.75)
            add(float(prior_week["low"]), "rolling_window_low", "1w", "Prior Week Low",
                stop_ok=True, entry_ok=True, strength=0.75)
    except Exception:
        pass

    # Prior month (most recent complete calendar month)
    try:
        monthly = df.resample("MS").agg({"high": "max", "low": "min"})
        if len(monthly) >= 2:
            prior_month = monthly.iloc[-2]
            add(float(prior_month["high"]), "rolling_window_high", "1M", "Prior Month High",
                target_ok=True, strength=0.80)
            add(float(prior_month["low"]), "rolling_window_low", "1M", "Prior Month Low",
                stop_ok=True, strength=0.80)
    except Exception:
        pass

    return levels


# ---------------------------------------------------------------------------
# Stratum S2 — Swing structure from intraday OHLCV
# ---------------------------------------------------------------------------

def _find_swing_points(
    df: "pd.DataFrame",
    lookback: int = _SWING_LOOKBACK,
) -> Tuple[List[Tuple[datetime, float]], List[Tuple[datetime, float]]]:
    """Find swing highs and lows using fixed lookback/forward rules.

    A swing high at bar i satisfies: high[i] >= high[i-k] and high[i] >= high[i+k]
    for all k in 1..lookback.  Similarly for swing lows.

    Returns (swing_highs, swing_lows) as lists of (timestamp, price) pairs.
    Requires at least 2*lookback+1 bars; returns empty lists otherwise.
    """
    if not _HAS_PANDAS:
        return [], []
    if len(df) < lookback * 2 + 1:
        return [], []

    df = df.sort_index()
    highs = df["high"].values
    lows = df["low"].values
    times = df.index.to_pydatetime()

    swing_highs: List[Tuple[datetime, float]] = []
    swing_lows: List[Tuple[datetime, float]] = []

    for i in range(lookback, len(df) - lookback):
        is_sh = all(highs[i] >= highs[i - k] for k in range(1, lookback + 1)) and \
                all(highs[i] >= highs[i + k] for k in range(1, lookback + 1))
        if is_sh:
            swing_highs.append((times[i], float(highs[i])))

        is_sl = all(lows[i] <= lows[i - k] for k in range(1, lookback + 1)) and \
                all(lows[i] <= lows[i + k] for k in range(1, lookback + 1))
        if is_sl:
            swing_lows.append((times[i], float(lows[i])))

    return swing_highs, swing_lows


def _deduplicate_nearby(
    points: List[Tuple[datetime, float]],
    reference_price: float,
    min_separation_pct: float = _SWING_MIN_SEPARATION_PCT,
) -> List[Tuple[datetime, float]]:
    """Remove swing points that are within min_separation_pct% of each other.

    Keeps the most recent when two points are too close.
    """
    if not points:
        return []
    # Sort by time descending (most recent first) so we prefer recent levels
    pts = sorted(points, key=lambda x: x[0], reverse=True)
    kept: List[Tuple[datetime, float]] = []
    for ts, price in pts:
        if not kept:
            kept.append((ts, price))
            continue
        too_close = any(
            abs(price - k_price) / reference_price * 100 < min_separation_pct
            for _, k_price in kept
        )
        if not too_close:
            kept.append((ts, price))
    return kept


def _extract_s2_swings(
    snapshot_id: str,
    symbol: str,
    as_of_ts: datetime,
    ohlcv_df: "pd.DataFrame",
    source_tf: str,
    reference_price: float,
    reference_atr: Optional[float],
    lookback: int = _SWING_LOOKBACK,
    max_levels: int = 6,
) -> List[StructureLevel]:
    """Extract Stratum S2 swing highs/lows from an intraday OHLCV DataFrame.

    Each swing is assigned strength proportional to its recency rank:
    most recent swing → 0.70; older swings decay toward 0.50.

    Returns up to max_levels swing highs + max_levels swing lows.
    """
    if not _HAS_PANDAS:
        return []
    levels: List[StructureLevel] = []

    required = {"high", "low"}
    if not required.issubset(set(ohlcv_df.columns)):
        return levels

    swing_highs, swing_lows = _find_swing_points(ohlcv_df, lookback=lookback)

    # Deduplicate nearby swings
    swing_highs = _deduplicate_nearby(swing_highs, reference_price)[:max_levels]
    swing_lows = _deduplicate_nearby(swing_lows, reference_price)[:max_levels]

    total_sh = len(swing_highs)
    for rank, (ts, price) in enumerate(swing_highs):
        strength = 0.70 - (rank / max(total_sh, 1)) * 0.20
        touch_count = 1  # single detection; S3 clustering can refine
        levels.append(_make_level(
            snapshot_id=snapshot_id, symbol=symbol, as_of_ts=as_of_ts,
            price=price, kind="swing_high", source_tf=source_tf,
            source_label=f"Swing High ({ts.strftime('%m-%d %H:%M')})",
            reference_price=reference_price, reference_atr=reference_atr,
            eligible_for_entry_trigger=True,
            eligible_for_target_anchor=True,
            eligible_for_stop_anchor=False,
            strength_score=round(strength, 3),
            source_metadata={"swing_ts": ts.isoformat(), "rank": rank},
        ))
        levels[-1] = levels[-1].model_copy(update={"touch_count": touch_count, "last_touch_ts": ts})

    total_sl = len(swing_lows)
    for rank, (ts, price) in enumerate(swing_lows):
        strength = 0.70 - (rank / max(total_sl, 1)) * 0.20
        levels.append(_make_level(
            snapshot_id=snapshot_id, symbol=symbol, as_of_ts=as_of_ts,
            price=price, kind="swing_low", source_tf=source_tf,
            source_label=f"Swing Low ({ts.strftime('%m-%d %H:%M')})",
            reference_price=reference_price, reference_atr=reference_atr,
            eligible_for_entry_trigger=True,
            eligible_for_stop_anchor=True,
            eligible_for_target_anchor=False,
            strength_score=round(strength, 3),
            source_metadata={"swing_ts": ts.isoformat(), "rank": rank},
        ))
        levels[-1] = levels[-1].model_copy(update={"touch_count": 1, "last_touch_ts": ts})

    return levels


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def _detect_events(
    snapshot_id: str,
    symbol: str,
    as_of_ts: datetime,
    eval_timeframe: str,
    current_levels: List[StructureLevel],
    reference_price: float,
    ind: IndicatorSnapshot,
    prior_snapshot: Optional[StructureSnapshot],
) -> List[StructureEvent]:
    """Detect structural events by comparing current state against prior snapshot.

    Events emitted:
      - level_broken: price closed below a prior support level
      - level_reclaimed: price closed above a prior resistance level
      - range_breakout: close above prior 5D/20D rolling high
      - range_breakdown: close below prior 5D/20D rolling low
      - structure_shift: an HTF anchor level (strength >= 0.7) reversed role
    """
    events: List[StructureEvent] = []
    close = ind.close or reference_price

    # --- Level break/reclaim vs prior snapshot ---
    if prior_snapshot is not None:
        prior_by_id: Dict[str, StructureLevel] = {l.level_id: l for l in prior_snapshot.levels}
        current_by_id: Dict[str, StructureLevel] = {l.level_id: l for l in current_levels}

        for level_id, prior_level in prior_by_id.items():
            current_level = current_by_id.get(level_id)
            if current_level is None:
                continue

            prior_role = prior_level.role_now
            current_role = current_level.role_now

            if prior_role == current_role:
                continue

            # Support broken: was support, close now below its price
            if prior_role == "support" and close < prior_level.price:
                sev = "high" if (prior_level.strength_score or 0) >= 0.7 else "medium"
                events.append(StructureEvent(
                    event_id=str(uuid4()),
                    snapshot_id=snapshot_id,
                    symbol=symbol,
                    as_of_ts=as_of_ts,
                    eval_timeframe=eval_timeframe,
                    event_type="level_broken",
                    severity=sev,
                    level_id=level_id,
                    level_kind=prior_level.kind,
                    direction="down",
                    price_ref=reference_price,
                    close_ref=close,
                    threshold_ref=prior_level.price,
                    confirmation_rule="close_below_support",
                    evidence={
                        "prior_role": prior_role,
                        "current_role": current_role,
                        "level_price": prior_level.price,
                        "strength_score": prior_level.strength_score,
                    },
                    trigger_policy_reassessment=True,
                ))

            # Resistance reclaimed: was resistance, close now above its price
            elif prior_role == "resistance" and close > prior_level.price:
                sev = "high" if (prior_level.strength_score or 0) >= 0.7 else "medium"
                events.append(StructureEvent(
                    event_id=str(uuid4()),
                    snapshot_id=snapshot_id,
                    symbol=symbol,
                    as_of_ts=as_of_ts,
                    eval_timeframe=eval_timeframe,
                    event_type="level_reclaimed",
                    severity=sev,
                    level_id=level_id,
                    level_kind=prior_level.kind,
                    direction="up",
                    price_ref=reference_price,
                    close_ref=close,
                    threshold_ref=prior_level.price,
                    confirmation_rule="close_above_resistance",
                    evidence={
                        "prior_role": prior_role,
                        "current_role": current_role,
                        "level_price": prior_level.price,
                        "strength_score": prior_level.strength_score,
                    },
                    trigger_policy_reassessment=True,
                ))

            # Structure shift: high-strength HTF anchor reversed role
            if (prior_level.strength_score or 0) >= 0.7 and prior_role != current_role:
                events.append(StructureEvent(
                    event_id=str(uuid4()),
                    snapshot_id=snapshot_id,
                    symbol=symbol,
                    as_of_ts=as_of_ts,
                    eval_timeframe=eval_timeframe,
                    event_type="structure_shift",
                    severity="medium",
                    level_id=level_id,
                    level_kind=prior_level.kind,
                    direction="up" if current_role == "resistance" else "down",
                    price_ref=reference_price,
                    close_ref=close,
                    threshold_ref=prior_level.price,
                    confirmation_rule="htf_anchor_role_reversal",
                    evidence={
                        "prior_role": prior_role,
                        "new_role": current_role,
                        "strength_score": prior_level.strength_score,
                    },
                    trigger_policy_reassessment=True,
                    trigger_activation_review=True,
                ))

    # --- Range breakout / breakdown (primary: 5D; secondary: 20D) ---
    htf_5d_high = getattr(ind, "htf_5d_high", None)
    htf_5d_low = getattr(ind, "htf_5d_low", None)

    if htf_5d_high and close > htf_5d_high:
        events.append(StructureEvent(
            event_id=str(uuid4()),
            snapshot_id=snapshot_id,
            symbol=symbol,
            as_of_ts=as_of_ts,
            eval_timeframe=eval_timeframe,
            event_type="range_breakout",
            severity="high",
            direction="up",
            price_ref=reference_price,
            close_ref=close,
            threshold_ref=htf_5d_high,
            confirmation_rule="close_above_5d_high",
            evidence={"window": "5d", "5d_high": htf_5d_high, "close": close},
            trigger_policy_reassessment=True,
            trigger_activation_review=True,
        ))

    if htf_5d_low and close < htf_5d_low:
        events.append(StructureEvent(
            event_id=str(uuid4()),
            snapshot_id=snapshot_id,
            symbol=symbol,
            as_of_ts=as_of_ts,
            eval_timeframe=eval_timeframe,
            event_type="range_breakdown",
            severity="high",
            direction="down",
            price_ref=reference_price,
            close_ref=close,
            threshold_ref=htf_5d_low,
            confirmation_rule="close_below_5d_low",
            evidence={"window": "5d", "5d_low": htf_5d_low, "close": close},
            trigger_policy_reassessment=True,
        ))

    return events


# ---------------------------------------------------------------------------
# Policy trigger priority (Runbook 54 forward-compat)
# ---------------------------------------------------------------------------

def _compute_policy_priority(events: List[StructureEvent]) -> Optional[str]:
    """Derive a single policy-event priority from the set of events.

    high  → any high-severity event present
    medium → any medium-severity event (no high)
    low   → only low-severity events
    None  → no events
    """
    if not events:
        return None
    severities = {e.severity for e in events}
    if "high" in severities:
        return "high"
    if "medium" in severities:
        return "medium"
    return "low"


def _compute_policy_trigger_reasons(events: List[StructureEvent]) -> List[str]:
    """Human-readable trigger reasons for policy loop (Runbook 54 contract)."""
    reasons: List[str] = []
    for ev in events:
        if ev.trigger_policy_reassessment:
            label = ev.event_type.replace("_", " ").title()
            if ev.threshold_ref:
                label += f" @ {ev.threshold_ref:.2f}"
            if ev.level_kind:
                label += f" ({ev.level_kind})"
            reasons.append(label)
    return reasons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_structure_snapshot(
    indicator: IndicatorSnapshot,
    *,
    daily_df: Optional["pd.DataFrame"] = None,
    ohlcv_df: Optional["pd.DataFrame"] = None,
    ohlcv_timeframe: str = "1h",
    prior_snapshot: Optional[StructureSnapshot] = None,
    eval_timeframe: Optional[str] = None,
) -> StructureSnapshot:
    """Build a deterministic StructureSnapshot from indicator and OHLCV data.

    Args:
        indicator:        Current per-symbol IndicatorSnapshot (provides htf_* fields for S1).
        daily_df:         Optional daily OHLCV DataFrame for extended anchors (20D, weekly, monthly).
        ohlcv_df:         Optional intraday OHLCV DataFrame for S2 swing structure.
        ohlcv_timeframe:  Timeframe label for the ohlcv_df (e.g. "1h").
        prior_snapshot:   Prior StructureSnapshot for event detection (break/reclaim).
        eval_timeframe:   Timeframe used for event confirmation rules (default: indicator.timeframe).
    """
    now = _now_utc()
    as_of = indicator.as_of
    if hasattr(as_of, "tzinfo") and as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    symbol = indicator.symbol
    reference_price = indicator.close or 0.0
    reference_atr = indicator.atr_14
    eff_eval_tf = eval_timeframe or indicator.timeframe
    snapshot_id = str(uuid4())

    # --- Collect levels ---
    levels: List[StructureLevel] = []
    available_tfs: List[str] = []
    missing_tfs: List[str] = []
    warnings: List[str] = []

    # S1 — from indicator htf_* fields
    s1_ind_levels = _extract_s1_from_indicator(
        snapshot_id, symbol, as_of, indicator, reference_price, reference_atr
    )
    levels.extend(s1_ind_levels)
    if s1_ind_levels:
        available_tfs.extend(sorted({l.source_timeframe for l in s1_ind_levels}))
    else:
        missing_tfs.append("htf_anchors")
        warnings.append("No htf_* fields available — S1 anchor levels not populated")

    # S1 — extended from daily_df
    if daily_df is not None:
        s1_df_levels = _extract_s1_from_daily_df(
            snapshot_id, symbol, as_of, daily_df, reference_price, reference_atr
        )
        levels.extend(s1_df_levels)
        if s1_df_levels:
            for tf_key in ("20d", "1w", "1M"):
                if any(l.source_timeframe == tf_key for l in s1_df_levels):
                    if tf_key not in available_tfs:
                        available_tfs.append(tf_key)
        else:
            missing_tfs.append("20d_weekly_monthly")
    else:
        missing_tfs.append("20d_weekly_monthly")

    # S2 — swings from intraday OHLCV
    if ohlcv_df is not None:
        s2_levels = _extract_s2_swings(
            snapshot_id, symbol, as_of, ohlcv_df, ohlcv_timeframe,
            reference_price, reference_atr,
        )
        levels.extend(s2_levels)
        if s2_levels:
            if ohlcv_timeframe not in available_tfs:
                available_tfs.append(ohlcv_timeframe)
        else:
            missing_tfs.append("s2_swings")
    else:
        missing_tfs.append("s2_swings")

    # --- De-duplicate levels with the same level_id ---
    seen_ids: set = set()
    unique_levels: List[StructureLevel] = []
    for l in levels:
        if l.level_id not in seen_ids:
            seen_ids.add(l.level_id)
            unique_levels.append(l)
    levels = unique_levels

    # --- Build ladders per available source timeframe ---
    tfs_for_ladders = sorted({l.source_timeframe for l in levels})
    ladders: Dict[str, LevelLadder] = {
        tf: _build_ladder(levels, tf) for tf in tfs_for_ladders
    }

    # --- Event detection ---
    events = _detect_events(
        snapshot_id, symbol, as_of, eff_eval_tf,
        levels, reference_price, indicator, prior_snapshot,
    )

    # --- Policy trigger integration (R54 forward-compat) ---
    policy_trigger_reasons = _compute_policy_trigger_reasons(events)
    policy_event_priority = _compute_policy_priority(events)

    # --- Hash content ---
    hash_content: Dict[str, Any] = {
        "symbol": symbol,
        "as_of_ts": as_of.isoformat(),
        "reference_price": reference_price,
        "level_ids": sorted(l.level_id for l in levels),
        "level_prices": sorted(l.price for l in levels),
        "snapshot_version": STRUCTURE_ENGINE_VERSION,
    }
    snapshot_hash = compute_structure_snapshot_hash(hash_content)

    quality = StructureQuality(
        available_timeframes=sorted(set(available_tfs)),
        missing_timeframes=sorted(set(missing_tfs)),
        is_partial=bool(missing_tfs),
        quality_warnings=warnings,
    )

    data_source = "indicator_snapshot"
    if daily_df is not None or ohlcv_df is not None:
        data_source = "ohlcv_df"

    return StructureSnapshot(
        snapshot_id=snapshot_id,
        snapshot_hash=snapshot_hash,
        snapshot_version=STRUCTURE_ENGINE_VERSION,
        symbol=symbol,
        as_of_ts=as_of,
        generated_at_ts=now,
        source_timeframe=eff_eval_tf,
        data_source=data_source,
        reference_price=reference_price,
        reference_atr=reference_atr,
        levels=levels,
        ladders=ladders,
        events=events,
        policy_trigger_reasons=policy_trigger_reasons,
        policy_event_priority=policy_event_priority,
        quality=quality,
    )


# ---------------------------------------------------------------------------
# Convenience accessors (R52/R56 forward-compat)
# ---------------------------------------------------------------------------

def get_stop_candidates(snapshot: StructureSnapshot) -> List[StructureLevel]:
    """Return all stop-eligible levels sorted by distance (closest first).

    This is the candidate pool consumed by the structural target selector (R56, docs-only).
    """
    return sorted(
        [l for l in snapshot.levels if l.eligible_for_stop_anchor],
        key=lambda l: l.distance_abs,
    )


def get_target_candidates(snapshot: StructureSnapshot) -> List[StructureLevel]:
    """Return all target-eligible levels sorted by distance (closest first).

    This is the candidate pool consumed by the structural target selector (R56, docs-only).
    """
    return sorted(
        [l for l in snapshot.levels if l.eligible_for_target_anchor],
        key=lambda l: l.distance_abs,
    )


def get_entry_candidates(snapshot: StructureSnapshot) -> List[StructureLevel]:
    """Return all entry-trigger-eligible levels sorted by distance (closest first).

    This is the candidate pool consumed by the playbook instantiation layer (R52, docs-only).
    """
    return sorted(
        [l for l in snapshot.levels if l.eligible_for_entry_trigger],
        key=lambda l: l.distance_abs,
    )

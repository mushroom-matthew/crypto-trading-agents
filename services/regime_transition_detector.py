"""Deterministic regime fingerprint transition detector (Runbook 55).

Provides:
  - Vocabulary mapping from existing schema labels to R55 canonical enums.
  - build_regime_fingerprint() — construct from IndicatorSnapshot + AssetState.
  - regime_fingerprint_distance() — bounded [0,1] distance with decomposed contributions.
  - RegimeTransitionDetector — state machine with hysteresis, dwell, cooldown, shock override.

Vocabulary mapping version: 1.0
Distance formula version: 1.0 (locked to FINGERPRINT_VERSION)

HTF-close gating is managed by the caller — the detector's evaluate() accepts an
htf_gate_eligible flag and a shock_override flag is auto-detected from the fingerprint.
This keeps the state machine deterministic and replayable given the same inputs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from schemas.regime_fingerprint import (
    FINGERPRINT_SCHEMA_VERSION,
    FINGERPRINT_VERSION,
    NUMERIC_VECTOR_FEATURE_NAMES_V1,
    RegimeDistanceResult,
    RegimeFingerprint,
    RegimeTransitionDecision,
    RegimeTransitionDetectorState,
    RegimeTransitionTelemetryEvent,
)

if TYPE_CHECKING:
    from schemas.llm_strategist import AssetState, IndicatorSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Distance weights (version-locked with FINGERPRINT_VERSION "1.0.0")
# Weights sum to 1.0 → distance_value is naturally bounded [0, 1].
# ---------------------------------------------------------------------------
_DISTANCE_WEIGHTS: Dict[str, float] = {
    "trend_state": 0.25,
    "vol_state": 0.20,
    "structure_state": 0.25,
    "numeric_vector": 0.20,
    "confidence": 0.10,
}

# Default asymmetric thresholds
_DEFAULT_ENTER_THRESHOLD = 0.30  # require 30% of max distance to trigger
_DEFAULT_EXIT_THRESHOLD = 0.15   # lower bar — regime stays until clearly stable

# Shock override triggers
_SHOCK_VOL_PERCENTILE = 0.90     # extreme vol percentile
_SHOCK_REALIZED_VOL_Z = 2.5      # extreme vol z-score

# Normalization constants for numeric feature construction
_ATR_RATIO_MAX = 0.08            # 8% ATR/close → vol_percentile = 1.0
_VOL_Z_MEAN = 0.02               # typical ATR/close ratio (mean)
_VOL_Z_STD = 0.01                # std for z-score computation
_HTF_ANCHOR_MAX_ATR = 5.0        # clamp at 5× ATR from HTF anchor level

_VOCAB_MAPPING_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Vocabulary mapping
# ---------------------------------------------------------------------------

def map_trend_state(trend: str) -> Literal["up", "down", "sideways"]:
    """Map existing trend_state labels to R55 canonical vocabulary.

    Existing labels  →  R55 canonical
      'uptrend'      →  'up'
      'downtrend'    →  'down'
      'sideways'     →  'sideways'
      'bull'         →  'up'
      'bear'         →  'down'
      (unknown)      →  'sideways'

    Mapping version: 1.0
    """
    _map: Dict[str, Literal["up", "down", "sideways"]] = {
        "uptrend": "up",
        "up": "up",
        "bullish": "up",
        "bull": "up",
        "downtrend": "down",
        "down": "down",
        "bearish": "down",
        "bear": "down",
        "sideways": "sideways",
        "neutral": "sideways",
        "ranging": "sideways",
        "range": "sideways",
        "lateral": "sideways",
    }
    return _map.get((trend or "").lower(), "sideways")


def map_vol_state(vol: str) -> Literal["low", "mid", "high", "extreme"]:
    """Map existing vol_state labels to R55 canonical vocabulary.

    Existing labels  →  R55 canonical
      'low'          →  'low'
      'normal'       →  'mid'    ← key mapping difference
      'high'         →  'high'
      'extreme'      →  'extreme'
      (unknown)      →  'mid'

    Mapping version: 1.0
    """
    _map: Dict[str, Literal["low", "mid", "high", "extreme"]] = {
        "low": "low",
        "normal": "mid",
        "mid": "mid",
        "medium": "mid",
        "moderate": "mid",
        "high": "high",
        "extreme": "extreme",
        "very_high": "extreme",
    }
    return _map.get((vol or "").lower(), "mid")


def map_structure_state(
    compression_flag: Optional[float],
    expansion_flag: Optional[float],
    breakout_confirmed: Optional[float],
    trend_state: Literal["up", "down", "sideways"],
    regime: Optional[str] = None,
) -> Literal[
    "compression", "expansion", "mean_reverting",
    "breakout_active", "breakdown_active", "neutral",
]:
    """Derive structure_state from R40 indicator flags + trend + regime.

    Priority (highest to lowest):
    1. breakout_confirmed > 0.5 → breakout_active (or breakdown_active if trend==down)
    2. compression_flag > 0.5  → compression
    3. expansion_flag > 0.5    → expansion
    4. regime in ('range', 'ranging') → mean_reverting
    5. default → neutral

    Mapping version: 1.0
    """
    if breakout_confirmed is not None and breakout_confirmed > 0.5:
        return "breakdown_active" if trend_state == "down" else "breakout_active"
    if compression_flag is not None and compression_flag > 0.5:
        return "compression"
    if expansion_flag is not None and expansion_flag > 0.5:
        return "expansion"
    if regime in ("range", "ranging"):
        return "mean_reverting"
    return "neutral"


# ---------------------------------------------------------------------------
# Normalized numeric feature construction
# ---------------------------------------------------------------------------

def _compute_normalized_features(
    indicator: "IndicatorSnapshot",
) -> Tuple[float, float, float, float, float, float]:
    """Compute the 6 normalized feature components from an IndicatorSnapshot.

    Returns:
        (vol_percentile, atr_percentile, volume_percentile,
         range_expansion_percentile, realized_vol_z, distance_to_htf_anchor_atr)

    All returned values are either:
    - Bounded [0, 1] (percentiles) — safe for direct use in numeric_vector.
    - Unbounded (z-scores) — caller must normalize before adding to numeric_vector.
    - Unbounded (ATR multiples) — caller must clamp before adding to numeric_vector.
    """
    close = indicator.close or 1.0
    atr_14 = indicator.atr_14

    # vol_percentile / atr_percentile — from ATR/close ratio
    vol_ratio = (atr_14 / close) if (atr_14 and close > 0) else _VOL_Z_MEAN
    vol_percentile = min(1.0, max(0.0, vol_ratio / _ATR_RATIO_MAX))
    atr_percentile = vol_percentile  # same underlying signal in this release

    # volume_percentile — from volume_multiple (1.0 = average volume)
    vol_mult = indicator.volume_multiple
    if vol_mult is not None:
        # 0.5 → 0.0, 1.0 → 0.2, 2.5 → 1.0
        volume_percentile = min(1.0, max(0.0, (vol_mult - 0.5) / 2.0))
    else:
        volume_percentile = 0.5  # imputed: average

    # range_expansion_percentile — bb_bandwidth_pct_rank (already [0, 1])
    if indicator.bb_bandwidth_pct_rank is not None:
        range_expansion_percentile = float(indicator.bb_bandwidth_pct_rank)
    else:
        range_expansion_percentile = 0.5  # imputed: median

    # realized_vol_z — z-score of vol_ratio (typical mean=0.02, std=0.01)
    realized_vol_z = (vol_ratio - _VOL_Z_MEAN) / _VOL_Z_STD

    # distance_to_htf_anchor_atr — |close - htf_daily_close| / atr_14
    htf_close = indicator.htf_daily_close
    if htf_close and atr_14 and atr_14 > 0:
        distance_to_htf_anchor_atr = abs(close - htf_close) / atr_14
    else:
        distance_to_htf_anchor_atr = 0.0

    return (
        vol_percentile,
        atr_percentile,
        volume_percentile,
        range_expansion_percentile,
        realized_vol_z,
        distance_to_htf_anchor_atr,
    )


def _build_numeric_vector(
    vol_percentile: float,
    atr_percentile: float,
    volume_percentile: float,
    range_expansion_percentile: float,
    realized_vol_z: float,
    distance_to_htf_anchor_atr: float,
) -> List[float]:
    """Build the version-locked numeric vector (all components normalized to [0, 1]).

    Feature order is fixed by NUMERIC_VECTOR_FEATURE_NAMES_V1.
    Changing this function requires bumping FINGERPRINT_VERSION.
    """
    realized_vol_z_normed = min(1.0, max(0.0, (realized_vol_z + 3.0) / 6.0))
    distance_normed = min(1.0, max(0.0, distance_to_htf_anchor_atr / _HTF_ANCHOR_MAX_ATR))
    return [
        vol_percentile,
        atr_percentile,
        volume_percentile,
        range_expansion_percentile,
        realized_vol_z_normed,
        distance_normed,
    ]


def build_normalized_features_dict(
    vol_percentile: float,
    atr_percentile: float,
    volume_percentile: float,
    range_expansion_percentile: float,
    realized_vol_z: float,
    distance_to_htf_anchor_atr: float,
) -> Dict[str, float]:
    """Build the normalized_features dict for PolicySnapshot.DerivedSignalBlock.

    Keys match NUMERIC_VECTOR_FEATURE_NAMES_V1, values are all in [0, 1].
    This populates DerivedSignalBlock.normalized_features (R55 integration with R49).
    """
    vector = _build_numeric_vector(
        vol_percentile, atr_percentile, volume_percentile,
        range_expansion_percentile, realized_vol_z, distance_to_htf_anchor_atr,
    )
    return dict(zip(NUMERIC_VECTOR_FEATURE_NAMES_V1, vector))


# ---------------------------------------------------------------------------
# Fingerprint construction
# ---------------------------------------------------------------------------

def build_regime_fingerprint(
    indicator: "IndicatorSnapshot",
    asset_state: "AssetState",
    *,
    scope: Literal["symbol", "cohort"] = "symbol",
    cohort_id: Optional[str] = None,
    bar_id: Optional[str] = None,
    source_timeframe: Optional[str] = None,
) -> RegimeFingerprint:
    """Construct a RegimeFingerprint from a live indicator snapshot + asset state.

    Args:
        indicator: The current per-symbol indicator snapshot.
        asset_state: The summarized asset view (trend_state, vol_state, regime_assessment).
        scope: 'symbol' for per-symbol detector; 'cohort' for shared cohort detector.
        cohort_id: Cohort identifier (only set for scope='cohort').
        bar_id: Canonical bar key; auto-derived if not provided.
        source_timeframe: Timeframe used; defaults to indicator.timeframe.

    Returns:
        RegimeFingerprint with all categorical, confidence, and normalized numeric fields.
    """
    sym = indicator.symbol
    tf = source_timeframe or indicator.timeframe
    as_of = indicator.as_of
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    effective_bar_id = bar_id or f"{sym}|{tf}|{as_of.isoformat()}"

    # Vocabulary mapping — legacy labels → R55 canonical
    trend_canonical = map_trend_state(asset_state.trend_state)
    vol_canonical = map_vol_state(asset_state.vol_state)
    regime = asset_state.regime_assessment.regime if asset_state.regime_assessment else None
    structure = map_structure_state(
        compression_flag=indicator.compression_flag,
        expansion_flag=indicator.expansion_flag,
        breakout_confirmed=indicator.breakout_confirmed,
        trend_state=trend_canonical,
        regime=regime,
    )

    # Normalized numeric features
    (
        vol_percentile,
        atr_percentile,
        volume_percentile,
        range_expansion_percentile,
        realized_vol_z,
        distance_to_htf_anchor_atr,
    ) = _compute_normalized_features(indicator)

    numeric_vector = _build_numeric_vector(
        vol_percentile, atr_percentile, volume_percentile,
        range_expansion_percentile, realized_vol_z, distance_to_htf_anchor_atr,
    )

    # Confidence values (derived heuristically from signal clarity)
    trend_confidence = 0.8 if trend_canonical in ("up", "down") else 0.5

    # vol_confidence: higher when vol is at extremes (far from neutral 0.5)
    vol_confidence = min(1.0, abs(vol_percentile - 0.5) * 2.0 + 0.4)

    # structure_confidence: higher when R40 flags are unambiguous
    flags = [indicator.compression_flag, indicator.expansion_flag, indicator.breakout_confirmed]
    clear_flags = [f for f in flags if f is not None and (f > 0.7 or f < 0.3)]
    structure_confidence = min(0.9, 0.4 + len(clear_flags) * 0.2)

    regime_confidence = (
        asset_state.regime_assessment.confidence
        if asset_state.regime_assessment
        else 0.4
    )

    return RegimeFingerprint(
        fingerprint_version=FINGERPRINT_VERSION,
        schema_version=FINGERPRINT_SCHEMA_VERSION,
        symbol=sym,
        scope=scope,
        cohort_id=cohort_id,
        as_of_ts=as_of,
        bar_id=effective_bar_id,
        source_timeframe=tf,
        trend_state=trend_canonical,
        vol_state=vol_canonical,
        structure_state=structure,
        trend_confidence=trend_confidence,
        vol_confidence=vol_confidence,
        structure_confidence=structure_confidence,
        regime_confidence=regime_confidence,
        vol_percentile=vol_percentile,
        atr_percentile=atr_percentile,
        volume_percentile=volume_percentile,
        range_expansion_percentile=range_expansion_percentile,
        realized_vol_z=realized_vol_z,
        distance_to_htf_anchor_atr=distance_to_htf_anchor_atr,
        numeric_vector=numeric_vector,
        numeric_vector_feature_names=NUMERIC_VECTOR_FEATURE_NAMES_V1,
    )


# ---------------------------------------------------------------------------
# Distance function
# ---------------------------------------------------------------------------

def regime_fingerprint_distance(
    curr: RegimeFingerprint,
    prev: RegimeFingerprint,
    enter_threshold: float = _DEFAULT_ENTER_THRESHOLD,
    exit_threshold: float = _DEFAULT_EXIT_THRESHOLD,
    threshold_type: Literal["enter", "exit"] = "enter",
) -> RegimeDistanceResult:
    """Compute bounded [0, 1] distance between two regime fingerprints.

    Distance formula (version 1.0):
    1. Categorical mismatch terms (trend_state, vol_state, structure_state) — weighted
    2. Normalized L1 distance over numeric_vector — weighted
    3. Smoothed confidence delta — bounded contribution

    Since all weights sum to 1.0 and each term is bounded by its weight,
    the total distance is naturally bounded [0, 1] without further clamping.

    Returns RegimeDistanceResult with:
    - distance_value: bounded [0, 1]
    - component_contributions: per-term contributions (approximately sum to distance_value)
    - component_deltas: human-readable delta descriptions
    - weights: the weight vector used (version-locked)
    """
    weights = dict(_DISTANCE_WEIGHTS)

    # 1. Categorical mismatch terms (0 or 1, weighted)
    trend_changed = float(curr.trend_state != prev.trend_state)
    vol_changed = float(curr.vol_state != prev.vol_state)
    structure_changed = float(curr.structure_state != prev.structure_state)

    contrib_trend = weights["trend_state"] * trend_changed
    contrib_vol = weights["vol_state"] * vol_changed
    contrib_structure = weights["structure_state"] * structure_changed

    # 2. Normalized L1 distance over numeric_vector (result in [0, 1])
    n = min(len(curr.numeric_vector), len(prev.numeric_vector))
    if n > 0:
        l1 = sum(abs(a - b) for a, b in zip(curr.numeric_vector[:n], prev.numeric_vector[:n]))
        numeric_dist = l1 / n  # normalized; each component is in [0,1] so L1/n ∈ [0,1]
    else:
        numeric_dist = 0.0
    contrib_numeric = weights["numeric_vector"] * numeric_dist

    # 3. Confidence delta (bounded contribution)
    abs_conf_delta = abs(curr.regime_confidence - prev.regime_confidence)
    contrib_confidence = weights["confidence"] * min(1.0, abs_conf_delta)

    # Total distance
    distance = contrib_trend + contrib_vol + contrib_structure + contrib_numeric + contrib_confidence
    distance = min(1.0, max(0.0, distance))

    threshold_used = exit_threshold if threshold_type == "exit" else enter_threshold

    # Human-readable deltas
    deltas: Dict[str, Any] = {
        "trend_state": (
            f"{prev.trend_state} -> {curr.trend_state}" if trend_changed else "unchanged"
        ),
        "vol_state": (
            f"{prev.vol_state} -> {curr.vol_state}" if vol_changed else "unchanged"
        ),
        "structure_state": (
            f"{prev.structure_state} -> {curr.structure_state}" if structure_changed else "unchanged"
        ),
        "regime_confidence": f"{prev.regime_confidence:.3f} -> {curr.regime_confidence:.3f}",
        "numeric_l1_normalized": round(numeric_dist, 4),
    }

    return RegimeDistanceResult(
        distance_value=distance,
        threshold_enter=enter_threshold,
        threshold_exit=exit_threshold,
        threshold_used=threshold_used,
        threshold_type=threshold_type,
        component_contributions={
            "trend_state": round(contrib_trend, 4),
            "vol_state": round(contrib_vol, 4),
            "structure_state": round(contrib_structure, 4),
            "numeric_vector": round(contrib_numeric, 4),
            "confidence": round(contrib_confidence, 4),
        },
        component_deltas=deltas,
        weights=weights,
        confidence_delta=curr.regime_confidence - prev.regime_confidence,
    )


# ---------------------------------------------------------------------------
# Transition detector state machine
# ---------------------------------------------------------------------------

class RegimeTransitionDetector:
    """Deterministic asymmetric-hysteresis regime transition detector.

    Per-symbol (or per-cohort) state machine that:
    - Evaluates transitions on explicit evaluate() calls
    - Caller manages HTF-close gating (pass htf_gate_eligible=True on closes)
    - Auto-detects shock conditions (extreme vol) and bypasses HTF gate
    - Uses asymmetric thresholds (enter_threshold > exit_threshold) for hysteresis
    - Enforces min_dwell (min time in a regime before new transition is allowed)
    - Enforces cooldown (min time after a transition before next one is considered)
    - Emits RegimeTransitionTelemetryEvent on EVERY evaluation (fired or not)

    Usage:
        detector = RegimeTransitionDetector("BTC-USD")
        ts = datetime.now(timezone.utc)
        event = detector.evaluate(fingerprint, ts, htf_gate_eligible=True)
        if event.decision.transition_fired:
            # regime changed — trigger policy loop reevaluation
            pass
    """

    def __init__(
        self,
        symbol: str,
        scope: Literal["symbol", "cohort"] = "symbol",
        enter_threshold: float = _DEFAULT_ENTER_THRESHOLD,
        exit_threshold: float = _DEFAULT_EXIT_THRESHOLD,
        min_dwell_seconds: float = 900.0,   # 15 minutes
        cooldown_seconds: float = 300.0,    # 5 minutes
        shock_vol_percentile: float = _SHOCK_VOL_PERCENTILE,
        shock_realized_vol_z: float = _SHOCK_REALIZED_VOL_Z,
    ) -> None:
        self.symbol = symbol
        self.scope: Literal["symbol", "cohort"] = scope
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.min_dwell_seconds = min_dwell_seconds
        self.cooldown_seconds = cooldown_seconds
        self.shock_vol_percentile = shock_vol_percentile
        self.shock_realized_vol_z = shock_realized_vol_z
        self._state = RegimeTransitionDetectorState(symbol=symbol, scope=scope)

    @property
    def state(self) -> RegimeTransitionDetectorState:
        return self._state

    def load_state(self, state: RegimeTransitionDetectorState) -> None:
        """Restore detector from a previously serialized state (deterministic replay)."""
        self._state = state

    def _is_shock(self, fingerprint: RegimeFingerprint) -> bool:
        """Return True if shock-override conditions are met on this fingerprint."""
        return (
            fingerprint.vol_percentile >= self.shock_vol_percentile
            or fingerprint.realized_vol_z >= self.shock_realized_vol_z
        )

    def _dwell_seconds(self, now: datetime) -> Optional[float]:
        if self._state.current_regime_entered_ts is None:
            return None
        entered = self._state.current_regime_entered_ts
        if entered.tzinfo is None:
            entered = entered.replace(tzinfo=timezone.utc)
        return (now - entered).total_seconds()

    def _cooldown_remaining(self, now: datetime) -> Optional[float]:
        if self._state.cooldown_until_ts is None:
            return None
        return max(0.0, (self._state.cooldown_until_ts - now).total_seconds())

    def _update_stable(self, now: datetime) -> None:
        """Advance consecutive_stable_evals counter without touching other state."""
        self._state = RegimeTransitionDetectorState(
            symbol=self._state.symbol,
            scope=self._state.scope,
            current_fingerprint=self._state.current_fingerprint,
            last_transition_ts=self._state.last_transition_ts,
            current_regime_entered_ts=self._state.current_regime_entered_ts,
            cooldown_until_ts=self._state.cooldown_until_ts,
            consecutive_stable_evals=self._state.consecutive_stable_evals + 1,
            total_transitions=self._state.total_transitions,
        )

    def evaluate(
        self,
        fingerprint: RegimeFingerprint,
        current_ts: datetime,
        htf_gate_eligible: bool = True,
    ) -> RegimeTransitionTelemetryEvent:
        """Evaluate whether a regime transition should fire.

        Every call emits a RegimeTransitionTelemetryEvent regardless of outcome.
        This enforces the non-negotiable telemetry requirement: every evaluation
        must be explainable.

        Args:
            fingerprint: The new regime fingerprint to evaluate.
            current_ts: Current evaluation timestamp (UTC; naive datetimes converted).
            htf_gate_eligible: True when an HTF bar has closed. Shock overrides bypass this.

        Returns:
            RegimeTransitionTelemetryEvent with full transition decision and telemetry.
        """
        if current_ts.tzinfo is None:
            current_ts = current_ts.replace(tzinfo=timezone.utc)

        event_id = str(uuid4())
        now = current_ts
        prior = self._state.current_fingerprint

        # ── Initial state ──────────────────────────────────────────────────
        # On first evaluation there is no prior fingerprint — the detector simply
        # bootstraps. We do NOT set a cooldown here because no real regime shift
        # occurred; the cooldown should only apply to genuine transitions.
        if prior is None:
            decision = RegimeTransitionDecision(
                transition_fired=True,
                reason_code="initial_state",
                prior_fingerprint=None,
                new_fingerprint=fingerprint,
                distance_result=None,
                suppressed=False,
                shock_override_used=False,
                htf_gate_eligible=htf_gate_eligible,
                as_of_ts=now,
                symbol=self.symbol,
                scope=self.scope,
            )
            self._state = RegimeTransitionDetectorState(
                symbol=self.symbol,
                scope=self.scope,
                current_fingerprint=fingerprint,
                last_transition_ts=now,
                current_regime_entered_ts=now,
                cooldown_until_ts=None,  # no cooldown on bootstrap
                consecutive_stable_evals=0,
                total_transitions=1,
            )
            return RegimeTransitionTelemetryEvent(
                event_id=event_id,
                emitted_at=now,
                decision=decision,
                dwell_seconds=0.0,
                cooldown_remaining_seconds=None,
                consecutive_stable_evals=0,
            )

        # ── Compute distance ───────────────────────────────────────────────
        distance_result = regime_fingerprint_distance(
            fingerprint,
            prior,
            enter_threshold=self.enter_threshold,
            exit_threshold=self.exit_threshold,
            threshold_type="enter",
        )

        # ── Shock override detection ────────────────────────────────────────
        shock_active = self._is_shock(fingerprint)
        effective_htf_eligible = htf_gate_eligible or shock_active

        # ── HTF gate check ─────────────────────────────────────────────────
        if not effective_htf_eligible:
            decision = RegimeTransitionDecision(
                transition_fired=False,
                reason_code="htf_gate_not_ready",
                prior_fingerprint=prior,
                new_fingerprint=fingerprint,
                distance_result=distance_result,
                suppressed=True,
                shock_override_used=False,
                htf_gate_eligible=htf_gate_eligible,
                as_of_ts=now,
                symbol=self.symbol,
                scope=self.scope,
            )
            self._update_stable(now)
            return RegimeTransitionTelemetryEvent(
                event_id=event_id,
                emitted_at=now,
                decision=decision,
                dwell_seconds=self._dwell_seconds(now),
                cooldown_remaining_seconds=self._cooldown_remaining(now),
                consecutive_stable_evals=self._state.consecutive_stable_evals,
            )

        # ── Cooldown check ─────────────────────────────────────────────────
        if self._state.cooldown_until_ts and now < self._state.cooldown_until_ts:
            cooldown_rem = (self._state.cooldown_until_ts - now).total_seconds()
            decision = RegimeTransitionDecision(
                transition_fired=False,
                reason_code="suppressed_cooldown",
                prior_fingerprint=prior,
                new_fingerprint=fingerprint,
                distance_result=distance_result,
                suppressed=True,
                shock_override_used=shock_active,
                htf_gate_eligible=htf_gate_eligible,
                as_of_ts=now,
                symbol=self.symbol,
                scope=self.scope,
            )
            self._update_stable(now)
            return RegimeTransitionTelemetryEvent(
                event_id=event_id,
                emitted_at=now,
                decision=decision,
                dwell_seconds=self._dwell_seconds(now),
                cooldown_remaining_seconds=cooldown_rem,
                consecutive_stable_evals=self._state.consecutive_stable_evals,
            )

        # ── Min dwell check (shock bypasses this too) ──────────────────────
        dwell_secs = self._dwell_seconds(now)
        if (
            dwell_secs is not None
            and dwell_secs < self.min_dwell_seconds
            and not shock_active
        ):
            decision = RegimeTransitionDecision(
                transition_fired=False,
                reason_code="suppressed_min_dwell",
                prior_fingerprint=prior,
                new_fingerprint=fingerprint,
                distance_result=distance_result,
                suppressed=True,
                shock_override_used=False,
                htf_gate_eligible=htf_gate_eligible,
                as_of_ts=now,
                symbol=self.symbol,
                scope=self.scope,
            )
            self._update_stable(now)
            return RegimeTransitionTelemetryEvent(
                event_id=event_id,
                emitted_at=now,
                decision=decision,
                dwell_seconds=dwell_secs,
                cooldown_remaining_seconds=self._cooldown_remaining(now),
                consecutive_stable_evals=self._state.consecutive_stable_evals,
            )

        # ── Distance threshold check ───────────────────────────────────────
        threshold_crossed = distance_result.distance_value >= self.enter_threshold
        shock_threshold_crossed = shock_active and distance_result.distance_value > 0.01

        if threshold_crossed or shock_threshold_crossed:
            reason = (
                "shock_override_volatility_jump" if shock_active
                else "distance_enter_threshold_crossed"
            )
            decision = RegimeTransitionDecision(
                transition_fired=True,
                reason_code=reason,
                prior_fingerprint=prior,
                new_fingerprint=fingerprint,
                distance_result=distance_result,
                suppressed=False,
                shock_override_used=shock_active,
                htf_gate_eligible=htf_gate_eligible,
                as_of_ts=now,
                symbol=self.symbol,
                scope=self.scope,
            )
            self._state = RegimeTransitionDetectorState(
                symbol=self.symbol,
                scope=self.scope,
                current_fingerprint=fingerprint,
                last_transition_ts=now,
                current_regime_entered_ts=now,
                cooldown_until_ts=now + timedelta(seconds=self.cooldown_seconds),
                consecutive_stable_evals=0,
                total_transitions=self._state.total_transitions + 1,
            )
            return RegimeTransitionTelemetryEvent(
                event_id=event_id,
                emitted_at=now,
                decision=decision,
                dwell_seconds=dwell_secs,
                cooldown_remaining_seconds=self.cooldown_seconds,
                consecutive_stable_evals=0,
            )

        # ── No transition ──────────────────────────────────────────────────
        decision = RegimeTransitionDecision(
            transition_fired=False,
            reason_code="distance_below_threshold",
            prior_fingerprint=prior,
            new_fingerprint=fingerprint,
            distance_result=distance_result,
            suppressed=False,
            shock_override_used=False,
            htf_gate_eligible=htf_gate_eligible,
            as_of_ts=now,
            symbol=self.symbol,
            scope=self.scope,
        )
        self._update_stable(now)
        return RegimeTransitionTelemetryEvent(
            event_id=event_id,
            emitted_at=now,
            decision=decision,
            dwell_seconds=self._dwell_seconds(now),
            cooldown_remaining_seconds=self._cooldown_remaining(now),
            consecutive_stable_evals=self._state.consecutive_stable_evals,
        )

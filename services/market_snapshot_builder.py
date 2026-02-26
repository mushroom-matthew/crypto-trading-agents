"""Builder functions for TickSnapshot and PolicySnapshot.

Runbook 49: Market Snapshot Definition.
Runbook 55: Regime fingerprint — normalized_features now populated.
Runbook 58: Structure engine — structure_snapshot_id/hash threaded into snapshots.

Usage:
    tick = build_tick_snapshot(indicator_snapshot)
    policy = build_policy_snapshot(llm_input)

Both builders:
- normalize values
- compute snapshot_hash over canonical JSON
- attach quality flags (staleness, missing sections)
- return an immutable (frozen-at-construction) snapshot object

Runbook 55 integration:
- DerivedSignalBlock.normalized_features is now populated from the R55 fingerprint
  builder's normalized feature computation (_compute_normalized_features).
- normalized_features_version is set to the FINGERPRINT_VERSION constant, linking
  the PolicySnapshot to the R55 numeric vector contract.
- memory_bundle remains deferred to Runbook 51.

Runbook 58 integration:
- build_tick_snapshot and build_policy_snapshot both accept an optional
  structure_snapshot parameter.  When provided, snapshot_id and snapshot_hash
  are threaded into the returned snapshot for R49 provenance linkage.
- DerivedSignalBlock gains nearest_support_pct and nearest_resistance_pct
  from the structure snapshot's level ladder.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from schemas.market_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    DerivedSignalBlock,
    FeatureDerivationEntry,
    FeatureDerivationLog,
    NumericalSignalBlock,
    PolicySnapshot,
    SnapshotProvenance,
    SnapshotQuality,
    TickSnapshot,
    compute_snapshot_hash,
)
from schemas.llm_strategist import IndicatorSnapshot, LLMInput
from schemas.regime_fingerprint import FINGERPRINT_VERSION as _R55_FINGERPRINT_VERSION
from schemas.structure_engine import StructureSnapshot
from services.regime_transition_detector import (
    _compute_normalized_features as _r55_compute_normalized_features,
    build_normalized_features_dict,
)

logger = logging.getLogger(__name__)

_FEATURE_PIPELINE_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _bar_id(symbol: str, timeframe: str, as_of: datetime) -> str:
    """Stable canonical bar key: <symbol>|<timeframe>|<iso8601>."""
    return f"{symbol}|{timeframe}|{as_of.isoformat()}"


def _compute_pipeline_hash(entries: list) -> str:
    """sha256 of sorted canonical derivation entry list."""
    payload = json.dumps(
        sorted([e.model_dump() for e in entries], key=lambda d: d.get("transform", "")),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _feature_pipeline_hash() -> str:
    """Version-stable hash representing the indicator pipeline config."""
    return hashlib.sha256(_FEATURE_PIPELINE_VERSION.encode("utf-8")).hexdigest()[:16]


def _staleness(as_of_ts: datetime, now: datetime, max_seconds: float) -> SnapshotQuality:
    """Compute staleness flag and duration."""
    age = (now - as_of_ts).total_seconds()
    is_stale = age > max_seconds
    warnings = [f"Snapshot is {age:.0f}s old (threshold {max_seconds:.0f}s)"] if is_stale else []
    return SnapshotQuality(
        is_stale=is_stale,
        staleness_seconds=max(0.0, age),
        quality_warnings=warnings,
    )


# ---------------------------------------------------------------------------
# TickSnapshot builder
# ---------------------------------------------------------------------------

def build_tick_snapshot(
    indicator: IndicatorSnapshot,
    *,
    bar_id: Optional[str] = None,
    max_staleness_seconds: float = 300.0,
    parent_tick_snapshot_id: Optional[str] = None,
    structure_snapshot: Optional[StructureSnapshot] = None,
) -> TickSnapshot:
    """Build a lightweight TickSnapshot from an IndicatorSnapshot.

    Args:
        indicator: The current per-symbol indicator snapshot.
        bar_id: Explicit bar key; auto-computed from symbol/timeframe/as_of when absent.
        max_staleness_seconds: Age threshold for the is_stale flag (default 5 min).
        parent_tick_snapshot_id: Optional link to a prior snapshot in a chain.
        structure_snapshot: Optional R58 StructureSnapshot to embed as a reference.
    """
    now = _now_utc()
    as_of = indicator.as_of
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    effective_bar_id = bar_id or _bar_id(indicator.symbol, indicator.timeframe, as_of)

    # Minimal derivation log
    deriv_entries = [
        FeatureDerivationEntry(
            transform="indicator_snapshot",
            version=_FEATURE_PIPELINE_VERSION,
            input_window_bars=None,
            params={"symbol": indicator.symbol, "timeframe": indicator.timeframe},
            output_fields=["close", "volume", "atr_14", "rsi_14",
                           "compression_flag", "expansion_flag", "breakout_confirmed"],
        )
    ]
    pipeline_hash = _compute_pipeline_hash(deriv_entries)
    deriv_log = FeatureDerivationLog(entries=deriv_entries, pipeline_hash=pipeline_hash)

    # Data fields for hashing
    data_fields: Dict[str, Any] = {
        "symbol": indicator.symbol,
        "timeframe": indicator.timeframe,
        "as_of_ts": as_of.isoformat(),
        "close": indicator.close,
        "volume": indicator.volume,
        "atr_14": indicator.atr_14,
        "rsi_14": indicator.rsi_14,
        "compression_flag": indicator.compression_flag,
        "expansion_flag": indicator.expansion_flag,
        "breakout_confirmed": indicator.breakout_confirmed,
        "bar_id": effective_bar_id,
    }
    snapshot_hash = compute_snapshot_hash(data_fields)
    snapshot_id = str(uuid4())

    provenance = SnapshotProvenance(
        snapshot_version=SNAPSHOT_SCHEMA_VERSION,
        snapshot_kind="tick",
        snapshot_id=snapshot_id,
        snapshot_hash=snapshot_hash,
        feature_pipeline_hash=pipeline_hash,
        as_of_ts=as_of,
        generated_at_ts=now,
        created_at_bar_id=effective_bar_id,
        symbol=indicator.symbol,
        timeframe=indicator.timeframe,
        parent_tick_snapshot_id=parent_tick_snapshot_id,
        feature_pipeline_version=_FEATURE_PIPELINE_VERSION,
    )

    quality = _staleness(as_of, now, max_staleness_seconds)

    # Pull AssetState-derived fields from the indicator where available
    # trend_state / vol_state are on AssetState, not IndicatorSnapshot — skip here

    # R58: embed structure snapshot reference
    struct_id = structure_snapshot.snapshot_id if structure_snapshot else None
    struct_hash = structure_snapshot.snapshot_hash if structure_snapshot else None

    return TickSnapshot(
        provenance=provenance,
        quality=quality,
        feature_derivation_log=deriv_log,
        close=indicator.close,
        volume=indicator.volume,
        atr_14=indicator.atr_14,
        rsi_14=indicator.rsi_14,
        compression_flag=indicator.compression_flag,
        expansion_flag=indicator.expansion_flag,
        breakout_confirmed=indicator.breakout_confirmed,
        structure_snapshot_id=struct_id,
        structure_snapshot_hash=struct_hash,
    )


# ---------------------------------------------------------------------------
# PolicySnapshot builder
# ---------------------------------------------------------------------------

def build_policy_snapshot(
    llm_input: LLMInput,
    *,
    bar_id: Optional[str] = None,
    policy_event_type: Optional[str] = None,
    policy_event_id: Optional[str] = None,
    policy_event_metadata: Optional[Dict[str, Any]] = None,
    max_staleness_seconds: float = 3600.0,
    memory_bundle_id: Optional[str] = None,
    memory_bundle_summary: Optional[str] = None,
    structure_snapshots: Optional[Dict[str, StructureSnapshot]] = None,
) -> PolicySnapshot:
    """Build a PolicySnapshot from an LLMInput bundle.

    Computes snapshot_hash over canonical serialization of portfolio and asset
    data, records provenance, and marks missing optional sections explicitly.

    normalized_features is left empty (populated by Runbook 55).
    memory bundle is optional (populated by Runbook 51/48).

    Args:
        llm_input: The LLMInput bundle going to the strategist.
        bar_id: Canonical bar key; auto-derived from first asset's latest indicator.
        policy_event_type: Event that triggered this snapshot (e.g. 'plan_generation').
        policy_event_id: UUID of the triggering policy event.
        policy_event_metadata: Arbitrary metadata about the policy event.
        max_staleness_seconds: Staleness threshold (default 60 min).
        memory_bundle_id: Optional reference to a memory bundle (Runbook 51).
        memory_bundle_summary: Optional text summary of retrieved memory context.
        structure_snapshots: Optional dict of {symbol: StructureSnapshot} for R58 integration.
    """
    now = _now_utc()

    # Derive canonical time from portfolio timestamp or first asset indicator
    as_of: datetime = llm_input.portfolio.timestamp
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    # Determine symbol and timeframe from the first asset's latest indicator
    primary_symbol = ""
    primary_timeframe = "1h"
    if llm_input.assets:
        first_asset = llm_input.assets[0]
        primary_symbol = first_asset.symbol
        if first_asset.indicators:
            primary_timeframe = first_asset.indicators[-1].timeframe

    effective_bar_id = bar_id or _bar_id(primary_symbol, primary_timeframe, as_of)

    # Build per-symbol numerical and derived blocks
    numerical: Dict[str, NumericalSignalBlock] = {}
    derived: Dict[str, DerivedSignalBlock] = {}

    for asset in llm_input.assets:
        sym = asset.symbol
        if asset.indicators:
            ind = asset.indicators[-1]  # most recent timeframe
            numerical[sym] = NumericalSignalBlock(
                close=ind.close,
                volume=ind.volume,
                atr_14=ind.atr_14,
                rsi_14=ind.rsi_14,
                macd_hist=ind.macd_hist,
                bb_bandwidth_pct_rank=ind.bb_bandwidth_pct_rank,
                htf_daily_atr=ind.htf_daily_atr,
            )
        # R55: build normalized_features dict from fingerprint feature components
        norm_features: Dict[str, float] = {}
        norm_version: Optional[str] = None
        if asset.indicators:
            ind = asset.indicators[-1]
            try:
                (
                    vol_pct, atr_pct, volume_pct,
                    range_exp_pct, rvol_z, htf_dist,
                ) = _r55_compute_normalized_features(ind)
                norm_features = build_normalized_features_dict(
                    vol_pct, atr_pct, volume_pct, range_exp_pct, rvol_z, htf_dist,
                )
                norm_version = _R55_FINGERPRINT_VERSION
            except Exception:
                logger.debug("R55 normalized_features build failed for %s — leaving empty", sym)

        # R58: extract nearest support/resistance from structure snapshot for this symbol
        struct_snap = (structure_snapshots or {}).get(sym)
        struct_id: Optional[str] = struct_snap.snapshot_id if struct_snap else None
        nearest_support_pct: Optional[float] = None
        nearest_resistance_pct: Optional[float] = None
        if struct_snap:
            for tf in struct_snap.ladders.values():
                all_sup = tf.near_supports + tf.mid_supports
                all_res = tf.near_resistances + tf.mid_resistances
                if all_sup:
                    cand = min(all_sup, key=lambda l: l.distance_abs)
                    if nearest_support_pct is None or cand.distance_pct < nearest_support_pct:
                        nearest_support_pct = cand.distance_pct
                if all_res:
                    cand = min(all_res, key=lambda l: l.distance_abs)
                    if nearest_resistance_pct is None or cand.distance_pct < nearest_resistance_pct:
                        nearest_resistance_pct = cand.distance_pct

        derived[sym] = DerivedSignalBlock(
            trend_state=asset.trend_state,
            vol_state=asset.vol_state,
            regime=asset.regime_assessment.regime if asset.regime_assessment else None,
            compression_flag=(
                asset.indicators[-1].compression_flag if asset.indicators else None
            ),
            expansion_flag=(
                asset.indicators[-1].expansion_flag if asset.indicators else None
            ),
            breakout_confirmed=(
                asset.indicators[-1].breakout_confirmed if asset.indicators else None
            ),
            normalized_features=norm_features,
            normalized_features_version=norm_version,
            structure_snapshot_id=struct_id,
            nearest_support_pct=nearest_support_pct,
            nearest_resistance_pct=nearest_resistance_pct,
        )

    # Mark absent optional sections
    missing_sections = []
    if not any(d.compression_flag is not None for d in derived.values()):
        missing_sections.append("compression_signals")
    missing_sections.append("text_signals")   # always absent until text integration
    missing_sections.append("visual_signals")  # always absent until visual integration
    if not memory_bundle_id:
        missing_sections.append("memory_bundle")
    if not any(d.normalized_features for d in derived.values()):
        missing_sections.append("normalized_features")  # populated by R55
    if not structure_snapshots:
        missing_sections.append("structure_engine")  # populated by R58 when active

    # Derivation log
    deriv_entries = [
        FeatureDerivationEntry(
            transform="llm_input_aggregate",
            version=_FEATURE_PIPELINE_VERSION,
            params={"symbols": [a.symbol for a in llm_input.assets]},
            output_fields=["numerical", "derived"],
        )
    ]
    pipeline_hash = _compute_pipeline_hash(deriv_entries)
    deriv_log = FeatureDerivationLog(entries=deriv_entries, pipeline_hash=pipeline_hash)

    # Canonical data dict for hashing
    data_fields: Dict[str, Any] = {
        "equity": llm_input.portfolio.equity,
        "cash": llm_input.portfolio.cash,
        "as_of_ts": as_of.isoformat(),
        "bar_id": effective_bar_id,
        "symbols": sorted(a.symbol for a in llm_input.assets),
        "numerical": {
            sym: blk.model_dump(exclude_none=True)
            for sym, blk in numerical.items()
        },
    }
    snapshot_hash = compute_snapshot_hash(data_fields)
    snapshot_id = str(uuid4())

    provenance = SnapshotProvenance(
        snapshot_version=SNAPSHOT_SCHEMA_VERSION,
        snapshot_kind="policy",
        snapshot_id=snapshot_id,
        snapshot_hash=snapshot_hash,
        feature_pipeline_hash=pipeline_hash,
        as_of_ts=as_of,
        generated_at_ts=now,
        created_at_bar_id=effective_bar_id,
        symbol=primary_symbol,
        timeframe=primary_timeframe,
        policy_event_id=policy_event_id,
        feature_pipeline_version=_FEATURE_PIPELINE_VERSION,
    )

    quality = _staleness(as_of, now, max_staleness_seconds)
    quality = SnapshotQuality(
        is_stale=quality.is_stale,
        staleness_seconds=quality.staleness_seconds,
        missing_sections=missing_sections,
        quality_warnings=quality.quality_warnings,
    )

    # R58: derive aggregate structure metadata from all symbol structure snapshots
    all_struct_snaps = list((structure_snapshots or {}).values())
    primary_struct = all_struct_snaps[0] if all_struct_snaps else None
    struct_events_count = sum(len(s.events) for s in all_struct_snaps) if all_struct_snaps else None
    struct_policy_priority = primary_struct.policy_event_priority if primary_struct else None

    return PolicySnapshot(
        provenance=provenance,
        quality=quality,
        feature_derivation_log=deriv_log,
        numerical=numerical,
        derived=derived,
        text_digest=None,
        visual_fingerprint=None,
        memory_bundle_id=memory_bundle_id,
        memory_bundle_summary=memory_bundle_summary,
        expectation_summary={},
        policy_event_type=policy_event_type or "plan_generation",
        policy_event_metadata=policy_event_metadata or {},
        equity=llm_input.portfolio.equity,
        cash=llm_input.portfolio.cash,
        open_positions=sorted(llm_input.portfolio.positions.keys()),
        structure_snapshot_id=primary_struct.snapshot_id if primary_struct else None,
        structure_snapshot_hash=primary_struct.snapshot_hash if primary_struct else None,
        structure_events_count=struct_events_count,
        structure_policy_priority=struct_policy_priority,
    )

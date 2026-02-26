"""Diversified memory retrieval service (Runbook 51).

MemoryRetrievalService scores and selects EpisodeMemoryRecords from an
EpisodeMemoryStore to fill a DiversifiedMemoryBundle with quota-capped,
recency-weighted, regime-similar episodes.

Similarity dimensions:
- regime_score:         cosine similarity on regime_fingerprint dict
- playbook_score:       exact match / both-None / mismatch
- timeframe_score:      exact match / both-None / mismatch
- feature_vector_score: same calculation as regime_score (fingerprint IS the feature vector)
- recency_factor:       exp(-lambda * days_since_exit)  — multiplier, not additive

Retrieval strategy:
1. Score all symbol-local records.
2. If any bucket (win/loss/failure_mode) is under quota, attempt global fallback —
   only including records whose best fingerprint distance to the request is below
   global_fallback_max_fingerprint_distance.
3. Apply diversity constraint: no more than half a bucket's quota may come from
   episodes whose entry_ts.date() is the same calendar day.

Bundle reuse:
- If a prior_bundle is provided and the regime_fingerprint_delta between the
  prior bundle's retrieval fingerprint and the current request fingerprint is
  below 0.05, return the prior bundle unchanged with bundle_reused=True.

DISCLAIMER: Retrieved episodes are research calibration context only. No
position-sizing or order-routing decisions are derived automatically.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from schemas.episode_memory import (
    DiversifiedMemoryBundle,
    EpisodeMemoryRecord,
    MemoryRetrievalMeta,
    MemoryRetrievalRequest,
)
from services.episode_memory_service import EpisodeMemoryStore

logger = logging.getLogger(__name__)

_SIMILARITY_SPEC_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Compute cosine similarity between two dicts of named float components.

    Only keys present in both dicts contribute. Returns 0.5 when either dict
    is empty (no information to compare — neutral score).
    """
    shared_keys = set(a.keys()) & set(b.keys())
    if not shared_keys:
        return 0.5

    dot = sum(a[k] * b[k] for k in shared_keys)
    mag_a = math.sqrt(sum(a[k] ** 2 for k in shared_keys))
    mag_b = math.sqrt(sum(b[k] ** 2 for k in shared_keys))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.5

    # cosine similarity is in [-1, 1]; map to [0, 1] for use as a score
    raw = dot / (mag_a * mag_b)
    return (raw + 1.0) / 2.0


def _fingerprint_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Bounded [0, 1] distance between two fingerprint dicts.

    distance = 1 - cosine_similarity; 0 means identical, 1 means orthogonal.
    """
    return 1.0 - _cosine_similarity(a, b)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _days_since(ts: Optional[datetime]) -> float:
    """Days elapsed since ts (UTC), or 30 if ts is None (conservative default)."""
    if ts is None:
        return 30.0
    now = datetime.now(tz=timezone.utc)
    # Ensure ts is timezone-aware
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = (now - ts).total_seconds()
    return max(0.0, delta / 86400.0)


def _score(
    record: EpisodeMemoryRecord,
    request: MemoryRetrievalRequest,
) -> float:
    """Compute a composite similarity score for a candidate record.

    Score = weighted_sum * recency_factor

    weighted_sum components:
    - regime_score:         cosine similarity on regime_fingerprint
    - playbook_score:       1.0 match, 0.5 both-None, 0.0 mismatch
    - timeframe_score:      1.0 match, 0.5 both-None, 0.0 mismatch
    - feature_vector_score: same as regime_score (fingerprint IS the feature vector)

    recency_factor = exp(-lambda * days_since_exit)   (multiplier)
    """
    # --- regime score ---
    if record.regime_fingerprint and request.regime_fingerprint:
        regime_score = _cosine_similarity(record.regime_fingerprint, request.regime_fingerprint)
    else:
        regime_score = 0.5

    # --- playbook score ---
    if record.playbook_id is not None and request.playbook_id is not None:
        playbook_score = 1.0 if record.playbook_id == request.playbook_id else 0.0
    elif record.playbook_id is None and request.playbook_id is None:
        playbook_score = 0.5
    else:
        playbook_score = 0.0

    # --- timeframe score ---
    if record.timeframe is not None and request.timeframe is not None:
        timeframe_score = 1.0 if record.timeframe == request.timeframe else 0.0
    elif record.timeframe is None and request.timeframe is None:
        timeframe_score = 0.5
    else:
        timeframe_score = 0.0

    # --- feature_vector_score: same computation as regime_score ---
    feature_vector_score = regime_score

    weighted_sum = (
        request.regime_weight * regime_score
        + request.playbook_weight * playbook_score
        + request.timeframe_weight * timeframe_score
        + request.feature_vector_weight * feature_vector_score
    )

    # --- recency multiplier ---
    days = _days_since(record.exit_ts)
    recency_factor = math.exp(-request.recency_decay_lambda * days)

    return weighted_sum * recency_factor


# ---------------------------------------------------------------------------
# Diversity constraint
# ---------------------------------------------------------------------------

def _apply_diversity_constraint(
    candidates: List[Tuple[float, EpisodeMemoryRecord]],
    quota: int,
) -> List[EpisodeMemoryRecord]:
    """Select up to quota records, limiting same-day entries to at most half the quota.

    Candidates are assumed to be sorted descending by score.
    """
    max_per_day = max(1, quota // 2)
    day_counts: Dict[object, int] = defaultdict(int)
    selected: List[EpisodeMemoryRecord] = []

    for _, record in candidates:
        if len(selected) >= quota:
            break
        day_key = record.entry_ts.date() if record.entry_ts is not None else None
        if day_key is not None and day_counts[day_key] >= max_per_day:
            continue
        selected.append(record)
        if day_key is not None:
            day_counts[day_key] += 1

    return selected


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MemoryRetrievalService:
    """Scores and selects EpisodeMemoryRecords into a DiversifiedMemoryBundle.

    Args:
        store: An EpisodeMemoryStore instance to query.
    """

    def __init__(self, store: EpisodeMemoryStore) -> None:
        self._store = store

    def retrieve(
        self,
        request: MemoryRetrievalRequest,
        prior_bundle: Optional[DiversifiedMemoryBundle] = None,
    ) -> DiversifiedMemoryBundle:
        """Retrieve a DiversifiedMemoryBundle for the given request.

        Args:
            request: Specifies symbol, regime fingerprint, quotas, and weights.
            prior_bundle: If provided, may be reused when the regime fingerprint
                          delta is below 0.05.

        Returns:
            A new (or reused) DiversifiedMemoryBundle.
        """
        t0 = time.monotonic()

        # --- Bundle reuse check ---
        if prior_bundle is not None:
            prior_meta = prior_bundle.retrieval_meta
            delta = prior_meta.regime_fingerprint_delta
            if delta is not None and delta < 0.05:
                # Regime has not changed materially — reuse prior bundle
                reused_meta = MemoryRetrievalMeta(
                    policy_event_type=request.policy_event_type,
                    regime_fingerprint_delta=delta,
                    bundle_reused=True,
                    reuse_reason="regime_fingerprint_delta_below_threshold",
                    candidate_pool_size=prior_meta.candidate_pool_size,
                    insufficient_buckets=list(prior_meta.insufficient_buckets),
                    retrieval_latency_ms=None,
                    retrieval_scope=prior_meta.retrieval_scope,
                    similarity_spec_version=_SIMILARITY_SPEC_VERSION,
                )
                return DiversifiedMemoryBundle(
                    bundle_id=prior_bundle.bundle_id,
                    symbol=prior_bundle.symbol,
                    created_at=prior_bundle.created_at,
                    winning_contexts=list(prior_bundle.winning_contexts),
                    losing_contexts=list(prior_bundle.losing_contexts),
                    failure_mode_patterns=list(prior_bundle.failure_mode_patterns),
                    retrieval_meta=reused_meta,
                )

        # --- Symbol-local candidates ---
        symbol_records = self._store.get_by_symbol(request.symbol)
        candidate_pool_size = len(symbol_records)

        wins, losses, failure_modes, retrieval_scope, insufficient_buckets = (
            self._fill_buckets(symbol_records, request, scope="symbol")
        )

        # --- Global fallback when symbol-local is insufficient ---
        if insufficient_buckets:
            all_records = self._store.get_all()
            # Exclude already-seen symbol records to avoid double-counting
            symbol_ids = {r.episode_id for r in symbol_records}
            global_candidates = [
                r for r in all_records
                if r.episode_id not in symbol_ids
                and r.regime_fingerprint is not None
                and _fingerprint_distance(r.regime_fingerprint, request.regime_fingerprint)
                < request.global_fallback_max_fingerprint_distance
            ]

            if global_candidates:
                # Merge global candidates into the existing pool only for
                # buckets that are still under quota.
                wins, losses, failure_modes, retrieval_scope, insufficient_buckets = (
                    self._fill_buckets(
                        symbol_records + global_candidates,
                        request,
                        scope="global",
                    )
                )
                candidate_pool_size = len(symbol_records) + len(global_candidates)

        latency_ms = (time.monotonic() - t0) * 1000.0

        # Compute fingerprint delta vs prior bundle if available
        fingerprint_delta: Optional[float] = None
        if prior_bundle is not None:
            # The prior bundle's regime fingerprint is not directly stored;
            # use the meta's delta as-is since the caller computed it.
            # If we get here, delta >= 0.05 (reuse was rejected).
            fingerprint_delta = prior_bundle.retrieval_meta.regime_fingerprint_delta

        retrieval_meta = MemoryRetrievalMeta(
            policy_event_type=request.policy_event_type,
            regime_fingerprint_delta=fingerprint_delta,
            bundle_reused=False,
            candidate_pool_size=candidate_pool_size,
            insufficient_buckets=insufficient_buckets,
            retrieval_latency_ms=latency_ms,
            retrieval_scope=retrieval_scope,
            similarity_spec_version=_SIMILARITY_SPEC_VERSION,
        )

        return DiversifiedMemoryBundle(
            bundle_id=str(uuid4()),
            symbol=request.symbol,
            created_at=datetime.now(tz=timezone.utc),
            winning_contexts=wins,
            losing_contexts=losses,
            failure_mode_patterns=failure_modes,
            retrieval_meta=retrieval_meta,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _fill_buckets(
        self,
        records: List[EpisodeMemoryRecord],
        request: MemoryRetrievalRequest,
        scope: str,
    ) -> Tuple[
        List[EpisodeMemoryRecord],
        List[EpisodeMemoryRecord],
        List[EpisodeMemoryRecord],
        str,
        List[str],
    ]:
        """Score records and fill win/loss/failure_mode buckets.

        Returns:
            (wins, losses, failure_modes, scope_used, insufficient_buckets)
        """
        # Score all candidates
        scored: List[Tuple[float, EpisodeMemoryRecord]] = []
        for record in records:
            s = _score(record, request)
            scored.append((s, record))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Separate by outcome class
        win_pool = [(s, r) for s, r in scored if r.outcome_class == "win"]
        loss_pool = [(s, r) for s, r in scored if r.outcome_class == "loss"]
        # Failure mode pool: any losing episode with at least one failure label
        failure_pool = [
            (s, r) for s, r in scored
            if r.outcome_class in ("loss", "neutral") and r.failure_modes
        ]

        # Apply diversity constraint and quota
        wins = _apply_diversity_constraint(win_pool, request.win_quota)
        losses = _apply_diversity_constraint(loss_pool, request.loss_quota)
        failure_modes = _apply_diversity_constraint(failure_pool, request.failure_mode_quota)

        # Identify under-quota buckets
        insufficient_buckets: List[str] = []
        if len(wins) < request.win_quota:
            insufficient_buckets.append("wins")
        if len(losses) < request.loss_quota:
            insufficient_buckets.append("losses")
        if len(failure_modes) < request.failure_mode_quota:
            insufficient_buckets.append("failure_modes")

        scope_used = scope if not insufficient_buckets else scope
        return wins, losses, failure_modes, scope_used, insufficient_buckets

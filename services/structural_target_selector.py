"""Structural target selector — Runbook 56.

Deterministic enforcement layer between the playbook schema (Runbook 52) and
the trigger engine / strategy primitives (Runbooks 40–42).

Responsibilities:
  1. CANDIDATE_SOURCE_REGISTRY — maps declared playbook candidate names to their
     deterministic primitive origins (Runbooks 40–42).  Versioned and testable.
  2. evaluate_expectancy_gate() — resolves all declared structural target candidates,
     filters invalids with typed reason codes, selects a target deterministically,
     computes structural R, and decides whether the expectancy gate passes.
  3. Enhanced select_stop_candidates() / select_target_candidates() / select_entry_candidates()
     from the R58 stub — kept for backward compatibility; R56 adds full telemetry.

No LLM calls.  No I/O.  All functions are pure and deterministic given the same inputs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from schemas.structure_engine import StructureLevel, StructureSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

STRUCTURAL_TARGET_SELECTOR_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Candidate source registry
# ---------------------------------------------------------------------------

# Maps the semantic candidate name declared in a PlaybookDefinition to:
#   (indicator_field_name, deterministic_origin_runbook)
#
# The indicator_field_name is the key to look up in the candidate_prices dict
# that callers populate from IndicatorSnapshot or StructureSnapshot fields.
#
# This registry is the code-level binding between playbook declarations and
# concrete Runbook 40–42 primitives.  Add entries here when new anchor
# sources are introduced.

CANDIDATE_SOURCE_REGISTRY: Dict[str, Dict[str, str]] = {
    # --- Runbook 41: HTF structure cascade fields ---
    "htf_daily_high": {
        "indicator_field": "htf_daily_high",
        "origin": "runbook_41_htf_structure_cascade",
        "direction": "resistance",  # natural role for a long target
    },
    "htf_daily_low": {
        "indicator_field": "htf_daily_low",
        "origin": "runbook_41_htf_structure_cascade",
        "direction": "support",
    },
    "htf_prev_daily_high": {
        "indicator_field": "htf_prev_daily_high",
        "origin": "runbook_41_htf_structure_cascade",
        "direction": "resistance",
    },
    "htf_prev_daily_low": {
        "indicator_field": "htf_prev_daily_low",
        "origin": "runbook_41_htf_structure_cascade",
        "direction": "support",
    },
    "htf_5d_high": {
        "indicator_field": "htf_5d_high",
        "origin": "runbook_41_htf_structure_cascade",
        "direction": "resistance",
    },
    "htf_5d_low": {
        "indicator_field": "htf_5d_low",
        "origin": "runbook_41_htf_structure_cascade",
        "direction": "support",
    },
    # --- Runbook 40: Compression/Donchian primitives ---
    "donchian_upper": {
        "indicator_field": "donchian_upper_short",
        "origin": "runbook_40_compression_breakout",
        "direction": "resistance",
    },
    "donchian_lower": {
        "indicator_field": "donchian_lower_short",
        "origin": "runbook_40_compression_breakout",
        "direction": "support",
    },
    "measured_move": {
        "indicator_field": "measured_move_projection",
        "origin": "runbook_40_compression_breakout",
        "direction": "any",  # computed from range projection
    },
    "range_projection": {
        "indicator_field": "range_projection",
        "origin": "runbook_40_compression_breakout",
        "direction": "any",
    },
    # --- Runbook 38/indicator infrastructure: Fibonacci ---
    "fib_extension": {
        "indicator_field": "fib_extension",
        "origin": "runbook_38_indicator_infrastructure",
        "direction": "any",
    },
    # --- Runbook 42: Level-anchored targets ---
    "r_multiple_2": {
        "indicator_field": "r_multiple_2",
        "origin": "runbook_42_level_anchored_stops",
        "direction": "any",
    },
    "r_multiple_3": {
        "indicator_field": "r_multiple_3",
        "origin": "runbook_42_level_anchored_stops",
        "direction": "any",
    },
}

CANDIDATE_SOURCE_REGISTRY_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Rejection reason codes
# ---------------------------------------------------------------------------

CandidateRejectionReason = Literal[
    "price_not_available",         # candidate_prices dict has no entry or entry is None
    "source_not_in_registry",      # declared source is not in CANDIDATE_SOURCE_REGISTRY
    "wrong_direction_long",        # candidate is below entry price for a long trade
    "wrong_direction_short",       # candidate is above entry price for a short trade
    "insufficient_r_multiple",     # R(candidate) < minimum_structural_r_multiple
    "price_equals_entry",          # target == entry (degenerate)
]


@dataclass
class StructuralCandidateRejection:
    """Typed rejection record for a structural target candidate."""

    source: str                      # candidate name as declared in playbook
    reason: CandidateRejectionReason
    price: Optional[float] = None    # resolved price if available, else None
    computed_r: Optional[float] = None  # R multiple if price was valid


# ---------------------------------------------------------------------------
# Expectancy gate telemetry
# ---------------------------------------------------------------------------

@dataclass
class ExpectancyGateTelemetry:
    """Full telemetry payload for one structural expectancy gate evaluation.

    Fields map 1-to-1 to the R56 required telemetry contract:
      - structural_target_source
      - all_candidate_sources_evaluated
      - selected_target_price
      - structural_r
      - candidate_rejections
      - target_selection_mode
      - expectancy_gate_passed
    """

    # Required telemetry fields (Runbook 56)
    structural_target_source: Optional[str]           # selected candidate name
    all_candidate_sources_evaluated: List[str]         # all declared candidates tried
    selected_target_price: Optional[float]             # resolved price of selected candidate
    structural_r: Optional[float]                      # (T - E) / (E - S) for longs; (E - T) / (S - E) for shorts
    candidate_rejections: List[StructuralCandidateRejection] = field(default_factory=list)
    target_selection_mode: str = "priority"            # "priority" | "ranked" | "scored"
    expectancy_gate_passed: bool = False

    # Diagnostic context
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    minimum_r_required: Optional[float] = None
    direction: Optional[str] = None
    selector_version: str = STRUCTURAL_TARGET_SELECTOR_VERSION
    registry_version: str = CANDIDATE_SOURCE_REGISTRY_VERSION


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_expectancy_gate(
    declared_candidates: List[str],
    candidate_prices: Dict[str, Optional[float]],
    entry_price: float,
    stop_price: float,
    direction: Literal["long", "short"],
    target_selection_mode: Literal["priority", "ranked", "scored"] = "priority",
    minimum_structural_r_multiple: Optional[float] = None,
) -> ExpectancyGateTelemetry:
    """Deterministic structural expectancy gate evaluation.

    Args:
        declared_candidates: Ordered list of candidate source names declared in the
            playbook's RiskRuleSet.structural_target_sources.
        candidate_prices: Dict mapping each source name to its resolved price.
            Caller populates this from IndicatorSnapshot / StructureSnapshot fields
            using CANDIDATE_SOURCE_REGISTRY to find the right indicator field.
        entry_price: Intended entry price (activation reference).
        stop_price: Structural stop price (from Runbook 42 stop anchor).
        direction: "long" or "short".
        target_selection_mode: How to pick among valid candidates.
            "priority" — use first valid candidate in declared order.
            "ranked"   — use candidate with highest R multiple.
            "scored"   — same as ranked (scoring not yet implemented; falls back to ranked).
        minimum_structural_r_multiple: If set, candidates with R < threshold are rejected.

    Returns:
        ExpectancyGateTelemetry with full audit trail.
    """
    telemetry = ExpectancyGateTelemetry(
        structural_target_source=None,
        all_candidate_sources_evaluated=list(declared_candidates),
        selected_target_price=None,
        structural_r=None,
        target_selection_mode=target_selection_mode,
        expectancy_gate_passed=False,
        entry_price=entry_price,
        stop_price=stop_price,
        minimum_r_required=minimum_structural_r_multiple,
        direction=direction,
    )

    if not declared_candidates:
        return telemetry

    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0:
        # Degenerate — cannot compute R; reject all
        for src in declared_candidates:
            telemetry.candidate_rejections.append(
                StructuralCandidateRejection(source=src, reason="price_not_available")
            )
        return telemetry

    # Resolve and validate each candidate
    valid_candidates: List[tuple[str, float, float]] = []  # (source, price, r_multiple)

    for src in declared_candidates:
        # Check registry membership
        if src not in CANDIDATE_SOURCE_REGISTRY:
            telemetry.candidate_rejections.append(
                StructuralCandidateRejection(source=src, reason="source_not_in_registry")
            )
            continue

        # Check price availability
        price = candidate_prices.get(src)
        if price is None:
            telemetry.candidate_rejections.append(
                StructuralCandidateRejection(source=src, reason="price_not_available")
            )
            continue

        # Check for degenerate price == entry
        if price == entry_price:
            telemetry.candidate_rejections.append(
                StructuralCandidateRejection(
                    source=src, reason="price_equals_entry", price=price
                )
            )
            continue

        # Directional check
        if direction == "long" and price <= entry_price:
            telemetry.candidate_rejections.append(
                StructuralCandidateRejection(
                    source=src,
                    reason="wrong_direction_long",
                    price=price,
                )
            )
            continue
        if direction == "short" and price >= entry_price:
            telemetry.candidate_rejections.append(
                StructuralCandidateRejection(
                    source=src,
                    reason="wrong_direction_short",
                    price=price,
                )
            )
            continue

        # Compute R
        if direction == "long":
            r = (price - entry_price) / stop_distance
        else:
            r = (entry_price - price) / stop_distance

        # Minimum R gate
        if minimum_structural_r_multiple is not None and r < minimum_structural_r_multiple:
            telemetry.candidate_rejections.append(
                StructuralCandidateRejection(
                    source=src,
                    reason="insufficient_r_multiple",
                    price=price,
                    computed_r=r,
                )
            )
            continue

        valid_candidates.append((src, price, r))

    if not valid_candidates:
        return telemetry  # gate not passed, no valid candidate

    # Select target according to target_selection_mode
    if target_selection_mode == "priority":
        # First valid candidate in declared order (valid_candidates preserves order)
        selected_src, selected_price, selected_r = valid_candidates[0]
    elif target_selection_mode in ("ranked", "scored"):
        # Highest R multiple wins
        selected_src, selected_price, selected_r = max(valid_candidates, key=lambda x: x[2])
    else:
        # Unknown mode — fall back to priority
        logger.warning("Unknown target_selection_mode '%s'; falling back to priority", target_selection_mode)
        selected_src, selected_price, selected_r = valid_candidates[0]

    telemetry.structural_target_source = selected_src
    telemetry.selected_target_price = selected_price
    telemetry.structural_r = selected_r
    telemetry.expectancy_gate_passed = True
    return telemetry


# ---------------------------------------------------------------------------
# Convenience: resolve candidate prices from an IndicatorSnapshot-like dict
# ---------------------------------------------------------------------------


def resolve_candidate_prices_from_indicator(
    declared_candidates: List[str],
    indicator_data: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """Map declared candidate names to prices using CANDIDATE_SOURCE_REGISTRY.

    Args:
        declared_candidates: Source names as declared in the playbook.
        indicator_data: Dict-like object (or IndicatorSnapshot.model_dump()) with
            numeric field values.

    Returns:
        Dict[source_name -> price_or_None].
    """
    result: Dict[str, Optional[float]] = {}
    for src in declared_candidates:
        registry_entry = CANDIDATE_SOURCE_REGISTRY.get(src)
        if registry_entry is None:
            result[src] = None
            continue
        field_name = registry_entry["indicator_field"]
        value = indicator_data.get(field_name)
        result[src] = float(value) if value is not None else None
    return result


# ---------------------------------------------------------------------------
# R58 stub API — preserved for backward compatibility, enhanced with telemetry
# ---------------------------------------------------------------------------


def select_stop_candidates(
    snapshot: StructureSnapshot,
    direction: Optional[str] = None,
    max_distance_atr: Optional[float] = None,
) -> List[StructureLevel]:
    """Return ordered stop-anchor candidates from a StructureSnapshot.

    Direction-aware: longs use support levels, shorts use resistance levels.
    """
    candidates = [lvl for lvl in snapshot.levels if lvl.eligible_for_stop_anchor]

    if direction == "long":
        candidates = [lvl for lvl in candidates if lvl.role_now == "support"]
    elif direction == "short":
        candidates = [lvl for lvl in candidates if lvl.role_now == "resistance"]

    if max_distance_atr is not None:
        candidates = [
            lvl for lvl in candidates
            if lvl.distance_atr is None or lvl.distance_atr <= max_distance_atr
        ]

    return sorted(candidates, key=lambda lvl: lvl.distance_abs)


def select_target_candidates(
    snapshot: StructureSnapshot,
    direction: Optional[str] = None,
    max_distance_atr: Optional[float] = None,
) -> List[StructureLevel]:
    """Return ordered target-anchor candidates from a StructureSnapshot.

    Direction-aware: longs seek resistance targets above, shorts seek support targets below.
    """
    candidates = [lvl for lvl in snapshot.levels if lvl.eligible_for_target_anchor]

    if direction == "long":
        candidates = [lvl for lvl in candidates if lvl.role_now == "resistance"]
    elif direction == "short":
        candidates = [lvl for lvl in candidates if lvl.role_now == "support"]

    if max_distance_atr is not None:
        candidates = [
            lvl for lvl in candidates
            if lvl.distance_atr is None or lvl.distance_atr <= max_distance_atr
        ]

    return sorted(candidates, key=lambda lvl: lvl.distance_abs)


def select_entry_candidates(
    snapshot: StructureSnapshot,
    direction: Optional[str] = None,
) -> List[StructureLevel]:
    """Return ordered entry-activation candidates from a StructureSnapshot."""
    candidates = [lvl for lvl in snapshot.levels if lvl.eligible_for_entry_trigger]

    if direction == "long":
        candidates = [lvl for lvl in candidates if lvl.role_now == "support"]
    elif direction == "short":
        candidates = [lvl for lvl in candidates if lvl.role_now == "resistance"]

    return sorted(candidates, key=lambda lvl: lvl.distance_abs)

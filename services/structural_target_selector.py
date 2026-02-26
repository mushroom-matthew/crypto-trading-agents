"""Structural target selector — Runbook 56 forward-compatibility stub.

Runbook 56 (docs-only) will implement the full deterministic enforcement layer
between playbook schema and trigger engine:
  - expectancy gate telemetry
  - activation refinement mode → trigger identifier / timeframe / confirmation mapping
  - compiler validation for structural target candidate selection

This stub exposes the selection API surface so callers can be written against
the final interface now and upgraded when R56 lands.

Runbook 58 integration:
  The structure engine populates eligible_for_target_anchor / eligible_for_stop_anchor
  / eligible_for_entry_trigger flags on StructureLevel.  This selector consumes
  those flags and applies ordering rules.

R56 will add:
  - expectancy gate filtering (minimum expectancy_r threshold)
  - refinement mode mapping (activation_refinement_mode → confirmation_rule)
  - compiler-visible rejection telemetry (StructuralCandidateRejection)
"""
from __future__ import annotations

import logging
from typing import List, Optional

from schemas.structure_engine import StructureLevel, StructureSnapshot

logger = logging.getLogger(__name__)


class StructuralCandidateRejection:
    """Placeholder — R56 will define a full typed rejection record."""

    def __init__(self, level_id: str, reason: str) -> None:
        self.level_id = level_id
        self.reason = reason

    def __repr__(self) -> str:
        return f"StructuralCandidateRejection(level_id={self.level_id!r}, reason={self.reason!r})"


def select_stop_candidates(
    snapshot: StructureSnapshot,
    direction: Optional[str] = None,
    max_distance_atr: Optional[float] = None,
) -> List[StructureLevel]:
    """Return ordered stop-anchor candidates from a StructureSnapshot.

    R56 will add:
      - expectancy gate filtering
      - direction-aware filtering (longs use support levels; shorts use resistance)
      - compiler-visible rejection telemetry

    Current behavior (pre-R56):
      Returns all eligible_for_stop_anchor levels, sorted by proximity.
      When direction is "long", only returns support levels.
      When direction is "short", only returns resistance levels.
      When max_distance_atr is set, filters levels beyond the threshold.
    """
    candidates = [l for l in snapshot.levels if l.eligible_for_stop_anchor]

    if direction == "long":
        candidates = [l for l in candidates if l.role_now == "support"]
    elif direction == "short":
        candidates = [l for l in candidates if l.role_now == "resistance"]

    if max_distance_atr is not None:
        candidates = [
            l for l in candidates
            if l.distance_atr is None or l.distance_atr <= max_distance_atr
        ]

    return sorted(candidates, key=lambda l: l.distance_abs)


def select_target_candidates(
    snapshot: StructureSnapshot,
    direction: Optional[str] = None,
    max_distance_atr: Optional[float] = None,
) -> List[StructureLevel]:
    """Return ordered target-anchor candidates from a StructureSnapshot.

    R56 will add:
      - expectancy gate telemetry (expected R-multiple per candidate)
      - refinement mode mapping

    Current behavior (pre-R56):
      Returns all eligible_for_target_anchor levels, sorted by proximity.
      When direction is "long", only returns resistance levels (targets above).
      When direction is "short", only returns support levels (targets below).
    """
    candidates = [l for l in snapshot.levels if l.eligible_for_target_anchor]

    if direction == "long":
        candidates = [l for l in candidates if l.role_now == "resistance"]
    elif direction == "short":
        candidates = [l for l in candidates if l.role_now == "support"]

    if max_distance_atr is not None:
        candidates = [
            l for l in candidates
            if l.distance_atr is None or l.distance_atr <= max_distance_atr
        ]

    return sorted(candidates, key=lambda l: l.distance_abs)


def select_entry_candidates(
    snapshot: StructureSnapshot,
    direction: Optional[str] = None,
) -> List[StructureLevel]:
    """Return ordered entry-activation candidates from a StructureSnapshot.

    R52 (playbook schema) and R56 (activation refinement) will add:
      - playbook eligibility filtering
      - activation refinement mode → confirmation_rule mapping

    Current behavior (pre-R52/R56):
      Returns all eligible_for_entry_trigger levels, sorted by proximity.
    """
    candidates = [l for l in snapshot.levels if l.eligible_for_entry_trigger]

    if direction == "long":
        candidates = [l for l in candidates if l.role_now == "support"]
    elif direction == "short":
        candidates = [l for l in candidates if l.role_now == "resistance"]

    return sorted(candidates, key=lambda l: l.distance_abs)

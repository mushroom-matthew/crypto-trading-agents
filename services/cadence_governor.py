"""CadenceGovernor — tracks round-trip completion and adapts symbol breadth (R77).

The governor monitors how many hypothesis round trips have completed in the session
and projects whether the session is on pace to hit the 5–10 trades/24h target from
SessionIntent. When below target, it suggests widening symbol breadth using the last
opportunity ranking. When above target, it suggests throttling.

Hard invariant: NEVER lowers quality gates (min_rr, stop/target requirements) to
meet cadence targets. Only permitted adaptations:
- Increase symbol breadth (add more from scanner ranking)
- Reallocate risk budgets (smaller per trade, more symbols)
- Reduce opportunity_score cutoff threshold (within quality floor)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Minimum quality floor constants — enforced unconditionally
# ---------------------------------------------------------------------------
_MIN_RR_FLOOR = float(os.environ.get("PAPER_TRADING_MIN_RR_RATIO", "1.2"))
_WIDEN_MIN_SCORE = float(os.environ.get("CADENCE_GOVERNOR_MIN_SCORE", "0.25"))
_WIDEN_COOLDOWN_HOURS = float(os.environ.get("CADENCE_GOVERNOR_WIDEN_COOLDOWN_H", "2.0"))
_THROTTLE_COOLDOWN_HOURS = float(os.environ.get("CADENCE_GOVERNOR_THROTTLE_COOLDOWN_H", "1.0"))
_BELOW_TARGET_WINDOW_HOURS = float(os.environ.get("CADENCE_GOVERNOR_BELOW_TARGET_WINDOW_H", "4.0"))


class RoundTripRecord(BaseModel):
    """A single completed round trip (entry + exit)."""

    model_config = {"extra": "forbid"}

    symbol: str
    outcome: Literal["win", "loss", "neutral"]
    completed_at: datetime
    r_achieved: Optional[float] = None
    playbook_id: Optional[str] = None


class CadenceGovernorState(BaseModel):
    """Serializable state for the CadenceGovernor (persisted in SessionState)."""

    model_config = {"extra": "forbid"}

    round_trips_completed: int = 0
    round_trips_target_min: int = 5
    round_trips_target_max: int = 10
    session_start_iso: Optional[str] = None  # ISO datetime string
    last_completed_at_iso: Optional[str] = None
    last_adaptation_at_iso: Optional[str] = None
    last_adaptation_action: Optional[str] = None
    last_widen_at_iso: Optional[str] = None
    last_throttle_at_iso: Optional[str] = None
    cadence_status: Literal["on_track", "below_target", "above_target", "not_started"] = "not_started"
    projected_24h_rate: float = 0.0
    hours_elapsed: float = 0.0
    round_trip_history: list = Field(default_factory=list)  # List[Dict] — RoundTripRecord dicts
    # Symbol breadth adjustments made by the governor
    added_symbols: list = Field(default_factory=list)  # List[str]


class CadenceGovernor:
    """Tracks round-trip completion and suggests breadth adjustments.

    Hard invariant: NEVER lowers quality gates (min_rr_ratio, stop/target
    requirements) to meet cadence targets.
    """

    def __init__(self, state: Optional[CadenceGovernorState] = None) -> None:
        self._state = state or CadenceGovernorState()

    @property
    def state(self) -> CadenceGovernorState:
        return self._state

    def initialize(
        self,
        session_start: datetime,
        target_min: int = 5,
        target_max: int = 10,
    ) -> None:
        """Set session start time and cadence targets."""
        if self._state.session_start_iso is None:
            self._state = self._state.model_copy(update={
                "session_start_iso": session_start.isoformat(),
                "round_trips_target_min": target_min,
                "round_trips_target_max": target_max,
                "cadence_status": "not_started",
            })

    def record_round_trip_complete(
        self,
        symbol: str,
        outcome: Literal["win", "loss", "neutral"],
        r_achieved: Optional[float] = None,
        playbook_id: Optional[str] = None,
    ) -> None:
        """Record a completed round trip (entry + exit for a hypothesis)."""
        now = datetime.now(timezone.utc)
        record = RoundTripRecord(
            symbol=symbol,
            outcome=outcome,
            completed_at=now,
            r_achieved=r_achieved,
            playbook_id=playbook_id,
        )
        history = list(self._state.round_trip_history) + [record.model_dump(mode="json")]
        # Keep only last 100 records
        history = history[-100:]

        self._state = self._state.model_copy(update={
            "round_trips_completed": self._state.round_trips_completed + 1,
            "last_completed_at_iso": now.isoformat(),
            "round_trip_history": history,
        })
        self._refresh_projection(now)

    def _refresh_projection(self, now: Optional[datetime] = None) -> None:
        """Recompute projected_24h_rate, hours_elapsed, and cadence_status."""
        if now is None:
            now = datetime.now(timezone.utc)

        if not self._state.session_start_iso:
            return

        start = datetime.fromisoformat(self._state.session_start_iso)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)

        hours_elapsed = max(0.001, (now - start).total_seconds() / 3600)
        completed = self._state.round_trips_completed
        projected = (completed / hours_elapsed) * 24.0 if hours_elapsed > 0 else 0.0

        target_mid = (self._state.round_trips_target_min + self._state.round_trips_target_max) / 2.0
        if projected < self._state.round_trips_target_min:
            status: Literal["on_track", "below_target", "above_target", "not_started"] = "below_target"
        elif projected > self._state.round_trips_target_max:
            status = "above_target"
        else:
            status = "on_track"

        self._state = self._state.model_copy(update={
            "hours_elapsed": round(hours_elapsed, 3),
            "projected_24h_rate": round(projected, 2),
            "cadence_status": status,
        })

    def should_widen_symbol_set(self, now: Optional[datetime] = None) -> bool:
        """True when below target, session has enough time remaining, and cooldown passed."""
        if now is None:
            now = datetime.now(timezone.utc)

        self._refresh_projection(now)

        if self._state.cadence_status != "below_target":
            return False

        # Only widen after sufficient elapsed time (avoid false positives at session start)
        if self._state.hours_elapsed < 1.0:
            return False

        # Enforce widen cooldown to prevent oscillation
        if self._state.last_widen_at_iso:
            last_widen = datetime.fromisoformat(self._state.last_widen_at_iso)
            if last_widen.tzinfo is None:
                last_widen = last_widen.replace(tzinfo=timezone.utc)
            hours_since_widen = (now - last_widen).total_seconds() / 3600
            if hours_since_widen < _WIDEN_COOLDOWN_HOURS:
                return False

        return True

    def get_adaptation_recommendation(
        self,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Return adaptation recommendation for the next plan cycle.

        Returns:
            dict with keys:
            - action: "widen_breadth" | "throttle" | "hold"
            - reason: human-readable explanation
            - quality_floor_min_rr: the min_rr floor (never reduced below this)
        """
        if now is None:
            now = datetime.now(timezone.utc)

        self._refresh_projection(now)
        status = self._state.cadence_status

        # Enforce: min_rr floor is never touched
        result: Dict[str, Any] = {
            "action": "hold",
            "reason": "on pace",
            "quality_floor_min_rr": _MIN_RR_FLOOR,
            "cadence_status": status,
            "projected_24h_rate": self._state.projected_24h_rate,
            "round_trips_completed": self._state.round_trips_completed,
        }

        if status == "below_target" and self.should_widen_symbol_set(now):
            result["action"] = "widen_breadth"
            result["reason"] = (
                f"Pace={self._state.projected_24h_rate:.1f}/24h below "
                f"target {self._state.round_trips_target_min}–{self._state.round_trips_target_max}/24h "
                f"after {self._state.hours_elapsed:.1f}h. Add symbols from ranking."
            )
            # Record adaptation
            self._state = self._state.model_copy(update={
                "last_widen_at_iso": now.isoformat(),
                "last_adaptation_at_iso": now.isoformat(),
                "last_adaptation_action": "widen_breadth",
            })

        elif status == "above_target":
            # Check throttle cooldown
            should_throttle = True
            if self._state.last_throttle_at_iso:
                last_throttle = datetime.fromisoformat(self._state.last_throttle_at_iso)
                if last_throttle.tzinfo is None:
                    last_throttle = last_throttle.replace(tzinfo=timezone.utc)
                hours_since = (now - last_throttle).total_seconds() / 3600
                if hours_since < _THROTTLE_COOLDOWN_HOURS:
                    should_throttle = False

            if should_throttle:
                result["action"] = "throttle"
                result["reason"] = (
                    f"Pace={self._state.projected_24h_rate:.1f}/24h above "
                    f"target max {self._state.round_trips_target_max}/24h. "
                    f"Reduce risk budget fraction."
                )
                self._state = self._state.model_copy(update={
                    "last_throttle_at_iso": now.isoformat(),
                    "last_adaptation_at_iso": now.isoformat(),
                    "last_adaptation_action": "throttle",
                })

        return result

    def mark_symbol_added(self, symbol: str) -> None:
        """Record that the governor added a symbol to the session."""
        added = list(self._state.added_symbols)
        if symbol not in added:
            added.append(symbol)
        self._state = self._state.model_copy(update={"added_symbols": added})

    def get_summary(self) -> Dict[str, Any]:
        """Return a lightweight summary dict for session queries and logging."""
        return {
            "round_trips_completed": self._state.round_trips_completed,
            "round_trips_target_min": self._state.round_trips_target_min,
            "round_trips_target_max": self._state.round_trips_target_max,
            "cadence_status": self._state.cadence_status,
            "projected_24h_rate": self._state.projected_24h_rate,
            "hours_elapsed": self._state.hours_elapsed,
            "last_adaptation_action": self._state.last_adaptation_action,
            "added_symbols": list(self._state.added_symbols),
        }

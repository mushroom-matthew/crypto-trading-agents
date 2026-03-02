"""Adaptive trade management state machine (Runbook 45 / R63).

Tracks an open position through four R-multiple phases:
  EARLY    — position just opened, R-multiple < 0.5
  MATURE   — R-multiple >= 0.5 (first milestone reached)
  EXTENDED — R-multiple >= 1.0 (full 1R exceeded)
  TRAIL    — R-multiple >= 1.5 (trailing-stop territory)

Phase transitions are monotonic — once a phase is entered, the position
never regresses to an earlier phase (peak_r_achieved drives state).

Usage (per-bar in the paper trading loop):
    from services.adaptive_trade_management import AdaptiveTradeManagementState

    state_dict = session_state.adaptive_management_states.get(symbol, {})
    state = (
        AdaptiveTradeManagementState.model_validate(state_dict)
        if state_dict
        else AdaptiveTradeManagementState.initial(position_meta)
    )
    state = state.tick(current_price=current_price)
    session_state.adaptive_management_states[symbol] = state.model_dump()
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

# R-multiple thresholds for phase transitions
_R_MATURE = 0.5
_R_EXTENDED = 1.0
_R_TRAIL = 1.5

TradingPhase = Literal["EARLY", "MATURE", "EXTENDED", "TRAIL"]


class AdaptiveTradeManagementState(BaseModel):
    """Serializable per-position state for adaptive trade management.

    Stored in SessionState.adaptive_management_states keyed by symbol.
    Updated each bar via tick().  Transitions are strictly monotonic
    (EARLY → MATURE → EXTENDED → TRAIL).
    """

    model_config = {"extra": "forbid"}

    symbol: str
    direction: Literal["long", "short"] = "long"
    entry_price: float
    stop_price_abs: Optional[float] = None
    target_price_abs: Optional[float] = None
    phase: TradingPhase = "EARLY"
    bars_held: int = 0
    peak_r_achieved: float = 0.0
    last_price: Optional[float] = None

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def initial(cls, position_meta: Dict[str, Any]) -> "AdaptiveTradeManagementState":
        """Create an initial EARLY-phase state from a position_meta dict.

        Args:
            position_meta: Dict from portfolio_state["position_meta"][symbol].
                Expected keys: entry_side, signal_entry_price, stop_price_abs,
                target_price_abs, symbol.
        """
        direction: str = position_meta.get("entry_side", "long")
        entry_price = float(
            position_meta.get("signal_entry_price")
            or position_meta.get("entry_price")
            or 0.0
        )
        stop_raw = position_meta.get("stop_price_abs")
        target_raw = position_meta.get("target_price_abs")
        return cls(
            symbol=str(position_meta.get("symbol", "")),
            direction=direction,
            entry_price=entry_price,
            stop_price_abs=float(stop_raw) if stop_raw is not None else None,
            target_price_abs=float(target_raw) if target_raw is not None else None,
            phase="EARLY",
            bars_held=0,
            peak_r_achieved=0.0,
        )

    # -------------------------------------------------------------------------
    # Per-bar update
    # -------------------------------------------------------------------------

    def tick(
        self,
        current_price: float,
        indicator: Any = None,
    ) -> "AdaptiveTradeManagementState":
        """Advance state by one bar given the current closing price.

        Computes current R-multiple and transitions phase monotonically.
        Returns a new (immutable) state object; does not mutate self.

        Args:
            current_price: Latest closing price for the symbol.
            indicator: Optional IndicatorSnapshot (unused; reserved for future
                       ATR-based stop adjustments).
        """
        # Compute current R-multiple relative to initial entry/stop
        r_current: float = 0.0
        if self.stop_price_abs is not None and self.entry_price > 0:
            risk = abs(self.entry_price - self.stop_price_abs)
            if risk > 0:
                if self.direction == "long":
                    r_current = (current_price - self.entry_price) / risk
                else:
                    r_current = (self.entry_price - current_price) / risk

        peak_r = max(self.peak_r_achieved, r_current)

        # Monotonic phase transition
        if peak_r >= _R_TRAIL:
            phase: TradingPhase = "TRAIL"
        elif peak_r >= _R_EXTENDED:
            phase = "EXTENDED"
        elif peak_r >= _R_MATURE:
            phase = "MATURE"
        else:
            phase = "EARLY"

        return self.model_copy(update={
            "phase": phase,
            "bars_held": self.bars_held + 1,
            "peak_r_achieved": round(peak_r, 4),
            "last_price": current_price,
        })

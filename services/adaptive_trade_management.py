"""Adaptive trade management state machine (Runbook 45 / R63 / R85).

Tracks an open position through four R-multiple phases:
  EARLY    — position just opened, R-multiple < 0.5
  MATURE   — R-multiple >= 0.5 (first milestone reached)
  EXTENDED — R-multiple >= 1.0 (full 1R exceeded)
  TRAIL    — R-multiple >= 1.5 (trailing-stop territory)

Phase transitions are monotonic — once a phase is entered, the position
never regresses to an earlier phase (peak_r_achieved drives state).

R85 additions:
  - trailing_config: TrailingStopConfig drives stop movement each bar
  - active_stop_price: dynamic stop (starts at entry stop, moves only in position's favor)
  - hwm_price: highest (long) or lowest (short) mark seen since entry
  - consecutive_closes_below_stop: close-confirmation counter for wick protection
  - stop_fired: True for the bar in which close confirmation threshold is met
  - partial_fires_pending: R milestones that should fire partial exits this bar

Usage (per-bar in the paper trading loop):
    from services.adaptive_trade_management import AdaptiveTradeManagementState
    from schemas.trailing_stop import TrailingStopConfig

    state_dict = session_state.adaptive_management_states.get(symbol, {})
    state = (
        AdaptiveTradeManagementState.model_validate(state_dict)
        if state_dict
        else AdaptiveTradeManagementState.initial(position_meta, trailing_config)
    )
    state = state.tick(current_price=current_price, atr=indicator.atr_14)
    session_state.adaptive_management_states[symbol] = state.model_dump()

    # Consume outputs:
    active_stop   = state.active_stop_price   # use this as the live stop level
    stop_fired    = state.stop_fired           # True = close confirmation met, exit now
    for r_target in state.partial_fires_pending:
        # trigger partial exit at r_target milestone
        ...
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from schemas.trailing_stop import PartialExitSpec, TrailingStopConfig

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

    R85: active_stop_price is the live stop level that moves under trailing rules.
    The trigger engine should reference active_stop_price (not the static position_meta
    stop_price_abs) once this state is initialised.
    """

    model_config = {"extra": "forbid"}

    symbol: str
    direction: Literal["long", "short"] = "long"
    entry_price: float
    initial_stop_price: Optional[float] = None   # stop at entry (never changes)
    active_stop_price: Optional[float] = None    # R85: current dynamic stop (moves)
    target_price_abs: Optional[float] = None
    phase: TradingPhase = "EARLY"
    bars_held: int = 0
    peak_r_achieved: float = 0.0
    hwm_price: Optional[float] = None            # R85: HWM for trailing calculation
    last_price: Optional[float] = None

    # Trailing config (serialised alongside state)
    trailing_config: TrailingStopConfig = Field(default_factory=TrailingStopConfig)

    # Milestones already fired (prevents re-firing)
    breakeven_ratchet_fired: bool = False
    partial_milestones_fired: List[float] = Field(default_factory=list)

    # Close-confirmation counter
    consecutive_closes_below_stop: int = 0

    # Per-bar outputs (reset each tick; read by caller immediately after tick())
    stop_fired: bool = False            # True = close-confirmation met → exit now
    partial_fires_pending: List[float] = Field(default_factory=list)  # R milestones to execute

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def initial(
        cls,
        position_meta: Dict[str, Any],
        trailing_config: Optional[TrailingStopConfig] = None,
    ) -> "AdaptiveTradeManagementState":
        """Create an initial EARLY-phase state from a position_meta dict.

        Args:
            position_meta: Dict from portfolio_state["position_meta"][symbol].
                Expected keys: entry_side, signal_entry_price, stop_price_abs,
                target_price_abs, symbol.
            trailing_config: Trailing stop configuration. Defaults to no trailing.
        """
        direction: str = position_meta.get("entry_side", "long")
        entry_price = float(
            position_meta.get("signal_entry_price")
            or position_meta.get("entry_price")
            or 0.0
        )
        stop_raw = position_meta.get("stop_price_abs")
        target_raw = position_meta.get("target_price_abs")
        stop_price = float(stop_raw) if stop_raw is not None else None

        return cls(
            symbol=str(position_meta.get("symbol", "")),
            direction=direction,
            entry_price=entry_price,
            initial_stop_price=stop_price,
            active_stop_price=stop_price,
            target_price_abs=float(target_raw) if target_raw is not None else None,
            phase="EARLY",
            bars_held=0,
            peak_r_achieved=0.0,
            hwm_price=entry_price if entry_price else None,
            trailing_config=trailing_config or TrailingStopConfig(),
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _compute_r(self, price: float) -> float:
        """Compute current R-multiple at `price` relative to initial entry/stop."""
        if self.initial_stop_price is None or self.entry_price <= 0:
            return 0.0
        risk = abs(self.entry_price - self.initial_stop_price)
        if risk <= 0:
            return 0.0
        if self.direction == "long":
            return (price - self.entry_price) / risk
        else:
            return (self.entry_price - price) / risk

    def _r_to_price(self, r_multiple: float) -> float:
        """Convert an R multiple back to an absolute price level."""
        if self.initial_stop_price is None or self.entry_price <= 0:
            return self.entry_price
        risk = abs(self.entry_price - self.initial_stop_price)
        if self.direction == "long":
            return self.entry_price + r_multiple * risk
        else:
            return self.entry_price - r_multiple * risk

    def _tighten_stop(self, proposed: float) -> Optional[float]:
        """Return proposed stop only if it improves on active_stop (tighten-only).

        For longs: new stop must be >= active (move up only).
        For shorts: new stop must be <= active (move down only).
        Returns None if proposed would widen the stop.
        """
        if self.active_stop_price is None:
            return proposed
        if self.direction == "long":
            return proposed if proposed > self.active_stop_price else None
        else:
            return proposed if proposed < self.active_stop_price else None

    # -------------------------------------------------------------------------
    # Per-bar update
    # -------------------------------------------------------------------------

    def tick(
        self,
        current_price: float,
        atr: Optional[float] = None,
        indicator: Any = None,  # IndicatorSnapshot — kept for forward-compat
    ) -> "AdaptiveTradeManagementState":
        """Advance state by one bar given the current closing price.

        Computes current R-multiple, transitions phase monotonically, applies
        trailing stop logic, and evaluates close-confirmation and partial exits.

        Returns a new (immutable) state object; does not mutate self.

        Args:
            current_price: Latest closing price for the symbol.
            atr: ATR(14) for the current bar. Required for atr_trail mode.
            indicator: Optional IndicatorSnapshot (fallback ATR source when atr not passed).
        """
        cfg = self.trailing_config

        # Resolve ATR from indicator if not passed directly
        if atr is None and indicator is not None:
            atr = getattr(indicator, "atr_14", None)

        # ── 1. Update HWM price ───────────────────────────────────────────────
        if self.hwm_price is None:
            hwm = current_price
        elif self.direction == "long":
            hwm = max(self.hwm_price, current_price)
        else:
            hwm = min(self.hwm_price, current_price)

        # ── 2. Compute current and peak R ────────────────────────────────────
        r_current = self._compute_r(current_price)
        peak_r = max(self.peak_r_achieved, r_current)

        # ── 3. Monotonic phase transition ────────────────────────────────────
        if peak_r >= _R_TRAIL:
            phase: TradingPhase = "TRAIL"
        elif peak_r >= _R_EXTENDED:
            phase = "EXTENDED"
        elif peak_r >= _R_MATURE:
            phase = "MATURE"
        else:
            phase = "EARLY"

        # ── 4. Stop movement ─────────────────────────────────────────────────
        active_stop = self.active_stop_price
        breakeven_fired = self.breakeven_ratchet_fired

        # Step 4a: breakeven ratchet (applies to all modes with breakeven_at_r set)
        if (
            not breakeven_fired
            and cfg.breakeven_at_r is not None
            and peak_r >= cfg.breakeven_at_r
            and self.entry_price > 0
        ):
            proposed = self._tighten_stop(self.entry_price)
            if proposed is not None:
                active_stop = proposed
                breakeven_fired = True

        # Step 4b: trailing logic (mode-specific, activated at trail_activation_r)
        trail_active = peak_r >= cfg.trail_activation_r

        if trail_active:
            if cfg.mode == "atr_trail" and atr and atr > 0:
                if self.direction == "long":
                    proposed_trail = hwm - cfg.atr_trail_multiple * atr
                else:
                    proposed_trail = hwm + cfg.atr_trail_multiple * atr
                tightened = self._tighten_stop(proposed_trail) if active_stop is None else (
                    proposed_trail if (
                        (self.direction == "long" and proposed_trail > active_stop) or
                        (self.direction == "short" and proposed_trail < active_stop)
                    ) else None
                )
                if tightened is not None:
                    active_stop = tightened

            elif cfg.mode == "pct_trail" and hwm and hwm > 0:
                factor = cfg.pct_trail_distance / 100.0
                if self.direction == "long":
                    proposed_trail = hwm * (1.0 - factor)
                else:
                    proposed_trail = hwm * (1.0 + factor)
                tightened = self._tighten_stop(proposed_trail) if active_stop is None else (
                    proposed_trail if (
                        (self.direction == "long" and proposed_trail > active_stop) or
                        (self.direction == "short" and proposed_trail < active_stop)
                    ) else None
                )
                if tightened is not None:
                    active_stop = tightened

            elif cfg.mode == "step_trail":
                # Step the stop to the price level corresponding to the most recent
                # lower R milestone (e.g., when at 2.5R → stop at 1R price level).
                # Milestones: entry (0R), 1R, 2R, 3R, 4R, 5R
                for step_r in [5.0, 4.0, 3.0, 2.0, 1.0]:
                    if peak_r >= step_r + 1.0:
                        proposed_step = self._r_to_price(step_r)
                        tightened = self._tighten_stop(proposed_step) if active_stop is None else (
                            proposed_step if (
                                (self.direction == "long" and proposed_step > active_stop) or
                                (self.direction == "short" and proposed_step < active_stop)
                            ) else None
                        )
                        if tightened is not None:
                            active_stop = tightened
                        break  # apply the highest applicable step only

        # ── 5. Close confirmation ────────────────────────────────────────────
        consec = self.consecutive_closes_below_stop
        stop_fired_this_bar = False

        if active_stop is not None:
            breached = (
                (self.direction == "long" and current_price < active_stop) or
                (self.direction == "short" and current_price > active_stop)
            )
            if breached:
                consec += 1
                if consec >= cfg.close_required_bars:
                    stop_fired_this_bar = True
                    consec = 0  # reset (position will be closed by caller)
            else:
                consec = 0  # price recovered above stop; reset counter

        # ── 6. Partial exit milestones ───────────────────────────────────────
        already_fired = set(self.partial_milestones_fired)
        new_fires: List[float] = []
        new_milestones_fired = list(self.partial_milestones_fired)

        for spec in sorted(cfg.partial_exits, key=lambda s: s.at_r_multiple):
            if spec.at_r_multiple not in already_fired and peak_r >= spec.at_r_multiple:
                new_fires.append(spec.at_r_multiple)
                new_milestones_fired.append(spec.at_r_multiple)
                # Apply any accompanying breakeven ratchet from partial spec
                if spec.move_stop_to_entry and not breakeven_fired and self.entry_price > 0:
                    proposed_be = self._tighten_stop(self.entry_price)
                    if proposed_be is not None:
                        active_stop = proposed_be
                    breakeven_fired = True

        return self.model_copy(update={
            "phase": phase,
            "bars_held": self.bars_held + 1,
            "peak_r_achieved": round(peak_r, 4),
            "hwm_price": hwm,
            "last_price": current_price,
            "active_stop_price": active_stop,
            "breakeven_ratchet_fired": breakeven_fired,
            "partial_milestones_fired": new_milestones_fired,
            "consecutive_closes_below_stop": consec,
            "stop_fired": stop_fired_this_bar,
            "partial_fires_pending": new_fires,
        })

    # -------------------------------------------------------------------------
    # Convenience read
    # -------------------------------------------------------------------------

    def stop_distance_r(self) -> float:
        """Distance between current price and active stop, in R units."""
        if self.active_stop_price is None or self.last_price is None:
            return 0.0
        return abs(self._compute_r(self.active_stop_price))

    def describe(self) -> str:
        """Human-readable summary for logging."""
        mode = self.trailing_config.mode
        stop = f"{self.active_stop_price:.4f}" if self.active_stop_price else "none"
        return (
            f"{self.symbol} {self.direction.upper()} | phase={self.phase} "
            f"R={self.peak_r_achieved:.2f} | stop={stop} (mode={mode}) "
            f"bars={self.bars_held}"
        )

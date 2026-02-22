"""Trade set and leg schemas for position lifecycle accounting.

This module implements the TradeSet model which represents a complete position
lifecycle from flat -> non-flat -> ... -> flat. Each fill within a lifecycle
is represented as a TradeLeg.

Key concepts:
- TradeSet: Groups all legs from position open to close (1:N entry/exit ratio)
- TradeLeg: Individual fill with unique ID, side, qty, price, fees
- WAC: Weighted Average Cost accounting for per-leg and per-set P&L

This replaces the 1:1 PairedTrade model to support partial exits and scale-ins.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, model_validator


# ---------------------------------------------------------------------------
# Runbook 45 — Trade management audit events
# ---------------------------------------------------------------------------

class StopAdjustmentEvent(BaseModel):
    """Audit event emitted whenever the adaptive trade manager advances a stop level."""

    model_config = {"extra": "forbid"}

    symbol: str
    timestamp: datetime
    rung: str                    # "r1_mature" | "r2_extended" | "r3_trail" | "atr_trail"
    old_stop: Optional[float]
    new_stop: float
    current_R: float
    mfe_r: float
    mae_r: float
    position_fraction: float     # Fraction of original qty still open
    rung_catch: bool = False     # True if this was a jump-catch (multi-rung bar)
    engine_version: str = "45.0.0"


class PartialExitEvent(BaseModel):
    """Audit event emitted when the adaptive trade manager schedules a partial exit."""

    model_config = {"extra": "forbid"}

    symbol: str
    timestamp: datetime
    rung: str                    # "r2_extended" (currently only R2 exits)
    fraction_exited: float
    exit_price: float
    exit_R: float                # R at time of partial exit
    mfe_r: float
    initial_risk_abs: float
    position_fraction_before: float
    position_fraction_after: float
    exit_blocked: bool = False   # True if hold constraint prevented execution
    exit_blocked_by: Optional[str] = None
    engine_version: str = "45.0.0"


class TradeLeg(BaseModel):
    """Individual fill within a position lifecycle.

    Each leg represents a single execution (buy or sell) with a unique ID.
    Legs are grouped into TradeSets based on position lifecycle.
    """

    model_config = {"extra": "forbid"}

    leg_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this leg (uuid, not timestamp-based)",
    )
    side: Literal["buy", "sell"] = Field(description="Trade direction")
    qty: float = Field(gt=0, description="Executed quantity (always positive)")
    price: float = Field(gt=0, description="Execution price")
    fees: float = Field(default=0.0, ge=0, description="Transaction fees")
    timestamp: datetime = Field(description="Execution timestamp")
    trigger_id: Optional[str] = Field(
        default=None, description="Trigger that generated this leg"
    )
    category: Optional[str] = Field(
        default=None,
        description="Trigger category (e.g., momentum_long, risk_reduce, emergency_exit)",
    )
    reason: Optional[str] = Field(
        default=None, description="Human-readable reason for trade"
    )
    is_entry: bool = Field(
        description="True if this leg opened or added to the position"
    )
    exit_fraction: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of position closed (for risk_reduce exits)",
    )
    # WAC accounting fields (computed at ingestion time)
    wac_at_fill: Optional[float] = Field(
        default=None,
        description="Weighted average cost basis at time of this fill",
    )
    realized_pnl: Optional[float] = Field(
        default=None,
        description="Realized P&L for this leg (exit legs only)",
    )
    position_after: Optional[float] = Field(
        default=None,
        description="Position quantity after this leg",
    )
    # Level-anchored stop/target fields (Runbook 42)
    stop_price_abs: Optional[float] = Field(
        default=None,
        description="Absolute stop price set at entry (below = exit for longs, above = exit for shorts)",
    )
    target_price_abs: Optional[float] = Field(
        default=None,
        description="Absolute profit target price set at entry",
    )
    stop_anchor_type: Optional[str] = Field(
        default=None,
        description="How the stop was computed: 'pct', 'atr', 'htf_daily_low', 'htf_prev_daily_low', "
                    "'donchian_lower', 'fib_level', 'candle_low', 'manual'",
    )
    target_anchor_type: Optional[str] = Field(
        default=None,
        description="How the target was computed: 'measured_move', 'htf_daily_high', 'htf_5d_high', "
                    "'fib_level', 'r_multiple', 'manual'",
    )
    # Learning book tags
    learning_book: bool = Field(
        default=False, description="True if this leg is part of learning book"
    )
    experiment_id: Optional[str] = Field(
        default=None, description="Experiment ID if learning book trade"
    )
    experiment_variant: Optional[str] = Field(
        default=None, description="Experiment variant if learning book trade"
    )
    # Time estimation fields (Runbook 49)
    estimated_bars_to_resolution: Optional[int] = Field(
        default=None,
        description="LLM's estimated bars-to-resolution at entry time (from TriggerCondition). "
                    "Measured against actual bars_held post-exit for playbook forecast accuracy.",
    )


class TradeSet(BaseModel):
    """Complete position lifecycle from flat to flat.

    A TradeSet groups all legs from position open through close:
    - Start: position goes from 0 -> non-zero
    - End: position returns to 0
    - Same-bar flatten + reopen: two separate TradeSets

    Uses Weighted Average Cost (WAC) accounting:
    - On buys: WAC recomputed as total_notional / abs(new_position)
    - On sells: realized P&L = sell_qty * (sell_price - WAC_at_time_of_sell)
    """

    model_config = {"extra": "forbid"}

    set_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this trade set",
    )
    symbol: str = Field(description="Trading symbol (e.g., BTC-USD)")
    timeframe: Optional[str] = Field(
        default=None, description="Primary timeframe for this trade set"
    )
    opened_at: datetime = Field(description="Timestamp when position was opened")
    closed_at: Optional[datetime] = Field(
        default=None, description="Timestamp when position was closed (None if open)"
    )
    legs: List[TradeLeg] = Field(
        default_factory=list, description="All legs in chronological order"
    )
    pnl_realized_total: float = Field(
        default=0.0, description="Total realized P&L across all legs"
    )
    fees_total: float = Field(default=0.0, description="Total fees across all legs")
    # Entry side for the overall trade set
    entry_side: Literal["long", "short"] = Field(
        default="long", description="Direction of initial entry"
    )
    # Learning book tags (inherited from first entry leg)
    learning_book: bool = Field(
        default=False, description="True if this trade set is part of learning book"
    )
    experiment_id: Optional[str] = Field(
        default=None, description="Experiment ID if learning book trade"
    )
    # Time-to-resolution metrics (Runbook 49)
    bars_held: Optional[int] = Field(
        default=None,
        description="Integer bar count from entry to close (timeframe-relative). "
                    "Set at position close time.",
    )
    minutes_to_R1: Optional[float] = Field(
        default=None, description="Minutes from entry to first R1 rung hit (R45 trade management)"
    )
    minutes_to_R2: Optional[float] = Field(
        default=None, description="Minutes from entry to R2 rung hit"
    )
    minutes_to_R3: Optional[float] = Field(
        default=None, description="Minutes from entry to R3 rung hit"
    )

    @computed_field  # type: ignore[misc]
    @property
    def r_per_hour(self) -> Optional[float]:
        """Risk-adjusted return per hour of capital exposure.

        Computes (realized_pnl / initial_risk_dollars) / hold_hours.
        Returns None when stop price, hold duration, or initial risk cannot be determined.
        """
        if not self.closed_at or not self.legs:
            return None
        hold_hours = (self.closed_at - self.opened_at).total_seconds() / 3600.0
        if hold_hours <= 0:
            return None
        entry_legs = [leg for leg in self.legs if leg.is_entry and leg.stop_price_abs is not None]
        if not entry_legs:
            return None
        entry = entry_legs[0]
        initial_risk = abs(entry.price - entry.stop_price_abs) * entry.qty
        if initial_risk <= 0:
            return None
        r_return = self.pnl_realized_total / initial_risk
        return round(r_return / hold_hours, 4)

    @computed_field  # type: ignore[misc]
    @property
    def num_legs(self) -> int:
        """Number of legs in this trade set."""
        return len(self.legs)

    @computed_field  # type: ignore[misc]
    @property
    def num_entries(self) -> int:
        """Number of entry legs."""
        return sum(1 for leg in self.legs if leg.is_entry)

    @computed_field  # type: ignore[misc]
    @property
    def num_exits(self) -> int:
        """Number of exit legs."""
        return sum(1 for leg in self.legs if not leg.is_entry)

    @computed_field  # type: ignore[misc]
    @property
    def is_closed(self) -> bool:
        """True if position has returned to flat."""
        return self.closed_at is not None

    @computed_field  # type: ignore[misc]
    @property
    def hold_duration_hours(self) -> Optional[float]:
        """Duration from open to close in hours."""
        if not self.closed_at:
            return None
        delta = (self.closed_at - self.opened_at).total_seconds()
        return round(delta / 3600, 2)

    @computed_field  # type: ignore[misc]
    @property
    def avg_entry_price(self) -> Optional[float]:
        """Volume-weighted average entry price."""
        entry_legs = [leg for leg in self.legs if leg.is_entry]
        if not entry_legs:
            return None
        total_notional = sum(leg.qty * leg.price for leg in entry_legs)
        total_qty = sum(leg.qty for leg in entry_legs)
        return total_notional / total_qty if total_qty > 0 else None

    @computed_field  # type: ignore[misc]
    @property
    def avg_exit_price(self) -> Optional[float]:
        """Volume-weighted average exit price."""
        exit_legs = [leg for leg in self.legs if not leg.is_entry]
        if not exit_legs:
            return None
        total_notional = sum(leg.qty * leg.price for leg in exit_legs)
        total_qty = sum(leg.qty for leg in exit_legs)
        return total_notional / total_qty if total_qty > 0 else None

    @computed_field  # type: ignore[misc]
    @property
    def total_entry_qty(self) -> float:
        """Total quantity entered."""
        return sum(leg.qty for leg in self.legs if leg.is_entry)

    @computed_field  # type: ignore[misc]
    @property
    def total_exit_qty(self) -> float:
        """Total quantity exited."""
        return sum(leg.qty for leg in self.legs if not leg.is_entry)

    @computed_field  # type: ignore[misc]
    @property
    def max_exposure(self) -> float:
        """Maximum position size during lifecycle."""
        max_pos = 0.0
        current_pos = 0.0
        for leg in self.legs:
            if leg.is_entry:
                current_pos += leg.qty
            else:
                current_pos -= leg.qty
            max_pos = max(max_pos, abs(current_pos))
        return max_pos

    @computed_field  # type: ignore[misc]
    @property
    def entry_trigger(self) -> Optional[str]:
        """Trigger ID of first entry leg."""
        for leg in self.legs:
            if leg.is_entry and leg.trigger_id:
                return leg.trigger_id
        return None

    @computed_field  # type: ignore[misc]
    @property
    def exit_trigger(self) -> Optional[str]:
        """Trigger ID of final exit leg."""
        for leg in reversed(self.legs):
            if not leg.is_entry and leg.trigger_id:
                return leg.trigger_id
        return None

    def add_leg(self, leg: TradeLeg) -> None:
        """Add a leg to this trade set and update totals."""
        self.legs.append(leg)
        self.fees_total += leg.fees
        if leg.realized_pnl is not None:
            self.pnl_realized_total += leg.realized_pnl
        # Update closed_at if position returned to flat
        if leg.position_after is not None and abs(leg.position_after) < 1e-10:
            self.closed_at = leg.timestamp

    def to_paired_trade_dict(self) -> Dict[str, Any]:
        """Convert to legacy PairedTrade format for backward compatibility.

        Only valid for trade sets with exactly 1 entry and 1 exit.
        Returns a dict matching the PairedTrade API response schema.
        """
        entry_legs = [leg for leg in self.legs if leg.is_entry]
        exit_legs = [leg for leg in self.legs if not leg.is_entry]

        entry_leg = entry_legs[0] if entry_legs else None
        exit_leg = exit_legs[-1] if exit_legs else None

        return {
            "symbol": self.symbol,
            "side": "buy" if self.entry_side == "long" else "sell",
            "entry_timestamp": entry_leg.timestamp.isoformat() if entry_leg else "",
            "exit_timestamp": exit_leg.timestamp.isoformat() if exit_leg else "",
            "entry_price": entry_leg.price if entry_leg else None,
            "exit_price": exit_leg.price if exit_leg else None,
            "entry_trigger": entry_leg.trigger_id if entry_leg else None,
            "exit_trigger": exit_leg.trigger_id if exit_leg else None,
            "entry_timeframe": self.timeframe,
            "qty": self.total_entry_qty,
            "pnl": self.pnl_realized_total,
            "fees": self.fees_total if self.fees_total > 0 else None,
            "hold_duration_hours": self.hold_duration_hours,
            "risk_used_abs": None,  # Not tracked in TradeSet yet
            "actual_risk_at_stop": None,  # Not tracked in TradeSet yet
            "r_multiple": None,  # Not tracked in TradeSet yet
        }


class TradeSetBuilder:
    """Helper class to build TradeSets from a stream of fills.

    Maintains state for open positions and creates new TradeSets
    when positions go from flat to non-flat.

    Uses WAC accounting:
    - On buys: wac = total_cost / position_qty
    - On sells: realized_pnl = qty * (price - wac)
    """

    def __init__(self) -> None:
        self._open_sets: Dict[str, TradeSet] = {}  # symbol -> open TradeSet
        self._closed_sets: List[TradeSet] = []
        self._position_qty: Dict[str, float] = {}  # symbol -> current position
        self._position_wac: Dict[str, float] = {}  # symbol -> WAC basis

    def process_fill(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        price: float,
        timestamp: datetime,
        fees: float = 0.0,
        trigger_id: Optional[str] = None,
        category: Optional[str] = None,
        reason: Optional[str] = None,
        timeframe: Optional[str] = None,
        exit_fraction: Optional[float] = None,
        learning_book: bool = False,
        experiment_id: Optional[str] = None,
        experiment_variant: Optional[str] = None,
    ) -> TradeLeg:
        """Process a fill and return the created TradeLeg.

        Automatically creates new TradeSets when opening positions,
        and closes them when positions return to flat.
        """
        current_pos = self._position_qty.get(symbol, 0.0)
        current_wac = self._position_wac.get(symbol, 0.0)

        # Determine if this is an entry or exit
        # Long: buy is entry, sell is exit
        # Short: sell is entry, buy is exit
        if current_pos == 0:
            # Opening a new position
            is_entry = True
            entry_side = "long" if side == "buy" else "short"
        elif current_pos > 0:
            # Long position: buy adds, sell reduces
            is_entry = side == "buy"
            entry_side = "long"
        else:
            # Short position: sell adds, buy reduces
            is_entry = side == "sell"
            entry_side = "short"

        # Calculate new position
        delta = qty if side == "buy" else -qty
        new_pos = current_pos + delta

        # Calculate WAC and realized P&L
        realized_pnl: Optional[float] = None
        new_wac = current_wac

        if is_entry:
            # Adding to position - update WAC
            if abs(new_pos) > 1e-10:
                total_cost = abs(current_pos) * current_wac + qty * price
                new_wac = total_cost / abs(new_pos)
        else:
            # Reducing position - calculate realized P&L
            if entry_side == "long":
                realized_pnl = qty * (price - current_wac) - fees
            else:
                realized_pnl = qty * (current_wac - price) - fees

        # Create the leg
        leg = TradeLeg(
            side=side,
            qty=qty,
            price=price,
            fees=fees,
            timestamp=timestamp,
            trigger_id=trigger_id,
            category=category,
            reason=reason,
            is_entry=is_entry,
            exit_fraction=exit_fraction,
            wac_at_fill=current_wac if not is_entry else new_wac,
            realized_pnl=realized_pnl,
            position_after=new_pos,
            learning_book=learning_book,
            experiment_id=experiment_id,
            experiment_variant=experiment_variant,
        )

        # Update position state
        self._position_qty[symbol] = new_pos
        self._position_wac[symbol] = new_wac

        # Manage TradeSets
        if current_pos == 0 and new_pos != 0:
            # Opening new position - create new TradeSet
            trade_set = TradeSet(
                symbol=symbol,
                timeframe=timeframe,
                opened_at=timestamp,
                entry_side=entry_side,
                learning_book=learning_book,
                experiment_id=experiment_id,
            )
            trade_set.add_leg(leg)
            self._open_sets[symbol] = trade_set
        elif symbol in self._open_sets:
            # Add to existing TradeSet
            trade_set = self._open_sets[symbol]
            trade_set.add_leg(leg)

            # Close TradeSet if position returned to flat
            if abs(new_pos) < 1e-10:
                self._closed_sets.append(trade_set)
                del self._open_sets[symbol]
                self._position_qty[symbol] = 0.0
                self._position_wac[symbol] = 0.0

        return leg

    @property
    def open_sets(self) -> Dict[str, TradeSet]:
        """Currently open TradeSets by symbol."""
        return self._open_sets

    @property
    def closed_sets(self) -> List[TradeSet]:
        """Completed TradeSets."""
        return self._closed_sets

    def all_sets(self) -> List[TradeSet]:
        """All TradeSets (open and closed), sorted by opened_at."""
        all_sets = list(self._open_sets.values()) + self._closed_sets
        return sorted(all_sets, key=lambda s: s.opened_at)

    def get_position(self, symbol: str) -> float:
        """Get current position for symbol."""
        return self._position_qty.get(symbol, 0.0)

    def get_wac(self, symbol: str) -> float:
        """Get current WAC for symbol."""
        return self._position_wac.get(symbol, 0.0)

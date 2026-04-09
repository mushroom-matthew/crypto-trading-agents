"""Trailing stop configuration and per-position state (Runbook 85).

Five trailing modes:
  none           — static stop at entry level (current default)
  breakeven_only — ratchet stop to entry at breakeven_at_r; no further movement
  atr_trail      — trail HWM at N×ATR below/above; activated at trail_activation_r
  pct_trail      — trail HWM at fixed percentage; activated at trail_activation_r
  step_trail     — step stop to each R milestone price point (1R→entry, 2R→1R price, etc.)

Close confirmation (wick protection):
  close_required_bars — how many consecutive closes must breach the stop before it fires
  Default = 1 (current single-bar behavior). Set to 2+ to require confirmation.

Partial exits:
  PartialExitSpec — at a given R milestone, exit a fraction of remaining position
  Triggered from AdaptiveTradeManagementState.tick() and consumed by paper_trading.

User-adjustable at session level via PaperTradingConfig.default_trailing_config.
LLM-specifiable per trade via HypothesisModel.trailing_config.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

TrailingStopMode = Literal[
    "none",
    "breakeven_only",
    "atr_trail",
    "pct_trail",
    "step_trail",
    "structure_trail",   # future: trail to nearest S/R level
]


class PartialExitSpec(BaseModel):
    """Partial exit milestone: exit a fraction of position when R multiple is reached."""

    model_config = {"extra": "forbid"}

    at_r_multiple: float = Field(gt=0.0, description="R multiple at which to take partial exit.")
    exit_fraction: float = Field(
        gt=0.0,
        le=1.0,
        description="Fraction of *remaining* position to exit (0 < f <= 1.0).",
    )
    move_stop_to_entry: bool = Field(
        default=False,
        description="Also ratchet stop to breakeven when this milestone is reached.",
    )

    def __hash__(self) -> int:
        return hash((self.at_r_multiple, self.exit_fraction, self.move_stop_to_entry))


class TrailingStopConfig(BaseModel):
    """Session-level or per-trade trailing stop configuration.

    Evaluated per bar by AdaptiveTradeManagementState.tick(). The stop can only
    move in the position's favor (monotonic) — it never widens.

    Mode precedence (applied in order each bar):
      1. Breakeven ratchet (if breakeven_at_r set and R threshold crossed)
      2. Active trail (atr_trail / pct_trail / step_trail) once trail_activation_r reached
      3. Partial exits (independent of stop movement)
    """

    model_config = {"extra": "forbid"}

    # ── Core mode ────────────────────────────────────────────────────────────
    mode: TrailingStopMode = Field(
        default="none",
        description=(
            "none: static stop. "
            "breakeven_only: move to entry at breakeven_at_r, then hold. "
            "atr_trail: trail HWM at atr_trail_multiple × ATR(14). "
            "pct_trail: trail HWM at pct_trail_distance %. "
            "step_trail: ratchet stop to each prior R-milestone price level."
        ),
    )

    # ── Breakeven ratchet (applies to all modes except 'none') ───────────────
    breakeven_at_r: Optional[float] = Field(
        default=None,
        ge=0.1,
        description=(
            "Move stop to entry price when R multiple reaches this value. "
            "E.g. 1.0 = lock in breakeven at 1R. "
            "Also applies to breakeven_only mode as the single ratchet point. "
            "Applies even in atr_trail/pct_trail modes before trail activation."
        ),
    )

    # ── Trail activation ─────────────────────────────────────────────────────
    trail_activation_r: float = Field(
        default=1.5,
        ge=0.5,
        description=(
            "Start trailing only when peak R multiple reaches this value. "
            "Prevents premature stop movement on early-phase noise. "
            "Used by atr_trail, pct_trail, and step_trail modes."
        ),
    )

    # ── ATR trail parameters ─────────────────────────────────────────────────
    atr_trail_multiple: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "Stop trails HWM at this multiple of ATR(14). "
            "E.g. 2.0: for a long at HWM=50000, ATR=200 → stop = 50000 - 2×200 = 49600. "
            "Lower = tighter trail; higher = more room for pullback. "
            "Only used when mode='atr_trail'."
        ),
    )

    # ── Percentage trail parameters ───────────────────────────────────────────
    pct_trail_distance: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "Stop trails HWM at this percentage. "
            "E.g. 2.0: for a long at HWM=50000 → stop = 50000 × (1 - 0.02) = 49000. "
            "Only used when mode='pct_trail'."
        ),
    )

    # ── Close confirmation (wick-out protection) ──────────────────────────────
    close_required_bars: int = Field(
        default=1,
        ge=1,
        le=5,
        description=(
            "Number of consecutive bar closes that must breach the stop before the exit fires. "
            "1 = single-bar close (current default behavior). "
            "2 = requires two consecutive closes below/above stop. "
            "Prevents whipsaw exits on wicks that don't close through the level."
        ),
    )

    # ── Partial exits ─────────────────────────────────────────────────────────
    partial_exits: List[PartialExitSpec] = Field(
        default_factory=list,
        description=(
            "Ordered list of partial exit milestones. Each fires once when peak R "
            "crosses the specified at_r_multiple. "
            "E.g. [{at_r_multiple: 1.0, exit_fraction: 0.33, move_stop_to_entry: true}, "
            "{at_r_multiple: 2.0, exit_fraction: 0.5}] exits 33% at 1R (moving stop to "
            "entry) and 50% of remainder at 2R."
        ),
    )

    @model_validator(mode="after")
    def _validate_partial_exits(self) -> "TrailingStopConfig":
        # Partial fractions don't need to sum to 1.0 (final exit is handled separately)
        # but each spec's at_r_multiple should be unique
        r_multiples = [p.at_r_multiple for p in self.partial_exits]
        if len(r_multiples) != len(set(r_multiples)):
            raise ValueError("partial_exits: at_r_multiple values must be unique.")
        return self

    # ── Presets ───────────────────────────────────────────────────────────────

    @classmethod
    def preset_none(cls) -> "TrailingStopConfig":
        """Static stop — current default. No movement after entry."""
        return cls(mode="none")

    @classmethod
    def preset_breakeven(cls) -> "TrailingStopConfig":
        """Move stop to entry at 1R. No trailing."""
        return cls(mode="breakeven_only", breakeven_at_r=1.0)

    @classmethod
    def preset_atr_trail_standard(cls) -> "TrailingStopConfig":
        """Breakeven at 1R, ATR×2 trail activated at 1.5R. Typical for trending setups."""
        return cls(
            mode="atr_trail",
            breakeven_at_r=1.0,
            trail_activation_r=1.5,
            atr_trail_multiple=2.0,
            close_required_bars=1,
        )

    @classmethod
    def preset_atr_trail_tight(cls) -> "TrailingStopConfig":
        """Breakeven at 0.75R, ATR×1.5 trail activated at 1R. For volatile instruments."""
        return cls(
            mode="atr_trail",
            breakeven_at_r=0.75,
            trail_activation_r=1.0,
            atr_trail_multiple=1.5,
            close_required_bars=2,
        )

    @classmethod
    def preset_pct_trail(cls) -> "TrailingStopConfig":
        """Breakeven at 1R, 2% trail from HWM. Simple percentage-based."""
        return cls(
            mode="pct_trail",
            breakeven_at_r=1.0,
            trail_activation_r=1.5,
            pct_trail_distance=2.0,
            close_required_bars=1,
        )

    @classmethod
    def preset_step_trail(cls) -> "TrailingStopConfig":
        """Step stop to each R milestone. Conservative: stop follows realized gains."""
        return cls(
            mode="step_trail",
            breakeven_at_r=1.0,
            trail_activation_r=1.0,
            close_required_bars=1,
        )

    @classmethod
    def preset_partial_and_trail(cls) -> "TrailingStopConfig":
        """Exit 33% at 1R (move to breakeven), trail remainder at ATR×2."""
        return cls(
            mode="atr_trail",
            breakeven_at_r=1.0,
            trail_activation_r=1.5,
            atr_trail_multiple=2.0,
            close_required_bars=1,
            partial_exits=[
                PartialExitSpec(
                    at_r_multiple=1.0,
                    exit_fraction=0.33,
                    move_stop_to_entry=True,
                ),
                PartialExitSpec(
                    at_r_multiple=2.0,
                    exit_fraction=0.5,
                ),
            ],
        )

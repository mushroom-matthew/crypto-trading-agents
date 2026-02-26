"""Build PositionExitContract from trigger + fill metadata (Runbook 60 Phase M2).

Implements the deterministic contract materialization step at entry time:
  "We must know how we will get out before we get in."

This module is the bridge between:
  - TriggerCondition (LLM-authored trigger with anchor declarations)
  - PositionExitContract (typed, persisted exit plan)

All functions are pure and deterministic — no LLM calls, no I/O.
The caller supplies resolved prices (from Runbook 42 anchor resolution);
this module assembles them into the contract structure.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from schemas.llm_strategist import TriggerCondition
from schemas.position_exit_contract import (
    ExitLeg,
    PositionExitContract,
    TimeExitRule,
)

logger = logging.getLogger(__name__)

EXIT_CONTRACT_BUILDER_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Anchor → leg mapping tables
# ---------------------------------------------------------------------------

# Maps target_anchor_type values to their R-multiple values.
# These do not require a resolved price — they compute from entry/stop at runtime.
_R_MULTIPLE_ANCHOR_MAP: dict[str, float] = {
    "r_multiple_2": 2.0,
    "r_multiple_3": 3.0,
}

# Target anchor types that resolve to price-level legs.
# The caller must supply the resolved price via target_price_abs.
_PRICE_LEVEL_ANCHORS: frozenset[str] = frozenset({
    "htf_daily_high",
    "htf_daily_low",
    "htf_5d_high",
    "htf_5d_low",
    "htf_daily_extreme",
    "htf_5d_extreme",
    "measured_move",
    "donchian_upper",
    "donchian_lower",
    "fib_extension",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_exit_contract(
    trigger: TriggerCondition,
    position_id: str,
    entry_price: float,
    initial_qty: float,
    stop_price_abs: float,
    *,
    target_price_abs: Optional[float] = None,
    plan_id: Optional[str] = None,
    playbook_id: Optional[str] = None,
    template_id: Optional[str] = None,
    snapshot_id: Optional[str] = None,
    snapshot_hash: Optional[str] = None,
    created_at: Optional[datetime] = None,
    max_hold_bars: Optional[int] = None,
) -> Optional[PositionExitContract]:
    """Build a PositionExitContract from a trigger and fill metadata.

    Returns None if the trigger direction is not 'long' or 'short'
    (non-entry triggers cannot produce contracts).

    Args:
        trigger:          Entry trigger condition (direction must be 'long'/'short').
        position_id:      Caller-assigned position ID.
        entry_price:      Actual fill price.
        initial_qty:      Size of the position opened (must be > 0).
        stop_price_abs:   Resolved stop price from Runbook 42 anchor resolution.
        target_price_abs: Resolved target price for price-level anchor types.
                          Required when trigger.target_anchor_type is in
                          _PRICE_LEVEL_ANCHORS; ignored for R-multiple anchors.
        plan_id:          Source StrategyPlan.plan_id for provenance.
        playbook_id:      Playbook ID if a typed playbook was selected (Runbook 52).
        template_id:      Template ID if a strategy template was selected (Runbook 46).
        snapshot_id:      Structure/policy snapshot ID for audit provenance.
        snapshot_hash:    Snapshot hash for audit.
        created_at:       Override creation timestamp; defaults to UTC now.
        max_hold_bars:    Hard bar-count limit for the time exit rule.
                          If None and trigger.estimated_bars_to_resolution is set,
                          uses 2× that estimate as a conservative expiry.

    Returns:
        PositionExitContract or None for non-entry triggers.
    """
    if trigger.direction not in ("long", "short"):
        return None

    side: Literal["long", "short"] = trigger.direction  # type: ignore[assignment]
    ts = created_at or datetime.now(tz=timezone.utc)

    # ------------------------------------------------------------------
    # Build target legs from trigger.target_anchor_type
    # ------------------------------------------------------------------
    target_legs: list[ExitLeg] = []
    anchor = trigger.target_anchor_type

    if anchor is not None:
        if anchor in _R_MULTIPLE_ANCHOR_MAP:
            # R-multiple legs need no resolved price — computed from entry/stop at runtime
            target_legs.append(ExitLeg(
                kind="full_exit",
                trigger_mode="r_multiple",
                fraction=1.0,
                r_multiple=_R_MULTIPLE_ANCHOR_MAP[anchor],
            ))
        elif anchor in _PRICE_LEVEL_ANCHORS:
            if target_price_abs is not None:
                target_legs.append(ExitLeg(
                    kind="full_exit",
                    trigger_mode="price_level",
                    fraction=1.0,
                    price_abs=target_price_abs,
                ))
            else:
                logger.warning(
                    "Trigger '%s': target_anchor_type='%s' requires target_price_abs "
                    "but none was provided — contract created without a price-level leg.",
                    trigger.id, anchor,
                )
        else:
            logger.debug(
                "Trigger '%s': target_anchor_type='%s' not in any known anchor map — "
                "no target leg created.",
                trigger.id, anchor,
            )

    # ------------------------------------------------------------------
    # Build time exit rule
    # ------------------------------------------------------------------
    time_exit: Optional[TimeExitRule] = None
    hold_bars = max_hold_bars
    if hold_bars is None and trigger.estimated_bars_to_resolution is not None:
        # Conservative: 2× estimated bars as the hard expiry
        hold_bars = trigger.estimated_bars_to_resolution * 2
    if hold_bars is not None and hold_bars > 0:
        time_exit = TimeExitRule(max_hold_bars=hold_bars)

    return PositionExitContract(
        position_id=position_id,
        symbol=trigger.symbol,
        side=side,
        created_at=ts,
        source_plan_id=plan_id,
        source_trigger_id=trigger.id,
        source_category=trigger.category,
        template_id=template_id,
        playbook_id=playbook_id,
        snapshot_id=snapshot_id,
        snapshot_hash=snapshot_hash,
        entry_price=entry_price,
        initial_qty=initial_qty,
        stop_price_abs=stop_price_abs,
        target_legs=target_legs,
        time_exit=time_exit,
        remaining_qty=initial_qty,
    )


def can_build_contract(
    trigger: TriggerCondition,
    entry_price: float,
    stop_price_abs: float,
) -> tuple[bool, str]:
    """Check whether a valid contract can be built for the given trigger + prices.

    Returns (True, "") if buildable, or (False, reason) if not.

    This is the pre-entry gate check (Runbook 60 Constraint #1: no entry without
    a contract).  Callers should call this before committing to execution.
    """
    if trigger.direction not in ("long", "short"):
        return False, f"direction='{trigger.direction}' is not an entry direction"

    if entry_price <= 0:
        return False, f"entry_price={entry_price} must be positive"

    if stop_price_abs <= 0:
        return False, f"stop_price_abs={stop_price_abs} must be positive"

    if trigger.direction == "long" and stop_price_abs >= entry_price:
        return (
            False,
            f"Long entry: stop_price_abs={stop_price_abs} must be below "
            f"entry_price={entry_price}",
        )

    if trigger.direction == "short" and stop_price_abs <= entry_price:
        return (
            False,
            f"Short entry: stop_price_abs={stop_price_abs} must be above "
            f"entry_price={entry_price}",
        )

    return True, ""

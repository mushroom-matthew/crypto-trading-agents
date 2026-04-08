"""Hypothesis compiler.

Runbook 78: TradeHypothesis Schema + HypothesisExecutor.

Validates and enriches TradeHypothesis objects at plan-compile time.
Called by generate_strategy_plan_activity() after the LLM produces a plan
with hypotheses[] instead of triggers[].

Compile-time checks:
1. stop_price and target_price are set and directionally correct vs current price.
2. rr_ratio = abs(target - price) / abs(price - stop) >= min_rr_ratio.
3. stop is on the correct side (long: stop < price; short: stop > price).
4. target is on the correct side (long: target > price; short: target < price).
5. If stop_price or target_price are 0/missing, attempt resolution from
   stop_anchor_source / target_anchor_source via the indicator snapshot.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from schemas.hypothesis import TradeHypothesis
from schemas.llm_strategist import IndicatorSnapshot
from schemas.structure_engine import StructureSnapshot

logger = logging.getLogger(__name__)

# Minimum R:R ratio for a hypothesis to be accepted (same as trigger engine)
MIN_RR_RATIO: float = float(__import__("os").environ.get("HYPOTHESIS_MIN_RR", "1.2"))

# Map anchor source strings to indicator snapshot fields
# Long stops (below entry)
_LONG_STOP_ANCHOR_MAP: Dict[str, str] = {
    "htf_daily_low": "htf_daily_low",
    "htf_prev_daily_low": "htf_prev_daily_low",
    "donchian_lower": "donchian_lower_short",
    "donchian_lower_short": "donchian_lower_short",
    "candle_low": "low",
}
# Short stops (above entry)
_SHORT_STOP_ANCHOR_MAP: Dict[str, str] = {
    "htf_daily_high": "htf_daily_high",
    "htf_prev_daily_high": "htf_prev_daily_high",
    "donchian_upper": "donchian_upper_short",
    "donchian_upper_short": "donchian_upper_short",
    "candle_high": "high",
}
# Long targets (above entry)
_LONG_TARGET_ANCHOR_MAP: Dict[str, str] = {
    "htf_daily_high": "htf_daily_high",
    "htf_5d_high": "htf_5d_high",
    "htf_prev_daily_high": "htf_prev_daily_high",
    "donchian_upper": "donchian_upper_short",
    "donchian_upper_short": "donchian_upper_short",
}
# Short targets (below entry)
_SHORT_TARGET_ANCHOR_MAP: Dict[str, str] = {
    "htf_daily_low": "htf_daily_low",
    "htf_5d_low": "htf_5d_low",
    "htf_prev_daily_low": "htf_prev_daily_low",
    "donchian_lower": "donchian_lower_short",
    "donchian_lower_short": "donchian_lower_short",
}


@dataclass
class CompileViolation:
    """A single compile-time violation for one hypothesis."""

    hypothesis_id: str
    reason: str
    detail: str


@dataclass
class HypothesisCompileResult:
    """Output of compile_hypotheses()."""

    valid: List[TradeHypothesis] = field(default_factory=list)
    invalid: List[CompileViolation] = field(default_factory=list)

    @property
    def all_valid(self) -> bool:
        return len(self.invalid) == 0

    def summary(self) -> str:
        return (
            f"{len(self.valid)} valid, {len(self.invalid)} invalid hypotheses"
        )


def _resolve_anchor(
    anchor_source: str,
    indicator: IndicatorSnapshot,
    anchor_map: Dict[str, str],
) -> Optional[float]:
    """Attempt to resolve an anchor price from the indicator snapshot.

    Returns None when the anchor source is not recognized or the
    corresponding indicator field is not populated.
    """
    # Normalize: "r_multiple_2" / "r_multiple_3" → handled separately
    # "atr_1.5x" / "atr_Nx" → handled separately
    # Structural sources like "swing_low_2026-03-12" cannot be resolved here
    field_name = anchor_map.get(anchor_source)
    if field_name is None:
        return None
    value = getattr(indicator, field_name, None)
    if value is None or value <= 0:
        return None
    return float(value)


def _try_resolve_stop(
    hyp: TradeHypothesis,
    indicator: IndicatorSnapshot,
    current_price: float,
) -> Optional[float]:
    """Attempt to resolve stop price from anchor source when stop_price is 0 or missing.

    Returns resolved price or None if resolution fails.
    """
    source = hyp.structure.stop_anchor_source.lower()
    anchor_map = _LONG_STOP_ANCHOR_MAP if hyp.direction == "long" else _SHORT_STOP_ANCHOR_MAP

    # Try direct anchor map lookup
    resolved = _resolve_anchor(source, indicator, anchor_map)
    if resolved is not None:
        return resolved

    # Try pct fallback
    if source.startswith("pct_") or hyp.stop_loss_pct:
        pct = hyp.stop_loss_pct
        if pct and pct > 0:
            if hyp.direction == "long":
                return current_price * (1.0 - pct / 100.0)
            else:
                return current_price * (1.0 + pct / 100.0)

    # Try ATR-based: "atr_1.5x", "atr_2.0x", "atr"
    if source.startswith("atr"):
        atr = indicator.atr_14
        if atr is not None and atr > 0:
            try:
                # Parse multiplier: "atr_1.5x" → 1.5
                mult = 1.5
                if "_" in source and "x" in source:
                    mult = float(source.split("_")[1].rstrip("x"))
                if hyp.direction == "long":
                    return current_price - mult * atr
                else:
                    return current_price + mult * atr
            except (ValueError, IndexError):
                return current_price - 1.5 * atr if hyp.direction == "long" else current_price + 1.5 * atr

    return None


def _try_resolve_target(
    hyp: TradeHypothesis,
    indicator: IndicatorSnapshot,
    current_price: float,
    resolved_stop: float,
) -> Optional[float]:
    """Attempt to resolve target price from anchor source when target_price is 0 or missing."""
    source = hyp.structure.target_anchor_source.lower()
    anchor_map = _LONG_TARGET_ANCHOR_MAP if hyp.direction == "long" else _SHORT_TARGET_ANCHOR_MAP

    # Try direct anchor map lookup
    resolved = _resolve_anchor(source, indicator, anchor_map)
    if resolved is not None:
        return resolved

    # R-multiple targets: "r_multiple_2", "r_multiple_3"
    if source.startswith("r_multiple"):
        try:
            mult = float(source.split("_")[-1])
            stop_dist = abs(current_price - resolved_stop)
            if hyp.direction == "long":
                return current_price + mult * stop_dist
            else:
                return current_price - mult * stop_dist
        except (ValueError, IndexError):
            stop_dist = abs(current_price - resolved_stop)
            if hyp.direction == "long":
                return current_price + 2.0 * stop_dist
            else:
                return current_price - 2.0 * stop_dist

    # Measured move: Donchian range height
    if source == "measured_move":
        upper = getattr(indicator, "donchian_upper_short", None)
        lower = getattr(indicator, "donchian_lower_short", None)
        if upper and lower and upper > lower:
            range_height = upper - lower
            if hyp.direction == "long":
                return current_price + range_height
            else:
                return current_price - range_height

    return None


def compile_hypotheses(
    hypotheses: List[TradeHypothesis],
    indicator: IndicatorSnapshot,
    current_price: float,
    structure: Optional[StructureSnapshot] = None,
    min_rr_ratio: float = MIN_RR_RATIO,
) -> HypothesisCompileResult:
    """Validate and enrich hypotheses at plan-compile time.

    For each hypothesis:
    1. Attempt stop_price resolution if 0/missing via stop_anchor_source.
    2. Attempt target_price resolution if 0/missing via target_anchor_source.
    3. Verify stop is on the correct side of current_price.
    4. Verify target is on the correct side of current_price.
    5. Compute rr_ratio = abs(target - current_price) / abs(current_price - stop).
    6. Verify rr_ratio >= min_rr_ratio.
    7. Fill rr_ratio on valid hypotheses.

    Returns:
        HypothesisCompileResult with valid and invalid lists.
        Invalid hypotheses are NOT included in valid list.
        Each violation emits a structured reason for the repair prompt.
    """
    result = HypothesisCompileResult()

    for hyp in hypotheses:
        violations: List[str] = []

        stop_price = hyp.stop_price
        target_price = hyp.target_price

        # ── Step 1–2: Attempt resolution of missing prices ────────────────
        if stop_price == 0 or stop_price is None:
            resolved = _try_resolve_stop(hyp, indicator, current_price)
            if resolved is not None:
                stop_price = resolved
                logger.debug(
                    "compiler: resolved stop_price=%.4f for %s from '%s'",
                    stop_price, hyp.id, hyp.structure.stop_anchor_source,
                )
            else:
                violations.append(
                    f"stop_price=0 and could not resolve from stop_anchor_source="
                    f"'{hyp.structure.stop_anchor_source}'. "
                    "Set stop_price explicitly or use a resolvable anchor "
                    "(htf_daily_low, htf_prev_daily_low, atr_1.5x, pct)."
                )

        if target_price == 0 or target_price is None:
            resolved = _try_resolve_target(hyp, indicator, current_price, stop_price or current_price)
            if resolved is not None:
                target_price = resolved
                logger.debug(
                    "compiler: resolved target_price=%.4f for %s from '%s'",
                    target_price, hyp.id, hyp.structure.target_anchor_source,
                )
            else:
                violations.append(
                    f"target_price=0 and could not resolve from target_anchor_source="
                    f"'{hyp.structure.target_anchor_source}'. "
                    "Set target_price explicitly or use: htf_daily_high, r_multiple_2, "
                    "r_multiple_3, measured_move."
                )

        if violations:
            for v in violations:
                result.invalid.append(
                    CompileViolation(hyp.id, "missing_stop_or_target", v)
                )
            continue

        # ── Step 3–4: Directional correctness ─────────────────────────────
        if hyp.direction == "long":
            if stop_price >= current_price:
                result.invalid.append(CompileViolation(
                    hyp.id, "stop_wrong_side",
                    f"Long stop {stop_price:.4f} >= current_price {current_price:.4f}. "
                    "Long stop must be below current price."
                ))
                continue
            if target_price <= current_price:
                result.invalid.append(CompileViolation(
                    hyp.id, "target_wrong_side",
                    f"Long target {target_price:.4f} <= current_price {current_price:.4f}. "
                    "Long target must be above current price."
                ))
                continue
        else:  # short
            if stop_price <= current_price:
                result.invalid.append(CompileViolation(
                    hyp.id, "stop_wrong_side",
                    f"Short stop {stop_price:.4f} <= current_price {current_price:.4f}. "
                    "Short stop must be above current price."
                ))
                continue
            if target_price >= current_price:
                result.invalid.append(CompileViolation(
                    hyp.id, "target_wrong_side",
                    f"Short target {target_price:.4f} >= current_price {current_price:.4f}. "
                    "Short target must be below current price."
                ))
                continue

        # ── Step 5–6: R:R ratio ───────────────────────────────────────────
        entry_to_stop = abs(current_price - stop_price)
        entry_to_target = abs(current_price - target_price)

        if entry_to_stop <= 0:
            result.invalid.append(CompileViolation(
                hyp.id, "zero_stop_distance",
                f"entry_to_stop distance is zero (price={current_price:.4f}, stop={stop_price:.4f}). "
                "Stop must not equal current price."
            ))
            continue

        rr = entry_to_target / entry_to_stop

        if rr < min_rr_ratio:
            result.invalid.append(CompileViolation(
                hyp.id, "insufficient_rr",
                f"R:R={rr:.2f} < min_rr_ratio={min_rr_ratio}. "
                f"target={target_price:.4f}, stop={stop_price:.4f}, price={current_price:.4f}. "
                "Move target further from entry or stop closer to entry."
            ))
            continue

        # ── Step 7: Enrich with computed rr_ratio ─────────────────────────
        # Create a new hypothesis with rr_ratio filled (immutable model copy)
        enriched = hyp.model_copy(update={
            "stop_price": stop_price,
            "target_price": target_price,
            "rr_ratio": round(rr, 3),
        })
        result.valid.append(enriched)

    logger.info(
        "hypothesis_compiler: %s (price=%.4f, min_rr=%.1f)",
        result.summary(), current_price, min_rr_ratio,
    )
    for violation in result.invalid:
        logger.warning(
            "hypothesis_compiler: INVALID %s — %s: %s",
            violation.hypothesis_id, violation.reason, violation.detail,
        )

    return result

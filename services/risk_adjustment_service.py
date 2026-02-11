"""Helpers for interpreting judge feedback into persistent risk adjustments."""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Iterable, Optional

from agents.strategies.risk_engine import RiskProfile
from schemas.judge_feedback import JudgeFeedback
from schemas.strategy_run import RiskAdjustmentState, RiskLimitSettings, StrategyRun

logger = logging.getLogger(__name__)


_CUT_PATTERN = re.compile(r"cut\s+risk\s+by\s+(?P<pct>\d+(?:\.\d+)?)%")
_CAP_PATTERN = re.compile(r"cap(?:ping)?\s+(?:risk\s+)?at\s+(?P<pct>\d+(?:\.\d+)?)%")
_WIN_PATTERN = re.compile(r"(?P<count>\d+|one|two|three|four|five)\s+winning\s+day", re.IGNORECASE)
_NUM_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


_DEFAULT_MIN_MULTIPLIER = 0.25
_DEFAULT_MAX_MULTIPLIER = 3.0


def _load_multiplier_bounds() -> tuple[float, float]:
    """Load clamp bounds for judge risk multipliers from env."""

    def _read(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid %s=%r, using default %.2f", name, raw, default)
            return default

    min_mult = _read("JUDGE_RISK_MULTIPLIER_MIN", _DEFAULT_MIN_MULTIPLIER)
    max_mult = _read("JUDGE_RISK_MULTIPLIER_MAX", _DEFAULT_MAX_MULTIPLIER)
    if min_mult <= 0 or max_mult <= 0 or min_mult > max_mult:
        logger.warning(
            "Invalid risk multiplier bounds (min=%s max=%s); using defaults %.2f..%.2f",
            min_mult,
            max_mult,
            _DEFAULT_MIN_MULTIPLIER,
            _DEFAULT_MAX_MULTIPLIER,
        )
        return _DEFAULT_MIN_MULTIPLIER, _DEFAULT_MAX_MULTIPLIER
    return min_mult, max_mult


def _clamp_multiplier(value: float, symbol: str, source: str) -> float:
    """Clamp multipliers to a safe range and log when clamping occurs."""

    min_mult, max_mult = _load_multiplier_bounds()
    clamped = min(max(value, min_mult), max_mult)
    if clamped != value:
        logger.warning(
            "Clamped judge risk multiplier for %s (%s): %.4f -> %.4f (min=%.2f max=%.2f)",
            symbol,
            source,
            value,
            clamped,
            min_mult,
            max_mult,
        )
    return clamped


def _parse_multiplier(text: str) -> Optional[float]:
    lowered = text.lower()
    match = _CUT_PATTERN.search(lowered)
    if match:
        pct = float(match.group("pct"))
        return max(0.0, 1.0 - pct / 100.0)
    match = _CAP_PATTERN.search(lowered)
    if match:
        pct = float(match.group("pct"))
        return max(0.0, min(1.0, pct / 100.0))
    if "full allocation" in lowered or "restore risk" in lowered:
        return 1.0
    return None


def multiplier_from_instruction(instruction: str) -> Optional[float]:
    """Public helper to extract multiplier from sizing text."""

    return _parse_multiplier(instruction)


def _parse_win_requirement(text: str) -> Optional[int]:
    match = _WIN_PATTERN.search(text)
    if not match:
        if "two winning days" in text.lower():
            return 2
        return None
    raw = match.group("count").lower()
    if raw.isdigit():
        return max(1, int(raw))
    return _NUM_WORDS.get(raw)


def _global_multiplier(adjustments: Dict[str, RiskAdjustmentState]) -> float:
    """Return strictest multiplier across active adjustments."""

    if not adjustments:
        return 1.0
    return min(state.multiplier for state in adjustments.values())


def apply_judge_risk_feedback(
    run: StrategyRun,
    feedback: JudgeFeedback,
    winning_day: bool,
    *,
    advance_day: bool = True,
) -> bool:
    """Update StrategyRun risk adjustments based on judge feedback.

    Set advance_day=False to apply sizing instructions without progressing
    win-based restoration counters (useful for intraday adjustments).
    """

    adjustments = dict(run.risk_adjustments or {})
    changed = False
    # First update existing adjustments for new performance info (day-level only).
    if advance_day:
        for symbol, state in list(adjustments.items()):
            if state.record_day(winning_day):
                adjustments.pop(symbol, None)
                changed = True
            else:
                adjustments[symbol] = state

    instructions = feedback.strategist_constraints.sizing_adjustments or {}
    structured = feedback.constraints.symbol_risk_multipliers or {}
    all_symbols = set(structured) | set(instructions)
    for symbol in sorted(all_symbols):
        instruction = instructions.get(symbol)
        multiplier = None
        source = "instruction"
        if symbol in structured:
            multiplier = structured.get(symbol)
            source = "structured"
        elif instruction:
            multiplier = _parse_multiplier(instruction)

        if multiplier is None:
            continue

        multiplier = _clamp_multiplier(multiplier, symbol, source)
        wins_req = _parse_win_requirement(instruction) if instruction else None
        if multiplier >= 0.999:
            if symbol in adjustments:
                adjustments.pop(symbol, None)
                changed = True
            feedback.constraints.symbol_risk_multipliers[symbol] = 1.0
            continue
        existing = adjustments.get(symbol)
        if existing and existing.multiplier == multiplier and existing.restore_after_wins == wins_req:
            if existing.instruction != instruction:
                existing.instruction = instruction or existing.instruction
                adjustments[symbol] = existing
                changed = True
            feedback.constraints.symbol_risk_multipliers[symbol] = multiplier
            continue
        adjustments[symbol] = RiskAdjustmentState(
            multiplier=multiplier,
            restore_after_wins=wins_req,
            wins_progress=0,
            instruction=instruction or ("Structured multiplier from judge" if source == "structured" else None),
        )
        feedback.constraints.symbol_risk_multipliers[symbol] = multiplier
        changed = True

    run.risk_adjustments = adjustments
    run.latest_judge_feedback = feedback
    return changed


def effective_risk_limits(run: StrategyRun) -> RiskLimitSettings:
    """Return the base risk limits scaled by the strictest active adjustment."""

    base = run.config.risk_limits or RiskLimitSettings()
    multiplier = _global_multiplier(run.risk_adjustments or {})
    if multiplier >= 0.999:
        return base
    return base.scaled(multiplier)


def snapshot_adjustments(adjustments: Dict[str, RiskAdjustmentState]) -> Iterable[Dict[str, object]]:
    """Serialize adjustments for prompt context/humans."""

    for symbol, state in adjustments.items():
        yield {
            "symbol": symbol,
            "multiplier": state.multiplier,
            "restore_after_wins": state.restore_after_wins,
            "wins_progress": state.wins_progress,
            "instruction": state.instruction,
        }


def build_risk_profile(run: StrategyRun) -> RiskProfile:
    """Translate the current run adjustments into a structured risk profile."""

    adjustments = run.risk_adjustments or {}
    if not adjustments:
        return RiskProfile()
    global_multiplier = _global_multiplier(adjustments)
    symbol_multipliers: Dict[str, float] = {}
    base = global_multiplier if global_multiplier > 0 else 1.0
    for symbol, state in adjustments.items():
        symbol_multipliers[symbol] = state.multiplier / base if base > 0 else state.multiplier
    return RiskProfile(global_multiplier=global_multiplier, symbol_multipliers=symbol_multipliers)

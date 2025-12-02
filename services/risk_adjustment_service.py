"""Helpers for interpreting judge feedback into persistent risk adjustments."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

from schemas.judge_feedback import JudgeFeedback
from schemas.strategy_run import RiskAdjustmentState, RiskLimitSettings, StrategyRun


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


def apply_judge_risk_feedback(run: StrategyRun, feedback: JudgeFeedback, winning_day: bool) -> bool:
    """Update StrategyRun risk adjustments based on judge feedback."""

    adjustments = dict(run.risk_adjustments or {})
    changed = False
    # First update existing adjustments for new performance info.
    for symbol, state in list(adjustments.items()):
        if state.record_day(winning_day):
            adjustments.pop(symbol, None)
            changed = True
        else:
            adjustments[symbol] = state

    instructions = feedback.strategist_constraints.sizing_adjustments or {}
    for symbol, instruction in instructions.items():
        multiplier = _parse_multiplier(instruction)
        if multiplier is None:
            continue
        wins_req = _parse_win_requirement(instruction)
        if multiplier >= 0.999:
            if symbol in adjustments:
                adjustments.pop(symbol, None)
                changed = True
            continue
        adjustments[symbol] = RiskAdjustmentState(
            multiplier=multiplier,
            restore_after_wins=wins_req,
            wins_progress=0,
            instruction=instruction,
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

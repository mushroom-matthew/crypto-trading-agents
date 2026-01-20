"""Deterministic compiler that validates StrategyPlan trigger expressions."""

from __future__ import annotations

import ast
import logging
import re
from typing import Iterable, List, Optional, Set, Tuple

from schemas.llm_strategist import StrategyPlan, TriggerCondition
from schemas.compiled_plan import CompiledExpression, CompiledPlan, CompiledTrigger

logger = logging.getLogger(__name__)


class TriggerCompilationError(ValueError):
    """Raised when a trigger cannot be compiled deterministically."""


class TimeframeMismatchWarning:
    """Warning about triggers referencing unavailable timeframes."""

    def __init__(self, trigger_id: str, missing_timeframes: Set[str], rule_type: str):
        self.trigger_id = trigger_id
        self.missing_timeframes = missing_timeframes
        self.rule_type = rule_type  # 'entry' or 'exit'

    def __str__(self) -> str:
        tfs = ", ".join(sorted(self.missing_timeframes))
        return f"Trigger '{self.trigger_id}' {self.rule_type}_rule references unavailable timeframes: {tfs}"


# Pattern to extract timeframe prefixes like tf_4h_, tf_1h_, tf_15m_
_TIMEFRAME_PREFIX_PATTERN = re.compile(r"tf_(\d+[mhd])_")


def extract_referenced_timeframes(expr: str) -> Set[str]:
    """Extract all timeframe references from an expression.

    Looks for patterns like tf_4h_ema_short, tf_1h_rsi_14, tf_15m_close
    and extracts the timeframe part (4h, 1h, 15m).

    Args:
        expr: The trigger rule expression

    Returns:
        Set of timeframe strings found (e.g., {'4h', '1h', '15m'})
    """
    if not expr:
        return set()
    return set(_TIMEFRAME_PREFIX_PATTERN.findall(expr))


def validate_trigger_timeframes(
    trigger: TriggerCondition,
    available_timeframes: Set[str],
) -> List[TimeframeMismatchWarning]:
    """Validate that trigger rules only reference available timeframes.

    Args:
        trigger: The trigger condition to validate
        available_timeframes: Set of timeframes that are loaded (e.g., {'5m', '1h', '4h'})

    Returns:
        List of warnings for missing timeframe references
    """
    warnings: List[TimeframeMismatchWarning] = []

    # Check entry rule
    entry_timeframes = extract_referenced_timeframes(trigger.entry_rule)
    missing_entry = entry_timeframes - available_timeframes
    if missing_entry:
        warnings.append(TimeframeMismatchWarning(trigger.id, missing_entry, "entry"))

    # Check exit rule
    exit_timeframes = extract_referenced_timeframes(trigger.exit_rule)
    missing_exit = exit_timeframes - available_timeframes
    if missing_exit:
        warnings.append(TimeframeMismatchWarning(trigger.id, missing_exit, "exit"))

    return warnings


def validate_plan_timeframes(
    plan: StrategyPlan,
    available_timeframes: Set[str],
) -> Tuple[List[TimeframeMismatchWarning], List[str]]:
    """Validate all triggers in a plan for timeframe compatibility.

    Args:
        plan: The strategy plan to validate
        available_timeframes: Set of available timeframes

    Returns:
        Tuple of (warnings, blocked_trigger_ids)
        - warnings: All timeframe mismatch warnings
        - blocked_trigger_ids: IDs of triggers that will be blocked due to missing timeframes
    """
    all_warnings: List[TimeframeMismatchWarning] = []
    blocked_ids: List[str] = []

    for trigger in plan.triggers:
        warnings = validate_trigger_timeframes(trigger, available_timeframes)
        all_warnings.extend(warnings)

        # If any warning exists, this trigger will likely fail to fire
        if warnings:
            blocked_ids.append(trigger.id)
            for w in warnings:
                logger.warning(
                    "Trigger '%s' references unavailable timeframes %s in %s_rule; "
                    "trigger may never fire. Available: %s",
                    trigger.id,
                    sorted(w.missing_timeframes),
                    w.rule_type,
                    sorted(available_timeframes),
                )

    return all_warnings, blocked_ids


_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Attribute,
    ast.Subscript,
    ast.Constant,
    ast.Load,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.USub,
    ast.UAdd,
    ast.Mult,
    ast.Div,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
    ast.Is,
    ast.IsNot,
)


_BETWEEN_TOKEN = r"(?:-?\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*)"
_BETWEEN_PATTERN = re.compile(
    rf"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s+between\s+(?P<low>{_BETWEEN_TOKEN})\s+and\s+(?P<high>{_BETWEEN_TOKEN})",
    re.IGNORECASE,
)


def _normalize_expression(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return expr

    def replacer(match: re.Match[str]) -> str:
        field = match.group("field")
        low = match.group("low")
        high = match.group("high")
        return f"(({field}) >= ({low}) and ({field}) <= ({high}))"

    return _BETWEEN_PATTERN.sub(replacer, expr)


def _validate_ast(node: ast.AST, allowed_names: Set[str]) -> None:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise TriggerCompilationError(f"Unsupported syntax in trigger: {ast.dump(child)}")
        if isinstance(child, ast.Name):
            identifier = child.id
            if identifier not in allowed_names and identifier.lower() not in {"true", "false", "none"}:
                raise TriggerCompilationError(f"Unknown identifier '{identifier}' in trigger expression")
        if isinstance(child, ast.Subscript):
            # Only allow constant, non-negative integer indexes (e.g., recent_tests[0]).
            if not isinstance(child.slice, ast.Constant) or not isinstance(child.slice.value, int):
                raise TriggerCompilationError(f"Unsupported subscript in trigger: {ast.dump(child)}")
            if child.slice.value < 0:
                raise TriggerCompilationError(f"Negative index not allowed in trigger: {ast.dump(child)}")


def _compile_expression(expr: str, allowed_names: Set[str]) -> CompiledExpression:
    if not expr:
        return CompiledExpression(source="", normalized="")
    normalized = _normalize_expression(expr)
    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise TriggerCompilationError(
            f"Invalid syntax in expression '{expr}': {exc.msg} at {exc.offset}"
        ) from exc
    _validate_ast(parsed, allowed_names)
    return CompiledExpression(source=expr, normalized=normalized)


def _collect_identifiers(expr: str) -> Set[str]:
    if not expr:
        return set()
    normalized = _normalize_expression(expr)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return {token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)}
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _allowed_names_from_trigger(trigger: TriggerCondition) -> Set[str]:
    base_names = {
        "timeframe",
        "trend_state",
        "vol_state",
        "symbol",
        "position",  # Current position state: 'flat', 'long', or 'short'
        "close",
        "open",
        "high",
        "low",
        "volume",
        "sma_short",
        "sma_medium",
        "sma_long",
        "ema_short",
        "ema_medium",
        "ema_long",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "roc_short",
        "roc_medium",
        "realized_vol_short",
        "realized_vol_medium",
        "bollinger_upper",
        "bollinger_lower",
        "bollinger_middle",
        "donchian_upper_short",
        "donchian_lower_short",
        # Cycle indicators (200-bar window)
        "cycle_high_200",
        "cycle_low_200",
        "cycle_range_200",
        "cycle_position",
        # Fibonacci retracement levels
        "fib_236",
        "fib_382",
        "fib_500",
        "fib_618",
        "fib_786",
        # Expansion/contraction ratios
        "last_expansion_pct",
        "last_contraction_pct",
        "expansion_contraction_ratio",
    }
    identifiers = _collect_identifiers(trigger.entry_rule) | _collect_identifiers(trigger.exit_rule)
    return base_names | identifiers


def compile_trigger(trigger: TriggerCondition, allowed_names: Set[str]) -> CompiledTrigger:
    entry = _compile_expression(trigger.entry_rule, allowed_names)
    exit_expr = _compile_expression(trigger.exit_rule, allowed_names)
    return CompiledTrigger(
        trigger_id=trigger.id,
        symbol=trigger.symbol,
        direction=trigger.direction,
        category=trigger.category,
        entry=entry if entry.source else None,
        exit=exit_expr if exit_expr.source else None,
    )


def compile_plan(plan: StrategyPlan) -> CompiledPlan:
    if not plan.run_id:
        raise TriggerCompilationError("StrategyPlan.run_id is required before compilation")
    compiled_triggers: List[CompiledTrigger] = []
    for trigger in plan.triggers:
        allowed_names = _allowed_names_from_trigger(trigger)
        compiled_triggers.append(compile_trigger(trigger, allowed_names))
    return CompiledPlan(plan_id=plan.plan_id, run_id=plan.run_id, triggers=compiled_triggers)


def validate_plan(plan: StrategyPlan) -> List[str]:
    errors: List[str] = []
    try:
        compile_plan(plan)
    except TriggerCompilationError as exc:
        errors.append(str(exc))
    return errors

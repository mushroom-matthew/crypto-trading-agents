"""Deterministic compiler that validates StrategyPlan trigger expressions."""

from __future__ import annotations

import ast
import re
from typing import Iterable, List, Set

from schemas.llm_strategist import StrategyPlan, TriggerCondition
from schemas.compiled_plan import CompiledExpression, CompiledPlan, CompiledTrigger


class TriggerCompilationError(ValueError):
    """Raised when a trigger cannot be compiled deterministically."""


_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
)


_BETWEEN_PATTERN = re.compile(
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s+between\s+(?P<low>-?\d+\.?\d*)\s+and\s+(?P<high>-?\d+\.?\d*)",
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

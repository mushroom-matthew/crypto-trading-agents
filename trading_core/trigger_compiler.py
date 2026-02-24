"""Deterministic compiler that validates StrategyPlan trigger expressions."""

from __future__ import annotations

import ast
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Optional, Set, Tuple

from data_loader.utils import timeframe_to_seconds
from schemas.llm_strategist import IndicatorSnapshot, StrategyPlan, TriggerCondition
from schemas.compiled_plan import CompiledExpression, CompiledPlan, CompiledTrigger
from vector_store.retriever import allowed_identifiers_for_template as _vs_allowed_identifiers

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


class IdentifierMismatchWarning:
    """Warning about triggers referencing unknown identifiers."""

    def __init__(self, trigger_id: str, unknown_identifiers: Set[str], rule_type: str):
        self.trigger_id = trigger_id
        self.unknown_identifiers = unknown_identifiers
        self.rule_type = rule_type  # 'entry', 'exit', or 'hold'

    def __str__(self) -> str:
        identifiers = ", ".join(sorted(self.unknown_identifiers))
        return f"Trigger '{self.trigger_id}' {self.rule_type}_rule references unknown identifiers: {identifiers}"


class AtrTautologyWarning:
    """Warning about cross-timeframe ATR comparisons that are always true.

    Comparing ATR across timeframes without a ratio multiplier is a tautology
    because higher-timeframe ATR is ALWAYS larger than lower-timeframe ATR
    (e.g., tf_1d_atr > tf_4h_atr is always true).
    """

    def __init__(self, trigger_id: str, expr: str, rule_type: str, detail: str):
        self.trigger_id = trigger_id
        self.expr = expr
        self.rule_type = rule_type
        self.detail = detail

    def __str__(self) -> str:
        return (
            f"Trigger '{self.trigger_id}' {self.rule_type}_rule has ATR tautology: "
            f"{self.detail} — use ratio comparisons like '> 2.5 * tf_4h_atr' instead"
        )


# Pattern to extract timeframe prefixes like tf_4h_, tf_1h_, tf_15m_
_TIMEFRAME_PREFIX_PATTERN = re.compile(r"tf_(\d+[mhd])_")

# Pattern for cross-timeframe ATR identifiers: tf_<timeframe>_atr or tf_<timeframe>_atr_14
_TF_ATR_PATTERN = re.compile(r"tf_(\d+[mhd])_atr(?:_14)?$")


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


def _indicator_field_names() -> Set[str]:
    fields = set(IndicatorSnapshot.model_fields.keys())
    fields.discard("as_of")
    return fields


def _alias_names(names: Iterable[str]) -> Set[str]:
    aliases: Set[str] = set()
    for name in names:
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            aliases.add(parts[0])
    return aliases


def build_allowed_identifiers(available_timeframes: Iterable[str]) -> Set[str]:
    """Return the allowed identifier set for trigger expressions."""
    indicator_fields = _indicator_field_names()
    indicator_aliases = _alias_names(indicator_fields)

    derived_fields = {
        "bollinger_middle",
        "trend_state",
        "vol_state",
        "position",
        "is_flat",
        "is_long",
        "is_short",
        "position_qty",
        "position_value",
        "entry_price",
        "avg_entry_price",
        "entry_side",
        "position_opened_at",
        "unrealized_pnl_pct",
        "unrealized_pnl_abs",
        "unrealized_pnl",
        "position_pnl_pct",
        "position_pnl_abs",
        "position_age_minutes",
        "position_age_hours",
        "holding_minutes",
        "holding_hours",
        "time_in_trade_min",
        "time_in_trade_hours",
        "nearest_support",
        "nearest_resistance",
        "distance_to_support_pct",
        "distance_to_resistance_pct",
        "trend",
        "recent_tests",
        # Structural exit / stop-target identifiers (runtime context, not indicator snapshot)
        "stop_hit",
        "target_hit",
        "force_flatten",
        "below_stop",
        "above_target",
        "stop_price",
        "target_price",
        "stop_distance_pct",
        "target_distance_pct",
        # R-tracking identifiers (Runbook 45 — adaptive trade management)
        "current_R",
        "mfe_r",
        "mae_r",
        "r1_reached",
        "r2_reached",
        "r3_reached",
        "trade_state",
        "position_fraction",
    }

    allowed = set(indicator_fields) | indicator_aliases | derived_fields

    tf_indicator_fields = {name for name in indicator_fields if name not in {"symbol", "timeframe"}}
    tf_aliases = _alias_names(tf_indicator_fields)
    tf_fields = tf_indicator_fields | tf_aliases | {"bollinger_middle", "trend_state", "vol_state"}

    for timeframe in available_timeframes:
        prefix = f"tf_{timeframe.replace('-', '_')}_"
        for field in tf_fields:
            allowed.add(f"{prefix}{field}")

    return allowed


def validate_trigger_identifiers(
    trigger: TriggerCondition,
    allowed_identifiers: Set[str],
) -> List[IdentifierMismatchWarning]:
    warnings: List[IdentifierMismatchWarning] = []
    for rule_type, expr in (
        ("entry", trigger.entry_rule),
        ("exit", trigger.exit_rule),
        ("hold", trigger.hold_rule),
    ):
        if not expr:
            continue
        identifiers = _collect_identifiers(expr)
        unknown = {
            name for name in identifiers
            if name not in allowed_identifiers and name.lower() not in {"true", "false", "none"}
        }
        if unknown:
            warnings.append(IdentifierMismatchWarning(trigger.id, unknown, rule_type))
    return warnings


def validate_plan_identifiers(
    plan: StrategyPlan,
    available_timeframes: Set[str],
) -> List[IdentifierMismatchWarning]:
    warnings: List[IdentifierMismatchWarning] = []
    allowed = build_allowed_identifiers(available_timeframes)
    for trigger in plan.triggers:
        warnings.extend(validate_trigger_identifiers(trigger, allowed))
    return warnings


def validate_min_hold_vs_exits(
    plan: StrategyPlan,
    min_hold_hours: float,
) -> list[str]:
    """Check whether min_hold >= smallest exit trigger timeframe.

    Returns a list of warning strings (empty when OK).
    """
    if min_hold_hours <= 0:
        return []

    exit_timeframe_seconds: list[float] = []
    for trigger in plan.triggers:
        if trigger.category == "emergency_exit":
            continue
        is_exit = trigger.direction == "exit" or bool(trigger.exit_rule)
        if not is_exit:
            continue
        try:
            tf_sec = timeframe_to_seconds(trigger.timeframe)
            exit_timeframe_seconds.append(tf_sec)
        except (ValueError, TypeError):
            continue

    if not exit_timeframe_seconds:
        return []

    smallest_exit_sec = min(exit_timeframe_seconds)
    smallest_exit_hours = smallest_exit_sec / 3600.0
    min_hold_sec = min_hold_hours * 3600.0

    warnings: list[str] = []
    if min_hold_sec >= smallest_exit_sec:
        warnings.append(
            f"min_hold ({min_hold_hours:.2f}h) >= smallest exit timeframe "
            f"({smallest_exit_hours:.2f}h); exits will be mechanically forced "
            f"at the hold floor, preventing exit signal quality assessment"
        )
    return warnings


def detect_degenerate_hold_rules(
    hold_rule: str,
    trigger_id: str = "",
) -> List[str]:
    """Detect hold rules that are too permissive and suppress all normal exits.

    Degenerate hold rules include:
    - Single-condition rules (no ``and``): easy to be always true
    - ``rsi_14 > X`` where X < 50: RSI is above 50 most of the time in trending markets

    Returns:
        List of warning strings (empty when OK).
    """
    if not hold_rule:
        return []
    warnings: List[str] = []
    normalized = hold_rule.strip()

    # Check for single-condition rules (no 'and' operator)
    if " and " not in normalized.lower():
        warnings.append(
            f"Trigger '{trigger_id}': single-condition hold rule '{normalized}' "
            f"is likely too permissive — use compound conditions "
            f"(e.g., 'rsi_14 > 60 and close > sma_medium and atr_14 < atr_14_prev')"
        )

    # Check for RSI > X where X < 50
    rsi_match = re.search(r"rsi_14\s*>\s*(\d+(?:\.\d+)?)", normalized)
    if rsi_match:
        threshold = float(rsi_match.group(1))
        if threshold < 50:
            warnings.append(
                f"Trigger '{trigger_id}': hold rule uses rsi_14 > {threshold} "
                f"(below 50 is almost always true in trending markets — use > 60 or higher)"
            )

    return warnings


def warn_cross_category_exits(triggers: Iterable[TriggerCondition]) -> List[str]:
    """Warn if a symbol has entry triggers in multiple categories.

    Exit rules only close positions from the SAME category. If a symbol has
    entries in category A and category B, exits in A will NOT close positions
    opened by B. This is a common source of "stuck" positions.

    Returns:
        List of warning strings (empty when OK).
    """
    from collections import defaultdict

    symbol_categories: dict[str, set[str]] = defaultdict(set)
    for trigger in triggers:
        if trigger.direction in ("long", "short") and trigger.entry_rule:
            symbol_categories[trigger.symbol].add(trigger.category)

    warnings: List[str] = []
    for symbol, categories in symbol_categories.items():
        if len(categories) > 1:
            cats = ", ".join(sorted(categories))
            warnings.append(
                f"{symbol} has entry triggers in multiple categories ({cats}). "
                f"Exit rules only close positions from the SAME category — "
                f"positions opened by one category cannot be closed by another."
            )
    return warnings


def _timeframe_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes for comparison."""
    match = re.match(r"(\d+)([mhd])", tf)
    if not match:
        return 0
    val, unit = int(match.group(1)), match.group(2)
    if unit == "m":
        return val
    elif unit == "h":
        return val * 60
    elif unit == "d":
        return val * 1440
    return 0


def detect_atr_tautologies(
    expr: str,
    trigger_id: str = "",
    rule_type: str = "entry",
) -> List[AtrTautologyWarning]:
    """Detect cross-timeframe ATR comparisons that are always true.

    A comparison like ``tf_1d_atr > tf_4h_atr`` is always true because
    higher-timeframe ATR aggregates more price movement. The LLM should
    use ratio comparisons instead (e.g., ``tf_1d_atr > 2.5 * tf_4h_atr``).

    Returns:
        List of AtrTautologyWarning for any tautological comparisons found.
    """
    if not expr:
        return []
    warnings: List[AtrTautologyWarning] = []
    try:
        normalized = _normalize_expression(expr)
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        # Check each comparison pair: left op[i] comparator[i]
        operands = [node.left] + list(node.comparators)
        for i, op in enumerate(node.ops):
            left_node = operands[i]
            right_node = operands[i + 1]
            # Only check > and >= (and reversed < / <=)
            if isinstance(op, (ast.Gt, ast.GtE)):
                larger, smaller = left_node, right_node
            elif isinstance(op, (ast.Lt, ast.LtE)):
                larger, smaller = right_node, left_node
            else:
                continue

            # Both sides must be simple Name nodes (no multiplication = no ratio)
            if not (isinstance(larger, ast.Name) and isinstance(smaller, ast.Name)):
                continue

            larger_match = _TF_ATR_PATTERN.match(larger.id)
            smaller_match = _TF_ATR_PATTERN.match(smaller.id)
            if not (larger_match and smaller_match):
                continue

            larger_tf = larger_match.group(1)
            smaller_tf = smaller_match.group(1)
            larger_minutes = _timeframe_to_minutes(larger_tf)
            smaller_minutes = _timeframe_to_minutes(smaller_tf)

            if larger_minutes > smaller_minutes:
                warnings.append(AtrTautologyWarning(
                    trigger_id=trigger_id,
                    expr=expr,
                    rule_type=rule_type,
                    detail=f"tf_{larger_tf}_atr > tf_{smaller_tf}_atr is always true",
                ))

    return warnings


def detect_plan_atr_tautologies(plan: StrategyPlan) -> List[AtrTautologyWarning]:
    """Check all triggers in a plan for ATR tautologies."""
    warnings: List[AtrTautologyWarning] = []
    for trigger in plan.triggers:
        for rule_type, rule_expr in [
            ("entry", trigger.entry_rule),
            ("exit", trigger.exit_rule),
            ("hold", getattr(trigger, "hold_rule", None)),
        ]:
            if rule_expr:
                warnings.extend(detect_atr_tautologies(rule_expr, trigger.id, rule_type))
    return warnings


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
    ast.In,
    ast.NotIn,
    ast.List,
    ast.Tuple,
)


_BETWEEN_TOKEN = r"(?:-?\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*)"
_BETWEEN_PATTERN = re.compile(
    rf"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s+between\s+(?P<low>{_BETWEEN_TOKEN})\s+and\s+(?P<high>{_BETWEEN_TOKEN})",
    re.IGNORECASE,
)

# Pattern to match string literals with single quotes (for position comparisons)
# This converts position == 'flat' style expressions to use boolean indicators
_POSITION_STRING_PATTERN = re.compile(
    r"\bposition\s*(==|!=)\s*['\"]?(flat|long|short)['\"]?",
    re.IGNORECASE,
)


def _normalize_position_comparisons(expr: str) -> str:
    """Convert position string comparisons to boolean indicator checks.

    This handles expressions like:
        position == 'flat'  -> is_flat
        position != 'flat'  -> not is_flat
        position == "long"  -> is_long
        position != 'short' -> not is_short

    This avoids string literal parsing issues in the AST compiler.
    """
    def replacer(match: re.Match[str]) -> str:
        op = match.group(1)
        state = match.group(2).lower()
        indicator = f"is_{state}"
        if op == "==":
            return indicator
        else:  # !=
            return f"(not {indicator})"

    return _POSITION_STRING_PATTERN.sub(replacer, expr)


def _normalize_expression(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return expr

    # First, normalize position string comparisons to boolean indicators
    expr = _normalize_position_comparisons(expr)

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
        # Boolean position indicators (preferred - avoid string comparison issues)
        "is_flat",   # True if no position
        "is_long",   # True if long position
        "is_short",  # True if short position
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


# =============================================================================
# Runbook 32-34 — Compile-Time Enforcement
# =============================================================================


@dataclass
class ExitBindingCorrection:
    """Record of an exit trigger whose category was auto-relabeled."""
    trigger_id: str
    symbol: str
    original_category: str
    corrected_category: str | None  # None if exit_rule was stripped instead


@dataclass
class HoldRuleStripped:
    """Record of a degenerate hold rule that was stripped."""
    trigger_id: str
    original_rule: str
    reason: str


@dataclass
class IdentifierCorrection:
    """Record of an identifier autocorrect or rule strip."""
    trigger_id: str
    rule_type: str  # 'entry', 'exit', 'hold'
    action: str  # 'autocorrect' or 'strip'
    identifier: str
    replacement: str | None  # None if stripped


@dataclass
class ExitRuleSanitization:
    """Record of an exit_rule that failed the normal-form check and was replaced."""
    trigger_id: str
    original_rule: str
    sanitized_rule: str
    reason: str  # 'invalid_grammar' | 'unparseable'
    stripped_identifiers: List[str]  # non-terminal identifiers found (may be empty for structure violations)


@dataclass
class TemplateViolation:
    """Record of a trigger blocked due to using identifiers outside the declared template."""
    trigger_id: str
    template_id: str
    violations: List[str]  # identifier names not in the template's allowed set


@dataclass
class PlanEnforcementResult:
    """Summary of all corrections applied by enforce_plan_quality."""
    exit_binding_corrections: List[ExitBindingCorrection] = field(default_factory=list)
    hold_rules_stripped: List[HoldRuleStripped] = field(default_factory=list)
    identifier_corrections: List[IdentifierCorrection] = field(default_factory=list)
    exit_rule_sanitizations: List[ExitRuleSanitization] = field(default_factory=list)
    template_violations: List[TemplateViolation] = field(default_factory=list)

    @property
    def total_corrections(self) -> int:
        return (
            len(self.exit_binding_corrections)
            + len(self.hold_rules_stripped)
            + len(self.identifier_corrections)
            + len(self.exit_rule_sanitizations)
            + len(self.template_violations)
        )


# Common typos mapping to correct identifiers
KNOWN_TYPOS: dict[str, str] = {
    "realization_vol_short": "realized_vol_short",
    "realization_vol_medium": "realized_vol_medium",
    "realised_vol_short": "realized_vol_short",
    "realised_vol_medium": "realized_vol_medium",
    "bollinger_mid": "bollinger_middle",
    "boll_upper": "bollinger_upper",
    "boll_lower": "bollinger_lower",
    "sma_fast": "sma_short",
    "sma_slow": "sma_long",
    "ema_fast_period": "ema_short",
    "ema_slow": "ema_long",
    "donchian_upper": "donchian_upper_short",
    "donchian_lower": "donchian_lower_short",
    "roc_fast": "roc_short",
    "roc_slow": "roc_medium",
}

# ---------------------------------------------------------------------------
# Hard invariant: exit_rule must match a strict normal form.
#
# Because exit_rule = ONLY close_position(), the grammar is:
#
#   exit_rule  ::= "not is_flat and" "(" disjunction ")"
#   disjunction ::= terminal | terminal "or" disjunction
#   terminal    ::= stop_hit | target_hit | force_flatten | below_stop | above_target
#
# Anything outside this grammar — including indicator comparisons, R-multiple
# thresholds, time gates, nested Ands, function calls — is *illegal* and will
# be replaced with the canonical safe form at compile time.
#
# In strict mode (CI / backtest pipelines) the compiler raises instead of
# silently replacing, making violations loud during development.
# ---------------------------------------------------------------------------

# The only leaf-node names permitted on the RHS of a valid exit_rule.
_EXIT_RULE_TERMINALS: frozenset[str] = frozenset({
    "stop_hit",
    "target_hit",
    "force_flatten",
    "below_stop",   # backward-compat alias for stop_hit
    "above_target",  # backward-compat alias for target_hit
})

# Broader set used by autocorrect / identifier validation (allowed anywhere in rule).
# Kept as a named export for downstream tooling.
EXIT_RULE_ALLOWLIST: frozenset[str] = frozenset({
    # Structural closure triggers — the ONLY identifiers that may gate EXIT
    "stop_hit", "target_hit", "force_flatten",
    # Backward-compat aliases
    "below_stop", "above_target",
    # Structural price levels
    "stop_price", "target_price",
    "stop_distance_pct", "target_distance_pct",
    # R-tracking (Runbook 45 — adaptive trade management)
    "current_R", "mfe_r", "mae_r",
    "r1_reached", "r2_reached", "r3_reached",
    "trade_state", "position_fraction",
    # Position state guards (always needed in exit_rule guard expressions)
    "is_flat", "is_long", "is_short",
    "position", "position_qty",
    # Time-based holds (structural — e.g., max hold bars, EOD window)
    "position_age_minutes", "position_age_hours",
    "holding_minutes", "holding_hours",
    "time_in_trade_min", "time_in_trade_hours",
})

# Canonical safe exit rule applied whenever a rule fails the normal-form check.
_CANONICAL_EXIT_RULE = "not is_flat and (stop_hit or target_hit)"


def _is_valid_exit_rhs(node: ast.expr) -> bool:
    """Return True iff *node* is a valid exit-rule RHS: a terminal or Or of terminals."""
    if isinstance(node, ast.Name):
        return node.id in _EXIT_RULE_TERMINALS
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return all(_is_valid_exit_rhs(v) for v in node.values)
    return False


def _is_exit_rule_normal_form(expr: str) -> bool:
    """Return True iff *expr* matches the canonical exit-rule grammar.

    Valid forms::

        not is_flat and (stop_hit or target_hit)
        not is_flat and (stop_hit or target_hit or force_flatten)
        not is_flat and stop_hit          # single terminal — unusual but valid
        not is_flat and target_hit

    Any comparison operator, attribute access, function call, or non-terminal
    Name node causes this to return False.
    """
    if not expr:
        return False
    try:
        tree = ast.parse(_normalize_expression(expr), mode="eval")
    except SyntaxError:
        return False
    body = tree.body
    # Top-level must be BoolOp(And, [lhs, rhs]) with exactly two operands
    if not isinstance(body, ast.BoolOp) or not isinstance(body.op, ast.And):
        return False
    if len(body.values) != 2:
        return False
    lhs, rhs = body.values
    # LHS must be: not is_flat
    if not (isinstance(lhs, ast.UnaryOp) and isinstance(lhs.op, ast.Not)):
        return False
    if not (isinstance(lhs.operand, ast.Name) and lhs.operand.id == "is_flat"):
        return False
    # RHS must be a terminal or Or of terminals — no comparisons, no nested Ands
    return _is_valid_exit_rhs(rhs)


def enforce_exit_binding(triggers: List[TriggerCondition]) -> List[ExitBindingCorrection]:
    """Fix exit triggers whose category doesn't match any entry category for that symbol.

    If exactly one entry category exists for the symbol: relabel exit trigger's category.
    If zero or multiple entry categories: strip the exit_rule from the trigger.
    """
    corrections: List[ExitBindingCorrection] = []

    # Build map of symbol -> set of categories that have entry rules
    symbol_entry_categories: dict[str, set[str]] = defaultdict(set)
    for trigger in triggers:
        if trigger.direction in ("long", "short") and trigger.entry_rule:
            symbol_entry_categories[trigger.symbol].add(trigger.category)

    for trigger in triggers:
        if not trigger.exit_rule:
            continue
        # Emergency exits bypass all constraints — never relabel or strip them
        if trigger.category == "emergency_exit":
            continue
        entry_cats = symbol_entry_categories.get(trigger.symbol, set())
        if len(entry_cats) == 1 and trigger.category in entry_cats:
            continue  # Single entry category, exact match — no fix needed
        if len(entry_cats) > 1 and trigger.category in entry_cats:
            # Multi-category symbol: exit matches one of the entry categories.
            # Mark exempt so runtime binding check passes regardless of which
            # category actually opened the position.
            object.__setattr__(trigger, "exit_binding_exempt", True)
            continue

        if len(entry_cats) == 1:
            # Exactly one entry category — relabel to match
            correct_cat = next(iter(entry_cats))
            corrections.append(ExitBindingCorrection(
                trigger_id=trigger.id,
                symbol=trigger.symbol,
                original_category=trigger.category,
                corrected_category=correct_cat,
            ))
            trigger.category = correct_cat
        elif len(entry_cats) == 0 or len(entry_cats) > 1:
            # Zero entry categories OR multiple entry categories with a non-matching
            # exit category — strip exit to avoid confusion
            corrections.append(ExitBindingCorrection(
                trigger_id=trigger.id,
                symbol=trigger.symbol,
                original_category=trigger.category,
                corrected_category=None,
            ))
            trigger.exit_rule = ""

    return corrections


def strip_degenerate_hold_rules(triggers: List[TriggerCondition]) -> List[HoldRuleStripped]:
    """Strip hold rules flagged as degenerate by detect_degenerate_hold_rules."""
    stripped: List[HoldRuleStripped] = []
    for trigger in triggers:
        if not trigger.hold_rule:
            continue
        warnings = detect_degenerate_hold_rules(trigger.hold_rule, trigger_id=trigger.id)
        if warnings:
            stripped.append(HoldRuleStripped(
                trigger_id=trigger.id,
                original_rule=trigger.hold_rule,
                reason=warnings[0],
            ))
            trigger.hold_rule = None
    return stripped


def sanitize_exit_rules(
    triggers: List[TriggerCondition],
    strict: bool = False,
) -> List[ExitRuleSanitization]:
    """Enforce the exit-rule normal-form constraint (hard invariant).

    Because ``exit_rule`` *only* calls ``close_position()``, the valid grammar is::

        not is_flat and (stop_hit or target_hit [or force_flatten])

    Any rule that does not match this grammar — including:

    * indicator conditions (``rsi_14 > 75``, ``macd_hist < 0``)
    * comparison expressions on allowed identifiers (``position_age_minutes > 5``,
      ``current_R < 0.1``, ``stop_distance_pct < 0.001``)
    * nested boolean structures (``stop_hit and position_qty > 0``)
    * function calls, attribute access, subscripts

    will be replaced with the canonical safe form::

        "not is_flat and (stop_hit or target_hit)"

    Emergency exits are exempt (safety-critical; bypass all enforcement).

    Advisory indicator logic (MACD weakening → tighten stop, etc.) belongs in
    ``hold_rule`` or ``risk_reduce_rule``, NOT ``exit_rule``.

    Args:
        triggers: List of triggers to sanitize in-place.
        strict: If ``True``, raise :exc:`TriggerCompilationError` instead of
                silently replacing.  Intended for CI / backtest pipelines where
                violations should fail the build.

    Returns:
        List of :class:`ExitRuleSanitization` records (one per sanitized rule).
    """
    sanitizations: List[ExitRuleSanitization] = []

    for trigger in triggers:
        if not trigger.exit_rule:
            continue
        # Emergency exits bypass all enforcement (safety critical)
        if trigger.category == "emergency_exit":
            continue

        expr = trigger.exit_rule

        # Check AST parseability first
        try:
            ast.parse(_normalize_expression(expr), mode="eval")
            parseable = True
        except SyntaxError:
            parseable = False

        if parseable and _is_exit_rule_normal_form(expr):
            # Rule is already in canonical form — nothing to do.
            continue

        # Rule fails normal-form check.  Collect non-terminal identifiers for telemetry.
        all_ids = _collect_identifiers(expr)
        non_terminals = sorted(
            name for name in all_ids
            if name not in _EXIT_RULE_TERMINALS and name != "is_flat"
        )
        reason = "unparseable" if not parseable else "invalid_grammar"
        sanitized = _CANONICAL_EXIT_RULE

        if strict:
            raise TriggerCompilationError(
                f"Trigger '{trigger.id}' exit_rule fails normal-form constraint "
                f"({reason}). "
                f"Only 'not is_flat and (stop_hit or target_hit [or force_flatten])' "
                f"is permitted. Original rule: {expr!r}"
            )

        sanitizations.append(ExitRuleSanitization(
            trigger_id=trigger.id,
            original_rule=expr,
            sanitized_rule=sanitized,
            reason=reason,
            stripped_identifiers=non_terminals,
        ))
        trigger.exit_rule = sanitized

    return sanitizations


def tighten_stop_only(
    current_stop: float,
    proposed_stop: float,
    side: Literal["long", "short"],
) -> float:
    """Enforce monotonic stop advancement — stops may only tighten, never widen.

    For longs: stop may only move UP (raises the protective floor).
    For shorts: stop may only move DOWN (lowers the protective ceiling).

    Args:
        current_stop: The current stop price.
        proposed_stop: The proposed new stop price from an advisory action.
        side: Trade direction — ``'long'`` or ``'short'``.

    Returns:
        The tighter of *current_stop* and *proposed_stop*.
    """
    if side == "long":
        return max(current_stop, proposed_stop)
    else:
        return min(current_stop, proposed_stop)


def autocorrect_identifiers(
    plan: StrategyPlan,
    available_timeframes: Set[str],
) -> List[IdentifierCorrection]:
    """Fix known typos in trigger expressions and strip rules with unresolvable identifiers."""
    corrections: List[IdentifierCorrection] = []
    allowed = build_allowed_identifiers(available_timeframes)
    # Extend allowed with known-typo replacements so we can detect them
    typo_keys = set(KNOWN_TYPOS.keys())

    for trigger in plan.triggers:
        # Emergency exits are safety-critical — autocorrect typos but never strip rules
        is_emergency = trigger.category == "emergency_exit"
        for rule_type, attr in (("entry", "entry_rule"), ("exit", "exit_rule"), ("hold", "hold_rule")):
            expr = getattr(trigger, attr, None)
            if not expr:
                continue

            identifiers = _collect_identifiers(expr)
            fixed_expr = expr

            # Pass 1: autocorrect known typos
            for ident in identifiers:
                if ident in typo_keys:
                    replacement = KNOWN_TYPOS[ident]
                    fixed_expr = re.sub(rf"\b{re.escape(ident)}\b", replacement, fixed_expr)
                    corrections.append(IdentifierCorrection(
                        trigger_id=trigger.id,
                        rule_type=rule_type,
                        action="autocorrect",
                        identifier=ident,
                        replacement=replacement,
                    ))

            # Pass 2: check for remaining unknown identifiers after autocorrect
            post_fix_ids = _collect_identifiers(fixed_expr)
            unknown = {
                name for name in post_fix_ids
                if name not in allowed and name.lower() not in {"true", "false", "none"}
            }

            if unknown and not is_emergency:
                # Strip the entire rule — can't safely evaluate with unknown identifiers
                # (emergency exits are never stripped — safety critical)
                for ident in sorted(unknown):
                    corrections.append(IdentifierCorrection(
                        trigger_id=trigger.id,
                        rule_type=rule_type,
                        action="strip",
                        identifier=ident,
                        replacement=None,
                    ))
                fixed_expr = ""
            elif unknown and is_emergency:
                logger.warning(
                    "Emergency exit '%s' %s_rule has unknown identifiers %s — keeping rule intact",
                    trigger.id, rule_type, sorted(unknown),
                )

            if fixed_expr != expr:
                setattr(trigger, attr, fixed_expr)

    return corrections


# Identifiers that are always valid regardless of which template is declared.
# These cover position state, price, time-in-trade, exit terminals, and metadata
# that every trigger may legitimately reference.
_TEMPLATE_UNIVERSAL_ALLOWED: frozenset[str] = frozenset({
    # Position state
    "is_flat", "is_long", "is_short", "position", "position_qty", "position_value",
    "entry_price", "avg_entry_price", "entry_side", "position_opened_at",
    # PnL and time-in-trade
    "unrealized_pnl_pct", "unrealized_pnl_abs", "unrealized_pnl",
    "position_pnl_pct", "position_pnl_abs",
    "position_age_minutes", "position_age_hours",
    "holding_minutes", "holding_hours",
    "time_in_trade_min", "time_in_trade_hours",
    # Exit terminals (always valid — stop/target/flatten semantics)
    "stop_hit", "target_hit", "force_flatten", "below_stop", "above_target",
    # Market structure
    "nearest_support", "nearest_resistance",
    "distance_to_support_pct", "distance_to_resistance_pct",
    "trend", "recent_tests",
    # State labels
    "trend_state", "vol_state",
    # Symbol / timeframe metadata
    "symbol", "timeframe",
    # Raw price / volume (always available)
    "close", "open", "high", "low", "volume",
})


def enforce_template_identifiers(
    plan: StrategyPlan,
) -> List[TemplateViolation]:
    """Block triggers whose identifiers are not in the declared template's allowed set.

    Emergency exits are exempt (same rule as other enforcement passes).
    Returns the list of violations; mutates plan.triggers in-place to remove blocked ones.
    If TEMPLATE_ENFORCEMENT_ENABLED=false, runs but returns violations without mutating.
    """
    import os as _os
    template_id = plan.template_id
    if not template_id:
        return []

    enforcement_enabled = _os.environ.get("TEMPLATE_ENFORCEMENT_ENABLED", "true").strip().lower() not in {
        "0", "false", "no"
    }

    template_allowed = _vs_allowed_identifiers(template_id)
    if not template_allowed:
        logger.warning(
            "Template '%s' not found in vector store; skipping template enforcement (fail open)",
            template_id,
        )
        return []

    effective_allowed = template_allowed | _TEMPLATE_UNIVERSAL_ALLOWED

    violations: List[TemplateViolation] = []
    kept: List[TriggerCondition] = []
    for trigger in plan.triggers:
        if trigger.category == "emergency_exit":
            kept.append(trigger)
            continue
        bad: List[str] = []
        for expr in (trigger.entry_rule, trigger.hold_rule):
            if not expr:
                continue
            for name in _collect_identifiers(expr):
                if name.lower() in {"true", "false", "none"}:
                    continue
                if name not in effective_allowed:
                    bad.append(name)
        if bad:
            violations.append(TemplateViolation(
                trigger_id=trigger.id,
                template_id=template_id,
                violations=sorted(set(bad)),
            ))
        else:
            kept.append(trigger)

    if enforcement_enabled and violations:
        plan.triggers = kept

    return violations


def enforce_plan_quality(
    plan: StrategyPlan,
    available_timeframes: Set[str],
    strict: bool = False,
) -> PlanEnforcementResult:
    """Run all compile-time enforcement checks before compile_plan.

    Mutates triggers in-place and returns a summary of corrections.

    Args:
        plan: The strategy plan to enforce.
        available_timeframes: Timeframes present in the loaded data.
        strict: If ``True``, raise :exc:`TriggerCompilationError` on any
                exit_rule that fails the normal-form constraint instead of
                silently sanitizing it.  Use this in CI / backtest pipelines
                to make violations loud during development.
    """
    result = PlanEnforcementResult()

    # Run identifier autocorrect FIRST so exit binding sees accurate entry rules
    result.identifier_corrections = autocorrect_identifiers(plan, available_timeframes)
    for ic in result.identifier_corrections:
        if ic.action == "autocorrect":
            logger.warning(
                "Trigger '%s' %s_rule: autocorrected '%s' → '%s'",
                ic.trigger_id, ic.rule_type, ic.identifier, ic.replacement,
            )
        else:
            logger.warning(
                "Trigger '%s' %s_rule: stripped — unknown identifier '%s'",
                ic.trigger_id, ic.rule_type, ic.identifier,
            )

    # Exit binding runs AFTER identifiers so it sees accurate entry rules
    result.exit_binding_corrections = enforce_exit_binding(plan.triggers)
    for c in result.exit_binding_corrections:
        if c.corrected_category:
            logger.warning(
                "Trigger '%s' (%s): exit category relabeled from '%s' to '%s' to match entry",
                c.trigger_id, c.symbol, c.original_category, c.corrected_category,
            )
        else:
            logger.warning(
                "Trigger '%s' (%s): exit_rule stripped — category '%s' has no matching entry",
                c.trigger_id, c.symbol, c.original_category,
            )

    result.hold_rules_stripped = strip_degenerate_hold_rules(plan.triggers)
    for h in result.hold_rules_stripped:
        logger.warning(
            "Trigger '%s': degenerate hold rule stripped — %s",
            h.trigger_id, h.reason,
        )

    # Exit-rule sanitizer runs LAST — after autocorrect so it sees clean identifiers.
    # This is the hard invariant: only stop/target/force_flatten may close a position.
    result.exit_rule_sanitizations = sanitize_exit_rules(plan.triggers, strict=strict)
    for s in result.exit_rule_sanitizations:
        extra = f" (non-terminal identifiers: {', '.join(s.stripped_identifiers)})" if s.stripped_identifiers else ""
        logger.warning(
            "Trigger '%s' exit_rule SANITIZED [%s]%s. "
            "Original: %r → Canonical: %r",
            s.trigger_id,
            s.reason,
            extra,
            s.original_rule,
            s.sanitized_rule,
        )

    # Template enforcement runs after all other passes so it sees fully corrected triggers.
    result.template_violations = enforce_template_identifiers(plan)
    for v in result.template_violations:
        logger.warning(
            "Trigger '%s' BLOCKED [template_identifier_violation] "
            "template='%s' violating_identifiers=%s",
            v.trigger_id,
            v.template_id,
            v.violations,
        )

    if result.total_corrections:
        logger.info(
            "Plan enforcement: %d corrections applied "
            "(%d exit-binding, %d hold-rule, %d identifier, %d exit-sanitized, %d template-violations)",
            result.total_corrections,
            len(result.exit_binding_corrections),
            len(result.hold_rules_stripped),
            len(result.identifier_corrections),
            len(result.exit_rule_sanitizations),
            len(result.template_violations),
        )

    return result

"""Minimal DSL evaluator for LLM-provided trigger rules."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Mapping


class RuleSyntaxError(ValueError):
    """Raised when a rule contains unsupported syntax."""


@dataclass
class EvaluationTrace:
    """Debug trace of a rule evaluation showing intermediate values."""
    expression: str
    result: bool
    context_values: dict[str, Any] = field(default_factory=dict)
    sub_results: list[tuple[str, Any]] = field(default_factory=list)
    error: str | None = None


_BETWEEN_PATTERN = re.compile(
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s+between\s+(?P<low>-?\d+\.?\d*)\s+and\s+(?P<high>-?\d+\.?\d*)",
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

    This avoids string literal parsing issues in the AST.
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


class MissingIndicatorError(RuleSyntaxError):
    """Raised when a required indicator is missing from the context."""


class RuleEvaluator:
    """Safe expression evaluator supporting comparisons and boolean ops."""

    def __init__(self, allowed_names: set[str] | None = None) -> None:
        self.allowed_names = allowed_names or set()

    def _normalize(self, expr: str) -> str:
        # First, normalize position string comparisons to boolean indicators
        expr = _normalize_position_comparisons(expr)

        def replacer(match: re.Match[str]) -> str:
            field = match.group("field")
            low = match.group("low")
            high = match.group("high")
            return f"(({field}) >= ({low}) and ({field}) <= ({high}))"

        return _BETWEEN_PATTERN.sub(replacer, expr)

    def evaluate(self, expr: str, context: Mapping[str, Any]) -> bool:
        if not expr:
            return False
        normalized = self._normalize(expr)
        try:
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError as exc:
            raise RuleSyntaxError(f"invalid rule syntax: {expr}") from exc
        allowed = set(context.keys()) | self.allowed_names | {"True", "False", "true", "false", "None", "none"}
        return bool(self._eval_node(tree.body, context, allowed))

    def _eval_node(self, node: ast.AST, ctx: Mapping[str, Any], allowed: set[str]) -> Any:
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, ctx, allowed)
            right = self._eval_node(node.right, ctx, allowed)
            return self._eval_binop(node.op, left, right)
        if isinstance(node, ast.BoolOp):
            values = [self._eval_node(value, ctx, allowed) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise RuleSyntaxError("unsupported boolean operator")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self._eval_node(node.operand, ctx, allowed)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._eval_node(node.operand, ctx, allowed)
            if isinstance(value, (int, float)):
                return value if isinstance(node.op, ast.UAdd) else -value
            return None
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, ctx, allowed)
            result = True
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, ctx, allowed)
                result = result and self._compare(op, left, right)
                left = right
            return result
        if isinstance(node, ast.Name):
            name = node.id
            lower = name.lower()
            if lower in {"true", "false"}:
                return lower == "true"
            if lower == "none":
                return None
            if name not in allowed and lower not in allowed:
                raise RuleSyntaxError(f"unknown identifier '{node.id}' in rule")
            if name not in ctx:
                raise MissingIndicatorError(f"missing indicator '{name}' in rule")
            return ctx.get(name)
        if isinstance(node, ast.Subscript):
            base = self._eval_node(node.value, ctx, allowed)
            if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int):
                raise RuleSyntaxError(f"unsupported subscript in rule: {ast.dump(node)}")
            if node.slice.value < 0:
                raise RuleSyntaxError(f"negative index not allowed in rule: {ast.dump(node)}")
            if base is None:
                return None
            try:
                return base[node.slice.value]
            except Exception:
                return None
        if isinstance(node, ast.Attribute):
            base = self._eval_node(node.value, ctx, allowed)
            if base is None:
                return None
            if isinstance(base, Mapping):
                return base.get(node.attr)
            return getattr(base, node.attr, None)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(elt, ctx, allowed) for elt in node.elts]
        raise RuleSyntaxError(f"unsupported expression in rule: {ast.dump(node)}")

    def _compare(self, op: ast.cmpop, left: Any, right: Any) -> bool:
        if left is None or right is None:
            return False
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.In):
            return left in right
        if isinstance(op, ast.NotIn):
            return left not in right
        raise RuleSyntaxError("unsupported comparison operator")

    def _eval_binop(self, op: ast.operator, left: Any, right: Any) -> Any:
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                return None
            return left / right
        raise RuleSyntaxError("unsupported arithmetic operator")

    def evaluate_with_trace(self, expr: str, context: Mapping[str, Any]) -> EvaluationTrace:
        """Evaluate an expression and return detailed trace for debugging.

        Returns an EvaluationTrace with:
        - The original expression
        - The final result (True/False)
        - Context values for all referenced identifiers
        - Sub-results for each condition in the expression
        - Any error message if evaluation failed
        """
        trace = EvaluationTrace(expression=expr, result=False)

        if not expr:
            return trace

        # Extract identifier names from expression for context values
        try:
            normalized = self._normalize(expr)
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError as exc:
            trace.error = f"Syntax error: {exc}"
            return trace

        # Collect all Name nodes (identifiers) referenced in the expression
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)

        # Store context values for referenced identifiers
        for ident in identifiers:
            lower = ident.lower()
            if lower in {"true", "false", "none"}:
                continue
            if ident in context:
                trace.context_values[ident] = context.get(ident)
            else:
                trace.context_values[ident] = "<missing>"

        # Parse individual conditions for sub-results
        trace.sub_results = self._extract_conditions(tree.body, context)

        # Evaluate the full expression
        try:
            allowed = set(context.keys()) | self.allowed_names | {"True", "False", "true", "false", "None", "none"}
            trace.result = bool(self._eval_node(tree.body, context, allowed))
        except (RuleSyntaxError, MissingIndicatorError) as exc:
            trace.error = str(exc)
            trace.result = False

        return trace

    def _extract_conditions(self, node: ast.AST, ctx: Mapping[str, Any]) -> list[tuple[str, Any]]:
        """Extract individual conditions from an AST node for debugging."""
        results = []

        if isinstance(node, ast.BoolOp):
            # For AND/OR, recurse into each value
            for value in node.values:
                results.extend(self._extract_conditions(value, ctx))
        elif isinstance(node, ast.Compare):
            # Format the comparison as a readable string
            try:
                allowed = set(ctx.keys()) | self.allowed_names | {"True", "False", "true", "false", "None", "none"}
                left_val = self._eval_node(node.left, ctx, allowed)
                condition_parts = [f"{self._node_to_str(node.left)}={left_val}"]

                result = True
                for op, comparator in zip(node.ops, node.comparators):
                    right_val = self._eval_node(comparator, ctx, allowed)
                    op_str = self._op_to_str(op)
                    condition_parts.append(f"{op_str} {self._node_to_str(comparator)}={right_val}")
                    result = result and self._compare(op, left_val, right_val)
                    left_val = right_val

                condition_str = " ".join(condition_parts)
                results.append((condition_str, result))
            except Exception as e:
                results.append((f"<error: {e}>", False))
        else:
            # For other nodes, just try to evaluate and show result
            try:
                allowed = set(ctx.keys()) | self.allowed_names | {"True", "False", "true", "false", "None", "none"}
                val = self._eval_node(node, ctx, allowed)
                results.append((self._node_to_str(node), val))
            except Exception as e:
                results.append((f"<error: {e}>", None))

        return results

    def _node_to_str(self, node: ast.AST) -> str:
        """Convert an AST node back to a readable string."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.BinOp):
            left = self._node_to_str(node.left)
            right = self._node_to_str(node.right)
            op = self._binop_to_str(node.op)
            return f"({left} {op} {right})"
        if isinstance(node, ast.UnaryOp):
            operand = self._node_to_str(node.operand)
            if isinstance(node.op, ast.USub):
                return f"-{operand}"
            if isinstance(node.op, ast.UAdd):
                return f"+{operand}"
            return f"not {operand}"
        if isinstance(node, ast.Attribute):
            base = self._node_to_str(node.value)
            return f"{base}.{node.attr}"
        return "<expr>"

    def _op_to_str(self, op: ast.cmpop) -> str:
        """Convert comparison operator to string."""
        if isinstance(op, ast.Gt):
            return ">"
        if isinstance(op, ast.GtE):
            return ">="
        if isinstance(op, ast.Lt):
            return "<"
        if isinstance(op, ast.LtE):
            return "<="
        if isinstance(op, ast.Eq):
            return "=="
        if isinstance(op, ast.NotEq):
            return "!="
        return "?"

    def _binop_to_str(self, op: ast.operator) -> str:
        """Convert binary operator to string."""
        if isinstance(op, ast.Add):
            return "+"
        if isinstance(op, ast.Sub):
            return "-"
        if isinstance(op, ast.Mult):
            return "*"
        if isinstance(op, ast.Div):
            return "/"
        return "?"

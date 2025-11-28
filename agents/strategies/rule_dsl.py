"""Minimal DSL evaluator for LLM-provided trigger rules."""

from __future__ import annotations

import ast
import re
from typing import Any, Mapping


class RuleSyntaxError(ValueError):
    """Raised when a rule contains unsupported syntax."""


_BETWEEN_PATTERN = re.compile(
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s+between\s+(?P<low>-?\d+\.?\d*)\s+and\s+(?P<high>-?\d+\.?\d*)",
    re.IGNORECASE,
)


class RuleEvaluator:
    """Safe expression evaluator supporting comparisons and boolean ops."""

    def __init__(self, allowed_names: set[str] | None = None) -> None:
        self.allowed_names = allowed_names or set()

    def _normalize(self, expr: str) -> str:
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
        allowed = set(context.keys()) | self.allowed_names | {"True", "False"}
        return bool(self._eval_node(tree.body, context, allowed))

    def _eval_node(self, node: ast.AST, ctx: Mapping[str, Any], allowed: set[str]) -> Any:
        if isinstance(node, ast.BoolOp):
            values = [self._eval_node(value, ctx, allowed) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise RuleSyntaxError("unsupported boolean operator")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self._eval_node(node.operand, ctx, allowed)
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, ctx, allowed)
            result = True
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, ctx, allowed)
                result = result and self._compare(op, left, right)
                left = right
            return result
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                raise RuleSyntaxError(f"unknown identifier '{node.id}' in rule")
            if node.id in {"True", "False"}:
                return node.id == "True"
            return ctx.get(node.id)
        if isinstance(node, ast.Constant):
            return node.value
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
        raise RuleSyntaxError("unsupported comparison operator")

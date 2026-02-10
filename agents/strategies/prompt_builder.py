"""Prompt context assembly for the strategist."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

from schemas.llm_strategist import LLMInput
from trading_core.rule_registry import format_allowed_identifiers


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return str(value).strip() or None


def _section(title: str, body: str | None) -> str | None:
    if not body:
        return None
    return f"{title}:\n{body}"


def _format_list(items: Iterable[str]) -> str | None:
    items = [item.strip() for item in items if item and str(item).strip()]
    if not items:
        return None
    return ", ".join(items)


def build_prompt_context(llm_input: LLMInput) -> str:
    """Build dynamic prompt context injected into the strategist system prompt."""
    ctx: Dict[str, Any] = llm_input.global_context or {}
    sections: List[str] = []

    available_timeframes = ctx.get("available_timeframes") or []
    timeframes_str = _format_list(available_timeframes)
    if timeframes_str:
        sections.append(f"AVAILABLE_TIMEFRAMES:\n{timeframes_str}")

    strategy_guidance = _stringify(ctx.get("strategy_guidance") or ctx.get("strategy_profile"))
    if not strategy_guidance:
        strategy_guidance = _stringify(os.environ.get("LLM_STRATEGIST_PROMPT"))
    if strategy_guidance:
        sections.append(_section("STRATEGY_GUIDANCE", strategy_guidance) or "")

    regime_guidance = _stringify(ctx.get("regime_recommendations") or ctx.get("regime_guidance"))
    if regime_guidance:
        sections.append(_section("REGIME_RECOMMENDATIONS", regime_guidance) or "")

    allowed_block = format_allowed_identifiers(available_timeframes)
    sections.append(_section("ALLOWED_RULE_IDENTIFIERS", allowed_block) or "")

    return "\n\n".join(section for section in sections if section)


__all__ = ["build_prompt_context"]

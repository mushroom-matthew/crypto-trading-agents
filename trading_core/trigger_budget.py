"""Utilities for normalizing and capping strategist triggers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from schemas.llm_strategist import StrategyPlan, TriggerCondition

GRADE_SCORE = {"A": 3, "B": 2, "C": 1, None: 0}


def _score_trigger(trigger: TriggerCondition) -> Tuple[int, int, str]:
    grade = GRADE_SCORE.get(trigger.confidence_grade, 0)
    entry_len = len(trigger.entry_rule or "")
    return (grade, -entry_len, trigger.id)


def _dedupe_near_duplicates(triggers: Iterable[TriggerCondition]) -> List[TriggerCondition]:
    """Collapse triggers that only differ by small textual tweaks."""

    best: Dict[Tuple[str | None, str, str], TriggerCondition] = {}
    for trigger in triggers:
        key = (trigger.category or "other", trigger.direction, trigger.timeframe)
        current = best.get(key)
        if current is None or _score_trigger(trigger) > _score_trigger(current):
            best[key] = trigger
    return list(best.values())


def enforce_trigger_budget(
    plan: StrategyPlan,
    default_cap: int = 6,
    fallback_symbols: Iterable[str] | None = None,
    resolved_cap: int | None = None,
) -> tuple[StrategyPlan, Dict[str, int]]:
    """Return a new StrategyPlan with triggers trimmed to declared budgets."""

    symbol_caps = {symbol: max(0, int(cap)) for symbol, cap in (plan.trigger_budgets or {}).items() if cap is not None}
    stats: Dict[str, int] = {}
    grouped: Dict[str, List[TriggerCondition]] = defaultdict(list)
    for trigger in plan.triggers:
        grouped[trigger.symbol].append(trigger)

    if fallback_symbols:
        for symbol in fallback_symbols:
            grouped.setdefault(symbol, [])

    trimmed_triggers: List[TriggerCondition] = []
    cap_source = resolved_cap if resolved_cap is not None else plan.max_triggers_per_symbol_per_day
    global_cap = max(0, cap_source or default_cap or 0)

    for symbol, triggers in grouped.items():
        unique = _dedupe_near_duplicates(triggers)
        cap = symbol_caps.get(symbol)
        if cap is None:
            cap = global_cap
        elif global_cap:
            cap = min(cap, global_cap)
        if cap <= 0:
            stats[symbol] = len(unique)
            continue
        ranked = sorted(unique, key=_score_trigger, reverse=True)
        selected = ranked[:cap]
        stats[symbol] = max(0, len(unique) - len(selected))
        trimmed_triggers.extend(selected)

    new_plan = plan.model_copy(deep=True)
    new_plan.triggers = trimmed_triggers
    return new_plan, stats


__all__ = ["enforce_trigger_budget"]

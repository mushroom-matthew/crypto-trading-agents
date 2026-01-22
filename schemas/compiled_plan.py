"""Pydantic models for compiled strategy plans."""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from .llm_strategist import SerializableModel, TriggerCategory, TriggerDirection


class CompiledExpression(SerializableModel):
    """Normalized expression string produced by the compiler."""

    source: str
    normalized: str


class CompiledTrigger(SerializableModel):
    """Trigger metadata plus compiled entry/exit expressions."""

    trigger_id: str
    symbol: str
    direction: TriggerDirection
    category: TriggerCategory | None = None
    entry: Optional[CompiledExpression] = None
    exit: Optional[CompiledExpression] = None


class CompiledPlan(SerializableModel):
    """Serialized compiled plan artifact keyed by StrategyPlan/Run IDs."""

    plan_id: str
    run_id: str
    triggers: List[CompiledTrigger]


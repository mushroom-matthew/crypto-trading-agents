"""Registry of rule identifiers allowed in strategist trigger expressions."""

from __future__ import annotations

import re
from typing import Iterable, List, Set

from schemas.llm_strategist import IndicatorSnapshot


_NUMERIC_SUFFIX = re.compile(r"^(?P<base>[A-Za-z_]+)_(?P<digits>\d+)$")


def _indicator_fields() -> List[str]:
    fields = list(IndicatorSnapshot.model_fields.keys())
    # Exclude metadata-only fields
    fields = [field for field in fields if field not in {"as_of"}]
    return fields


def _numeric_aliases(fields: Iterable[str]) -> Set[str]:
    aliases: Set[str] = set()
    for field in fields:
        match = _NUMERIC_SUFFIX.match(field)
        if match:
            aliases.add(match.group("base"))
    return aliases


def _position_fields() -> List[str]:
    return [
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
        "position_age_minutes",
        "position_age_hours",
        "holding_minutes",
        "holding_hours",
        "time_in_trade_min",
        "time_in_trade_hours",
        "unrealized_pnl_pct",
        "unrealized_pnl_abs",
        "unrealized_pnl",
        "position_pnl_pct",
        "position_pnl_abs",
    ]


def _derived_fields() -> List[str]:
    return [
        "trend_state",
        "vol_state",
    ]


def _market_structure_fields() -> List[str]:
    return [
        "nearest_support",
        "nearest_resistance",
        "distance_to_support_pct",
        "distance_to_resistance_pct",
        "trend",
        "recent_tests",
    ]


def base_allowed_identifiers() -> Set[str]:
    indicator_fields = _indicator_fields()
    allowed = set(indicator_fields)
    allowed.update(_position_fields())
    allowed.update(_derived_fields())
    allowed.update(_market_structure_fields())
    allowed.update(_numeric_aliases(indicator_fields))
    return allowed


def allowed_identifiers(available_timeframes: Iterable[str] | None = None) -> Set[str]:
    """Return allowed identifiers, including cross-timeframe aliases."""
    allowed = set(base_allowed_identifiers())
    timeframes = list(available_timeframes or [])
    if not timeframes:
        return allowed
    indicator_fields = _indicator_fields()
    alias_fields = _numeric_aliases(indicator_fields)
    extra_fields = set(_derived_fields()) | alias_fields
    for tf in timeframes:
        prefix = f"tf_{tf.replace('-', '_')}"
        for field in indicator_fields:
            allowed.add(f"{prefix}_{field}")
        for field in extra_fields:
            allowed.add(f"{prefix}_{field}")
    return allowed


def format_allowed_identifiers(available_timeframes: Iterable[str] | None = None) -> str:
    identifiers = sorted(allowed_identifiers(available_timeframes))
    return "\n".join(f"- {identifier}" for identifier in identifiers)


__all__ = [
    "base_allowed_identifiers",
    "allowed_identifiers",
    "format_allowed_identifiers",
]

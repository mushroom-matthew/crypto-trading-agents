"""R96: Canonical trigger catalog — 9 archetypes with validated entry-rule templates.

The LLM selects archetypes and fills parameter slots rather than composing
free-form entry_rule strings. This prevents quality errors like backwards
bb_bandwidth conditions, missing targets, or wrong-timeframe indicators.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, model_validator

from schemas.llm_strategist import SerializableModel

# ── Archetype vocabulary ────────────────────────────────────────────────────

TriggerArchetype = Literal[
    "mean_reversion_long",
    "mean_reversion_short",
    "breakout_long",
    "breakout_short",
    "momentum_long",
    "momentum_short",
    "emergency_exit",
    "profit_take_exit",
    "stop_loss_exit",
]

ARCHETYPE_DIRECTIONS: dict[str, str] = {
    "mean_reversion_long": "long",
    "mean_reversion_short": "short",
    "breakout_long": "long",
    "breakout_short": "short",
    "momentum_long": "long",
    "momentum_short": "short",
    "emergency_exit": "exit",
    "profit_take_exit": "exit",
    "stop_loss_exit": "exit",
}

ARCHETYPE_CATEGORIES: dict[str, str] = {
    "mean_reversion_long": "mean_reversion",
    "mean_reversion_short": "mean_reversion",
    "breakout_long": "volatility_breakout",
    "breakout_short": "volatility_breakout",
    "momentum_long": "trend_continuation",
    "momentum_short": "trend_continuation",
    "emergency_exit": "emergency_exit",
    "profit_take_exit": "exit",
    "stop_loss_exit": "exit",
}

# For each archetype, the canonical entry_rule template and required/default params.
# Template slots like {rsi_oversold} are filled from TriggerInstance.params.
ARCHETYPE_SPECS: dict[str, dict[str, Any]] = {
    "mean_reversion_long": {
        "entry_rule_template": "is_flat and rsi_14 < {rsi_oversold} and close <= bollinger_lower",
        "exit_rule": "not is_flat and (stop_hit or target_hit)",
        "required_params": ["stop_loss_pct"],
        "defaults": {"rsi_oversold": 30, "target_anchor_type": "bollinger_middle"},
        "stop_anchor_type": "atr",
    },
    "mean_reversion_short": {
        "entry_rule_template": "is_flat and rsi_14 > {rsi_overbought} and close >= bollinger_upper",
        "exit_rule": "not is_flat and (stop_hit or target_hit)",
        "required_params": ["stop_loss_pct"],
        "defaults": {"rsi_overbought": 70, "target_anchor_type": "bollinger_middle"},
        "stop_anchor_type": "atr",
    },
    "breakout_long": {
        "entry_rule_template": (
            "is_flat and breakout_confirmed > 0.0 and close > donchian_upper_short"
            " and volume_multiple > {volume_threshold}"
        ),
        "exit_rule": "not is_flat and (stop_hit or target_hit)",
        "required_params": [],
        "defaults": {
            "volume_threshold": 0.15,
            "stop_anchor_type": "donchian_extreme",
            "target_anchor_type": "htf_5d_extreme",
        },
    },
    "breakout_short": {
        "entry_rule_template": (
            "is_flat and breakout_confirmed > 0.0 and close < donchian_lower_short"
            " and volume_multiple > {volume_threshold}"
        ),
        "exit_rule": "not is_flat and (stop_hit or target_hit)",
        "required_params": [],
        "defaults": {
            "volume_threshold": 0.15,
            "stop_anchor_type": "donchian_extreme",
            "target_anchor_type": "htf_5d_extreme",
        },
    },
    "momentum_long": {
        "entry_rule_template": "is_flat and {entry_condition}",
        "exit_rule": "not is_flat and (stop_hit or target_hit)",
        "required_params": ["entry_condition", "stop_loss_pct"],
        "defaults": {"target_anchor_type": "r_multiple_2"},
    },
    "momentum_short": {
        "entry_rule_template": "is_flat and {entry_condition}",
        "exit_rule": "not is_flat and (stop_hit or target_hit)",
        "required_params": ["entry_condition", "stop_loss_pct"],
        "defaults": {"target_anchor_type": "r_multiple_2"},
    },
    "emergency_exit": {
        "entry_rule_template": "not is_flat and ({exit_condition})",
        "exit_rule_template": "not is_flat and ({exit_condition})",
        "required_params": [],
        "defaults": {"exit_condition": "vol_state == 'extreme' or unrealized_pnl_pct < -6.0"},
    },
    "profit_take_exit": {
        "entry_rule": "not is_flat and target_hit",
        "exit_rule": "not is_flat and target_hit",
        "required_params": [],
        "defaults": {},
    },
    "stop_loss_exit": {
        "entry_rule": "not is_flat and stop_hit",
        "exit_rule": "not is_flat and stop_hit",
        "required_params": [],
        "defaults": {},
    },
}


# ── TriggerInstance ─────────────────────────────────────────────────────────

class TriggerInstance(SerializableModel):
    """A catalog archetype instantiated for a specific symbol with filled params."""

    instance_id: str = Field(
        description="Stable ID: '{archetype_id}:{symbol}:{session_id[:8]}'. "
                    "Deterministic and deduplicable across replans.",
    )
    archetype_id: TriggerArchetype
    symbol: str
    timeframe: str = "5m"
    params: Dict[str, Any] = Field(default_factory=dict)
    confidence_grade: Literal["A", "B", "C"] = "B"
    state: Literal["active", "pending", "removed"] = "active"
    created_at_cycle: int = 0
    bound_position_id: Optional[str] = None


def make_instance_id(archetype_id: str, symbol: str, session_id: str) -> str:
    return f"{archetype_id}:{symbol}:{session_id[:8]}"


# ── TriggerDiff ─────────────────────────────────────────────────────────────

class TriggerDiff(SerializableModel):
    """Incremental mutation from the LLM — add/remove/modify against the active registry.

    The LLM outputs this instead of a full trigger list. The registry applies the diff,
    preserving any trigger bound to an open position (POSITION_OPEN guard).
    """

    to_add: List[TriggerInstance] = Field(default_factory=list)
    to_remove: List[str] = Field(
        default_factory=list,
        description="instance_ids to deactivate. Ignored for exit triggers when POSITION_OPEN.",
    )
    to_modify: List[TriggerInstance] = Field(
        default_factory=list,
        description="Updated instances (matched by instance_id). "
                    "Ignored for exit triggers when POSITION_OPEN.",
    )


# ── Archetype → TriggerCondition conversion ─────────────────────────────────

def instance_to_trigger_condition(inst: TriggerInstance) -> dict[str, Any]:
    """Convert a TriggerInstance to a TriggerCondition-compatible dict.

    Uses the archetype spec to produce a validated entry_rule from the instance params.
    Missing optional params fall back to archetype defaults.
    """
    spec = ARCHETYPE_SPECS[inst.archetype_id]
    params = {**spec.get("defaults", {}), **inst.params}

    # Build entry_rule
    if "entry_rule_template" in spec:
        try:
            entry_rule = spec["entry_rule_template"].format(**params)
        except KeyError as exc:
            # Missing required template slot — use a safe fallback
            entry_rule = f"is_flat and False  # missing param {exc}"
    else:
        entry_rule = spec["entry_rule"]

    # Build exit_rule
    if "exit_rule_template" in spec:
        try:
            exit_rule = spec["exit_rule_template"].format(**params)
        except KeyError:
            exit_rule = spec.get("exit_rule", "not is_flat and stop_hit")
    else:
        exit_rule = spec.get("exit_rule", "not is_flat and stop_hit")

    direction = ARCHETYPE_DIRECTIONS[inst.archetype_id]
    category = ARCHETYPE_CATEGORIES[inst.archetype_id]

    tc: dict[str, Any] = {
        "id": inst.instance_id,
        "symbol": inst.symbol,
        "direction": direction,
        "category": category,
        "timeframe": inst.timeframe,
        "entry_rule": entry_rule,
        "exit_rule": exit_rule,
        "confidence_grade": inst.confidence_grade,
        "stop_loss_pct": params.get("stop_loss_pct"),
        "stop_anchor_type": params.get("stop_anchor_type") or spec.get("stop_anchor_type"),
        "target_anchor_type": params.get("target_anchor_type"),
        "stop_loss_atr_mult": params.get("stop_loss_atr_mult"),
    }

    # emergency_exit: stop_loss_pct must be 0.0 (TriggerCondition validator requirement)
    if inst.archetype_id == "emergency_exit":
        tc["stop_loss_pct"] = 0.0
        tc["stop_anchor_type"] = None

    return tc

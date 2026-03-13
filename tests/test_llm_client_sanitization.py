from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agents.strategies.llm_client import LLMClient
from schemas.llm_strategist import StrategyPlan


def _base_plan_dict() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "generated_at": now.isoformat(),
        "valid_until": (now + timedelta(days=1)).isoformat(),
        "regime": "range",
        "stance": "active",
        "triggers": [],
        "sizing_rules": [
            {
                "symbol": "BTC-USD",
                "sizing_mode": "fixed_fraction",
                "target_risk_pct": 1.0,
            }
        ],
    }


def test_sanitize_plan_dict_drops_trigger_extra_keys_and_maps_to_rationale():
    client = LLMClient(allow_fallback=False)
    plan = _base_plan_dict()
    plan["regime"] = "compression_breakout"
    plan["triggers"] = [
        {
            "id": "t1",
            "symbol": "BTC-USD",
            "category": "volatility_breakout",
            "confidence_grade": "B",
            "direction": "short",
            "timeframe": "1m",
            "entry_rule": "is_flat and close < donchian_lower_short",
            "exit_rule": "not is_flat and stop_hit",
            "stop_loss_pct": 1.2,
            "note": "Informational: logs compression state",
            "confidence_detail": "Moderate confidence due volume profile",
        }
    ]

    sanitized = client._sanitize_plan_dict(plan)

    assert sanitized["regime"] == "high_vol"
    trigger = sanitized["triggers"][0]
    assert "note" not in trigger
    assert "confidence_detail" not in trigger
    assert trigger.get("rationale") == "Informational: logs compression state"

    # Must validate after sanitization (regression for pydantic extra_forbidden errors).
    StrategyPlan.model_validate(sanitized)


def test_sanitize_plan_dict_unknown_regime_coerces_to_mixed():
    client = LLMClient(allow_fallback=False)
    plan = _base_plan_dict()
    plan["regime"] = "totally_unknown_regime_name"

    sanitized = client._sanitize_plan_dict(plan)

    assert sanitized["regime"] == "mixed"
    StrategyPlan.model_validate(sanitized)


def test_sanitize_plan_dict_drops_unknown_top_level_keys():
    client = LLMClient(allow_fallback=False)
    plan = _base_plan_dict()
    plan["debug_blob"] = {"foo": "bar"}

    sanitized = client._sanitize_plan_dict(plan)

    assert "debug_blob" not in sanitized
    StrategyPlan.model_validate(sanitized)

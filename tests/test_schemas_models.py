from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from schemas.judge_feedback import JudgeFeedback
from schemas.llm_strategist import StrategyPlan


def _legacy_plan_payload() -> dict:
    generated_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return {
        "generated_at": generated_at.isoformat(),
        "valid_until": (generated_at + timedelta(days=1)).isoformat(),
        "global_view": "legacy payload",
        "regime": "range",
        "triggers": [
            {
                "id": "buy_breakout",
                "symbol": "BTC-USD",
                "direction": "long",
                "timeframe": "1h",
                "entry_rule": "close > sma_short",
                "exit_rule": "close < sma_medium",
            }
        ],
        "risk_constraints": {
            "max_position_risk_pct": 2.0,
            "max_symbol_exposure_pct": 25.0,
            "max_portfolio_exposure_pct": 80.0,
            "max_daily_loss_pct": 3.0,
        },
        "sizing_rules": [
            {"symbol": "BTC-USD", "sizing_mode": "fixed_fraction", "target_risk_pct": 1.0}
        ],
    }


def test_strategy_plan_assigns_defaults_for_legacy_json() -> None:
    payload = _legacy_plan_payload()
    plan = StrategyPlan.model_validate_json(json.dumps(payload))
    assert plan.plan_id.startswith("plan_")
    assert plan.run_id is None
    assert plan.max_trades_per_day is None
    assert plan.min_trades_per_day is None
    assert plan.allowed_symbols == []
    assert plan.allowed_directions == []
    assert plan.allowed_trigger_categories == []


def test_judge_feedback_parses_constraints() -> None:
    payload = {
        "score": 47.5,
        "notes": "oversized exposure",
        "constraints": {
            "max_trades_per_day": 5,
            "risk_mode": "conservative",
            "disabled_trigger_ids": ["btc_scalp"],
            "disabled_categories": ["scalp"],
        },
        "strategist_constraints": {
            "must_fix": ["Trim overexposed assets"],
            "vetoes": ["Avoid leverage"],
            "boost": [],
            "regime_correction": "mixed",
            "sizing_adjustments": {"BTC-USD": "cap at 1%"},
        },
    }
    feedback = JudgeFeedback.model_validate_json(json.dumps(payload))
    assert feedback.constraints.max_trades_per_day == 5
    assert feedback.constraints.risk_mode == "conservative"
    assert feedback.constraints.disabled_trigger_ids == ["btc_scalp"]
    assert feedback.strategist_constraints.must_fix == ["Trim overexposed assets"]
    assert feedback.strategist_constraints.sizing_adjustments["BTC-USD"] == "cap at 1%"


def test_judge_feedback_defaults_when_constraints_omitted() -> None:
    payload = {
        "score": 60.0,
        "strategist_constraints": {
            "must_fix": [],
            "vetoes": [],
            "boost": [],
            "regime_correction": "keep balanced",
            "sizing_adjustments": {},
        },
    }
    feedback = JudgeFeedback.model_validate(payload)
    assert feedback.constraints.max_trades_per_day is None
    assert feedback.constraints.risk_mode == "normal"
    assert feedback.strategist_constraints.regime_correction == "keep balanced"

"""LLM shim payloads for strategist and judge backtesting."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List

from schemas.judge_feedback import DisplayConstraints, JudgeConstraints

_BASE_TRIGGER_TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "trend_continuation_long_1",
        "category": "trend_continuation",
        "confidence_grade": "A",
        "direction": "long",
        "timeframe": "1h",
        "entry_rule": "rsi_14 < 40 and close > sma_medium and position == 'flat'",
        "exit_rule": "rsi_14 > 60",
    },
    {
        "id": "trend_continuation_short_1",
        "category": "trend_continuation",
        "confidence_grade": "A",
        "direction": "short",
        "timeframe": "1h",
        "entry_rule": "rsi_14 > 60 and close < sma_medium and position == 'flat'",
        "exit_rule": "rsi_14 < 50",
    },
    {
        "id": "mean_reversion_long_1",
        "category": "mean_reversion",
        "confidence_grade": "B",
        "direction": "long",
        "timeframe": "15m",
        "entry_rule": "close < bollinger_lower and rsi_14 < 45 and position == 'flat'",
        "exit_rule": "close > sma_short",
    },
    {
        "id": "mean_reversion_short_1",
        "category": "mean_reversion",
        "confidence_grade": "B",
        "direction": "short",
        "timeframe": "15m",
        "entry_rule": "close > bollinger_upper and rsi_14 > 55 and position == 'flat'",
        "exit_rule": "close < sma_short",
    },
    {
        "id": "volatility_breakout_long_1",
        "category": "volatility_breakout",
        "confidence_grade": "C",
        "direction": "long",
        "timeframe": "5m",
        "entry_rule": "close > donchian_upper_short and atr_14 > 1.5 * tf_1h_atr_14 and position == 'flat'",
        "exit_rule": "",
    },
    {
        "id": "volatility_breakout_short_1",
        "category": "volatility_breakout",
        "confidence_grade": "C",
        "direction": "short",
        "timeframe": "5m",
        "entry_rule": "close < donchian_lower_short and atr_14 > 1.5 * tf_1h_atr_14 and position == 'flat'",
        "exit_rule": "",
    },
    {
        "id": "reversal_long_1",
        "category": "reversal",
        "confidence_grade": "A",
        "direction": "long",
        "timeframe": "30m",
        "entry_rule": "trend_state == 'downtrend' and close > sma_short and rsi_14 < 40 and position == 'flat'",
        "exit_rule": "close < sma_short",
    },
    {
        "id": "reversal_short_1",
        "category": "reversal",
        "confidence_grade": "A",
        "direction": "short",
        "timeframe": "30m",
        "entry_rule": "trend_state == 'uptrend' and close < sma_short and rsi_14 > 60 and position == 'flat'",
        "exit_rule": "close > sma_short",
    },
    {
        "id": "emergency_exit_1_flat",
        "category": "emergency_exit",
        "confidence_grade": "A",
        "direction": "exit",
        "timeframe": "1h",
        "entry_rule": "position != 'flat'",
        "exit_rule": "",
    },
]

_ALLOWED_REGIMES = {"bull", "bear", "range", "high_vol", "mixed"}


def make_strategist_shim_transport() -> Callable[[str], str]:
    """Return a transport callable that emits a deterministic strategist plan."""

    def _transport(payload: str) -> str:
        plan = build_strategist_shim_plan(payload)
        return json.dumps(plan)

    return _transport


def build_strategist_shim_plan(llm_input_json: str) -> Dict[str, Any]:
    """Build a strategist plan compatible with StrategyPlan.from_json."""
    try:
        llm_payload = json.loads(llm_input_json or "{}")
    except json.JSONDecodeError:
        llm_payload = {}

    symbols = _extract_symbols(llm_payload)
    risk_params = llm_payload.get("risk_params") or {}
    risk_constraints = _risk_constraints_from_params(risk_params)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    regime = _normalize_regime(llm_payload)

    plan = {
        "generated_at": _isoformat(now),
        "valid_until": _isoformat(now + timedelta(hours=24)),
        "regime": regime,
        "global_view": "LLM shim response for pipeline validation.",
        "risk_constraints": risk_constraints,
        "sizing_rules": _build_sizing_rules(symbols, risk_constraints),
        "triggers": _build_triggers(symbols),
    }
    return plan


def build_judge_shim_feedback(summary: Dict[str, Any], trade_metrics: Any | None = None) -> Dict[str, Any]:
    """Build a judge feedback payload that conforms to JudgeFeedback."""
    return_pct = float(summary.get("return_pct", 0.0) or 0.0)
    score = 55.0 + max(min(return_pct, 5.0), -5.0) * 2.0
    score = max(0.0, min(100.0, round(score, 1)))
    trade_count = int(summary.get("trade_count", 0) or 0)

    notes = (
        "Shim judge response. "
        f"Return={return_pct:.2f}% TradeCount={trade_count}."
    )
    constraints = JudgeConstraints().model_dump()
    strategist_constraints = DisplayConstraints().model_dump()

    return {
        "score": score,
        "notes": notes,
        "constraints": constraints,
        "strategist_constraints": strategist_constraints,
    }


def _extract_symbols(llm_payload: Dict[str, Any]) -> List[str]:
    assets = llm_payload.get("assets") or []
    symbols = [asset.get("symbol") for asset in assets if asset.get("symbol")]
    if not symbols:
        return ["BTC-USD"]
    return symbols


def _risk_constraints_from_params(risk_params: Dict[str, Any]) -> Dict[str, Any]:
    def _value(key: str, default: float) -> float:
        raw = risk_params.get(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    constraints = {
        "max_position_risk_pct": _value("max_position_risk_pct", 4.0),
        "max_symbol_exposure_pct": _value("max_symbol_exposure_pct", 50.0),
        "max_portfolio_exposure_pct": _value("max_portfolio_exposure_pct", 100.0),
        "max_daily_loss_pct": _value("max_daily_loss_pct", 6.0),
    }
    budget_pct = risk_params.get("max_daily_risk_budget_pct")
    if budget_pct is not None:
        try:
            constraints["max_daily_risk_budget_pct"] = float(budget_pct)
        except (TypeError, ValueError):
            pass
    return constraints


def _normalize_regime(llm_payload: Dict[str, Any]) -> str:
    global_context = llm_payload.get("global_context") or {}
    regime = global_context.get("regime") or "bull"
    if isinstance(regime, str):
        regime = regime.lower().strip()
    return regime if regime in _ALLOWED_REGIMES else "bull"


def _build_sizing_rules(symbols: List[str], risk_constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_risk = risk_constraints.get("max_position_risk_pct", 4.0)
    return [
        {
            "symbol": symbol,
            "sizing_mode": "fixed_fraction",
            "target_risk_pct": target_risk,
        }
        for symbol in symbols
    ]


def _build_triggers(symbols: List[str]) -> List[Dict[str, Any]]:
    triggers: List[Dict[str, Any]] = []
    for symbol in symbols:
        symbol_slug = symbol.lower().replace("-", "_")
        for template in _BASE_TRIGGER_TEMPLATES:
            trigger = copy.deepcopy(template)
            trigger["symbol"] = symbol
            trigger["id"] = f"{symbol_slug}_{template['id']}"
            triggers.append(trigger)
    return triggers


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def make_judge_shim_transport() -> Callable[[str], str]:
    """Return a transport callable that emits deterministic judge feedback.

    This transport uses the pre-computed heuristics from JudgeFeedbackService
    to build a deterministic response, enabling fast backtests without LLM calls.
    """

    def _transport(payload: str) -> str:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {}

        heuristics = data.get("heuristics", {})
        summary = data.get("summary", {})

        # Use heuristic analysis for deterministic response
        score = heuristics.get("final_score", 50.0)
        observations = heuristics.get("observations", [])
        suggested_constraints = heuristics.get("suggested_constraints", {})
        suggested_strategist = heuristics.get("suggested_strategist_constraints", {})

        # Build notes from observations
        notes_parts = observations[:5] if observations else []
        if not notes_parts:
            return_pct = summary.get("return_pct", 0.0)
            trade_count = summary.get("trade_count", 0)
            notes_parts = [f"Shim judge response. Return={return_pct:.2f}% TradeCount={trade_count}."]

        feedback = {
            "score": score,
            "notes": " ".join(notes_parts),
            "constraints": {
                "max_trades_per_day": suggested_constraints.get("max_trades_per_day"),
                "max_triggers_per_symbol_per_day": suggested_constraints.get(
                    "max_triggers_per_symbol_per_day"
                ),
                "risk_mode": suggested_constraints.get("risk_mode", "normal"),
                "disabled_trigger_ids": suggested_constraints.get("disabled_trigger_ids", []),
                "disabled_categories": suggested_constraints.get("disabled_categories", []),
            },
            "strategist_constraints": {
                "must_fix": suggested_strategist.get("must_fix", []),
                "vetoes": suggested_strategist.get("vetoes", []),
                "boost": suggested_strategist.get("boost", []),
                "regime_correction": suggested_strategist.get("regime_correction"),
                "sizing_adjustments": suggested_strategist.get("sizing_adjustments", {}),
            },
        }
        return json.dumps(feedback)

    return _transport

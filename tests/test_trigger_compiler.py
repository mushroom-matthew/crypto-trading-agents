from __future__ import annotations

import pytest

from schemas.compiled_plan import CompiledPlan
from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition
from schemas.strategy_run import StrategyRunConfig
from services.strategy_run_registry import StrategyRunRegistry
from tools import strategy_run_tools
from trading_core.trigger_compiler import (
    AtrTautologyWarning,
    ExitBindingCorrection,
    HoldRuleStripped,
    IdentifierCorrection,
    PlanEnforcementResult,
    TriggerCompilationError,
    autocorrect_identifiers,
    compile_plan,
    detect_atr_tautologies,
    detect_plan_atr_tautologies,
    enforce_exit_binding,
    enforce_plan_quality,
    strip_degenerate_hold_rules,
    warn_cross_category_exits,
    detect_degenerate_hold_rules,
)

from datetime import datetime, timedelta, timezone


def _strategy_plan(run_id: str, entry_rule: str = "close > 0") -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trigger = TriggerCondition(
        id="btc_long",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule=entry_rule,
        exit_rule="close < 0",
        category="trend_continuation",
    )
    return StrategyPlan(
        plan_id="plan_test",
        run_id=run_id,
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[trigger],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=1.0,
            max_symbol_exposure_pct=25.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=5,
    )


def test_compile_plan_succeeds_with_valid_expressions(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    plan = _strategy_plan(run.run_id, "close > sma_short")
    compiled = compile_plan(plan)
    assert isinstance(compiled, CompiledPlan)
    assert compiled.triggers[0].entry.normalized == "close > sma_short"


def test_compile_plan_fails_on_invalid_identifier(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    plan = _strategy_plan(run.run_id, "evil_call()")
    with pytest.raises(TriggerCompilationError):
        compile_plan(plan)


def test_compile_tool_updates_run(tmp_path, monkeypatch):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    monkeypatch.setattr(strategy_run_tools, "registry", registry)
    plan = _strategy_plan(run.run_id).model_dump()
    compiled = strategy_run_tools.compile_plan_tool(plan)
    assert compiled["plan_id"] == "plan_test"
    stored = registry.get_strategy_run(run.run_id)
    assert stored.plan_active is True
    assert stored.compiled_plan_id == "plan_test"


def test_compile_plan_allows_identity_checks(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs_identity")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    plan = _strategy_plan(run.run_id, "sma_short is not None and close > sma_short")
    compiled = compile_plan(plan)
    assert compiled.triggers[0].entry.normalized == "sma_short is not None and close > sma_short"


def test_compile_plan_allows_in_operator(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs_in")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    plan = _strategy_plan(run.run_id, "vol_state in ['high', 'extreme']")
    compiled = compile_plan(plan)
    assert "in" in compiled.triggers[0].entry.normalized


def test_compile_plan_allows_not_in_operator(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs_not_in")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    plan = _strategy_plan(run.run_id, "position not in ['long', 'short']")
    compiled = compile_plan(plan)
    assert "not in" in compiled.triggers[0].entry.normalized


def test_between_allows_identifiers(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs_between")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    rule = "close between donchian_lower_short and donchian_upper_short"
    plan = _strategy_plan(run.run_id, rule)
    compiled = compile_plan(plan)
    assert (
        compiled.triggers[0].entry.normalized
        == "((close) >= (donchian_lower_short) and (close) <= (donchian_upper_short))"
    )


# =============================================================================
# Runbook 21 — ATR Tautology Detection
# =============================================================================


def test_detects_atr_tautology_1d_vs_4h():
    """tf_1d_atr > tf_4h_atr is always true — should be detected."""
    warnings = detect_atr_tautologies(
        "tf_1d_atr > tf_4h_atr",
        trigger_id="btc_emergency",
        rule_type="entry",
    )
    assert len(warnings) == 1
    assert isinstance(warnings[0], AtrTautologyWarning)
    assert "always true" in str(warnings[0])


def test_detects_atr_tautology_4h_vs_1h():
    """tf_4h_atr > tf_1h_atr is always true."""
    warnings = detect_atr_tautologies("tf_4h_atr > tf_1h_atr")
    assert len(warnings) == 1


def test_detects_atr_tautology_with_atr_14_suffix():
    """tf_1d_atr_14 > tf_4h_atr_14 should also be detected."""
    warnings = detect_atr_tautologies("tf_1d_atr_14 > tf_4h_atr_14")
    assert len(warnings) == 1


def test_allows_atr_ratio_comparison():
    """tf_1d_atr > 2.5 * tf_4h_atr uses a ratio — NOT a tautology."""
    warnings = detect_atr_tautologies("tf_1d_atr > 2.5 * tf_4h_atr")
    assert len(warnings) == 0


def test_allows_same_timeframe_atr():
    """atr_14 > sma_medium * 0.03 is not a cross-timeframe comparison."""
    warnings = detect_atr_tautologies("atr_14 > sma_medium * 0.03")
    assert len(warnings) == 0


def test_no_tautology_lower_gt_higher():
    """tf_1h_atr > tf_4h_atr — lower > higher is NOT always true (it's usually false)."""
    warnings = detect_atr_tautologies("tf_1h_atr > tf_4h_atr")
    assert len(warnings) == 0


def test_detects_atr_tautology_lt_operator():
    """tf_4h_atr < tf_1d_atr is always true (reversed comparison)."""
    warnings = detect_atr_tautologies("tf_4h_atr < tf_1d_atr")
    assert len(warnings) == 1


# =============================================================================
# Runbook 22 — Cross-Category Exit Warning
# =============================================================================


def test_warns_cross_category_entries_same_symbol():
    """If a symbol has entries in multiple categories, warn."""
    triggers = [
        TriggerCondition(
            id="btc_trend",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="close > sma_medium",
            exit_rule="close < sma_short",
            category="trend_continuation",
        ),
        TriggerCondition(
            id="btc_reversal",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="rsi_14 < 30",
            exit_rule="rsi_14 > 70",
            category="reversal",
        ),
    ]
    warnings = warn_cross_category_exits(triggers)
    assert len(warnings) >= 1
    assert "BTC-USD" in warnings[0]


def test_no_warning_single_category():
    """Single category per symbol should produce no warnings."""
    triggers = [
        TriggerCondition(
            id="btc_trend_1",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="close > sma_medium",
            exit_rule="close < sma_short",
            category="trend_continuation",
        ),
        TriggerCondition(
            id="btc_trend_2",
            symbol="BTC-USD",
            direction="long",
            timeframe="4h",
            entry_rule="close > sma_long",
            exit_rule="close < sma_medium",
            category="trend_continuation",
        ),
    ]
    warnings = warn_cross_category_exits(triggers)
    assert len(warnings) == 0


# =============================================================================
# Runbook 23 — Degenerate Hold Rule Detection
# =============================================================================


def test_flags_degenerate_hold_rule():
    """Single-condition hold rule with rsi_14 > 45 should be flagged."""
    warnings = detect_degenerate_hold_rules("rsi_14 > 45", trigger_id="btc_hold")
    assert len(warnings) >= 1
    assert "degenerate" in warnings[0].lower() or "single" in warnings[0].lower()


def test_allows_compound_hold_rule():
    """Multi-condition hold rule should not be flagged."""
    warnings = detect_degenerate_hold_rules(
        "rsi_14 > 60 and close > sma_medium and atr_14 < 500",
        trigger_id="btc_hold",
    )
    assert len(warnings) == 0


# =============================================================================
# Runbook 32 — Exit Binding Enforcement
# =============================================================================


def test_enforce_exit_binding_relabels_mismatch():
    """Exit trigger in wrong category gets relabeled to match the only entry category."""
    triggers = [
        TriggerCondition(
            id="btc_entry",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="close > sma_medium",
            exit_rule="",
            category="trend_continuation",
        ),
        TriggerCondition(
            id="btc_exit",
            symbol="BTC-USD",
            direction="exit",
            timeframe="1h",
            entry_rule="",
            exit_rule="rsi_14 > 70",
            category="reversal",  # Wrong — should be trend_continuation
        ),
    ]
    corrections = enforce_exit_binding(triggers)
    assert len(corrections) == 1
    assert corrections[0].trigger_id == "btc_exit"
    assert corrections[0].original_category == "reversal"
    assert corrections[0].corrected_category == "trend_continuation"
    assert triggers[1].category == "trend_continuation"


def test_enforce_exit_binding_no_change_when_matching():
    """Matching exit category is left untouched."""
    triggers = [
        TriggerCondition(
            id="btc_entry",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="close > sma_medium",
            exit_rule="close < sma_short",
            category="trend_continuation",
        ),
        TriggerCondition(
            id="btc_exit",
            symbol="BTC-USD",
            direction="exit",
            timeframe="1h",
            entry_rule="",
            exit_rule="rsi_14 > 70",
            category="trend_continuation",
        ),
    ]
    corrections = enforce_exit_binding(triggers)
    assert len(corrections) == 0
    assert triggers[1].category == "trend_continuation"


# =============================================================================
# Runbook 33 — Degenerate Hold Rule Stripping (Enforcement)
# =============================================================================


def test_strip_degenerate_hold_rule():
    """Single-condition hold rule gets stripped to None."""
    triggers = [
        TriggerCondition(
            id="btc_long",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="close > sma_medium",
            exit_rule="close < sma_short",
            category="trend_continuation",
            hold_rule="rsi_14 > 45",  # Degenerate
        ),
    ]
    stripped = strip_degenerate_hold_rules(triggers)
    assert len(stripped) == 1
    assert stripped[0].trigger_id == "btc_long"
    assert stripped[0].original_rule == "rsi_14 > 45"
    assert triggers[0].hold_rule is None


def test_preserve_compound_hold_rule():
    """Compound hold rule is preserved."""
    triggers = [
        TriggerCondition(
            id="btc_long",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="close > sma_medium",
            exit_rule="close < sma_short",
            category="trend_continuation",
            hold_rule="rsi_14 > 60 and close > sma_medium and atr_14 < 500",
        ),
    ]
    stripped = strip_degenerate_hold_rules(triggers)
    assert len(stripped) == 0
    assert triggers[0].hold_rule == "rsi_14 > 60 and close > sma_medium and atr_14 < 500"


# =============================================================================
# Runbook 34 — Identifier Autocorrect
# =============================================================================


def _plan_with_rule(entry_rule: str, run_id: str = "run_test") -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trigger = TriggerCondition(
        id="btc_long",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule=entry_rule,
        exit_rule="close < 0",
        category="trend_continuation",
    )
    return StrategyPlan(
        plan_id="plan_test",
        run_id=run_id,
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[trigger],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=1.0,
            max_symbol_exposure_pct=25.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=5,
    )


def test_autocorrect_known_typo():
    """realization_vol_short → realized_vol_short via KNOWN_TYPOS."""
    plan = _plan_with_rule("realization_vol_short > 0.02")
    corrections = autocorrect_identifiers(plan, {"1h"})
    autocorrects = [c for c in corrections if c.action == "autocorrect"]
    assert len(autocorrects) == 1
    assert autocorrects[0].identifier == "realization_vol_short"
    assert autocorrects[0].replacement == "realized_vol_short"
    assert "realized_vol_short" in plan.triggers[0].entry_rule


def test_reject_unknown_identifier():
    """Trigger with unresolvable identifier gets rule stripped."""
    plan = _plan_with_rule("totally_fake_indicator > 100")
    corrections = autocorrect_identifiers(plan, {"1h"})
    strips = [c for c in corrections if c.action == "strip"]
    assert len(strips) >= 1
    assert strips[0].identifier == "totally_fake_indicator"
    assert not plan.triggers[0].entry_rule  # Rule was stripped


def test_valid_expression_passthrough():
    """Valid expressions are left unchanged."""
    plan = _plan_with_rule("close > sma_short and rsi_14 < 70")
    corrections = autocorrect_identifiers(plan, {"1h"})
    assert len(corrections) == 0
    assert plan.triggers[0].entry_rule == "close > sma_short and rsi_14 < 70"


# =============================================================================
# Runbook 32-34 — Orchestrator (enforce_plan_quality)
# =============================================================================


def test_enforce_plan_quality_full_pipeline():
    """enforce_plan_quality runs all checks and returns combined result."""
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    triggers = [
        TriggerCondition(
            id="btc_entry",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="close > sma_medium",
            exit_rule="",
            category="trend_continuation",
        ),
        TriggerCondition(
            id="btc_exit",
            symbol="BTC-USD",
            direction="exit",
            timeframe="1h",
            entry_rule="",
            exit_rule="rsi_14 > 70",
            category="reversal",  # Exit binding mismatch
            hold_rule="rsi_14 > 45",  # Degenerate hold
        ),
    ]
    plan = StrategyPlan(
        plan_id="plan_test",
        run_id="run_test",
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=triggers,
        risk_constraints=RiskConstraint(
            max_position_risk_pct=1.0,
            max_symbol_exposure_pct=25.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=5,
    )
    result = enforce_plan_quality(plan, {"1h"})
    assert isinstance(result, PlanEnforcementResult)
    assert result.total_corrections >= 2  # At least exit binding + hold rule
    assert len(result.exit_binding_corrections) >= 1
    assert len(result.hold_rules_stripped) >= 1

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
    ExitRuleSanitization,
    HoldRuleStripped,
    IdentifierCorrection,
    PlanEnforcementResult,
    TriggerCompilationError,
    _is_exit_rule_normal_form,
    autocorrect_identifiers,
    compile_plan,
    detect_atr_tautologies,
    detect_plan_atr_tautologies,
    enforce_exit_binding,
    enforce_plan_quality,
    sanitize_exit_rules,
    strip_degenerate_hold_rules,
    tighten_stop_only,
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
    plan = _strategy_plan(run.run_id, "vol_state == 'extreme'")
    compiled = compile_plan(plan)
    assert "==" in compiled.triggers[0].entry.normalized


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


def test_enforce_exit_binding_skips_emergency_exit():
    """Emergency exit triggers are never relabeled or stripped."""
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
            id="btc_emergency",
            symbol="BTC-USD",
            direction="exit",
            timeframe="1h",
            entry_rule="",
            exit_rule="atr_14 > 1000",
            category="emergency_exit",
        ),
    ]
    corrections = enforce_exit_binding(triggers)
    assert len(corrections) == 0
    assert triggers[1].category == "emergency_exit"
    assert triggers[1].exit_rule == "atr_14 > 1000"


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


def test_emergency_exit_survives_enforce_plan_quality():
    """Emergency exit triggers with unknown identifiers must NOT have their exit_rule stripped."""
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
            id="btc_emergency",
            symbol="BTC-USD",
            direction="exit",
            timeframe="1h",
            entry_rule="",
            exit_rule="some_custom_signal > 100",  # Unknown identifier
            category="emergency_exit",
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
    # Emergency exit must keep its exit_rule intact
    assert triggers[1].exit_rule == "some_custom_signal > 100"
    # No strip corrections for the emergency exit
    emergency_strips = [c for c in result.identifier_corrections
                        if c.trigger_id == "btc_emergency" and c.action == "strip"]
    assert len(emergency_strips) == 0


# =============================================================================
# Exit Binding Exempt — Multi-Category Symbols
# =============================================================================


def test_enforce_exit_binding_exempts_multi_category_match():
    """Symbol with 2 entry cats, exit matches one → exit_binding_exempt=True, exit_rule preserved."""
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
            id="btc_mean_rev",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="rsi_14 < 30",
            exit_rule="rsi_14 > 70",
            category="mean_reversion",
        ),
        TriggerCondition(
            id="btc_exit_trend",
            symbol="BTC-USD",
            direction="exit",
            timeframe="1h",
            entry_rule="",
            exit_rule="close < sma_medium",
            category="trend_continuation",  # Matches one of the two entry cats
        ),
    ]
    corrections = enforce_exit_binding(triggers)
    # No corrections — the exit matches an entry category
    assert len(corrections) == 0
    # Exit rule preserved (not stripped)
    assert triggers[2].exit_rule == "close < sma_medium"
    # Exempt flag set so runtime binding check passes regardless of which category opened the position
    assert triggers[2].exit_binding_exempt is True


def test_enforce_exit_binding_strips_multi_category_no_match():
    """Symbol with 2 entry cats, exit matches NONE → exit_rule stripped."""
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
            id="btc_mean_rev",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            entry_rule="rsi_14 < 30",
            exit_rule="rsi_14 > 70",
            category="mean_reversion",
        ),
        TriggerCondition(
            id="btc_exit_other",
            symbol="BTC-USD",
            direction="exit",
            timeframe="1h",
            entry_rule="",
            exit_rule="atr_14 > 500",
            category="other",  # Matches neither entry category
        ),
    ]
    corrections = enforce_exit_binding(triggers)
    assert len(corrections) == 1
    assert corrections[0].trigger_id == "btc_exit_other"
    assert corrections[0].corrected_category is None  # stripped, not relabeled
    # Exit rule stripped
    assert triggers[2].exit_rule == ""
    # Exempt flag NOT set
    assert triggers[2].exit_binding_exempt is False


def test_exit_binding_exempt_reset_on_external_input():
    """exit_binding_exempt=True in external input is reset to False by validator."""
    trigger = TriggerCondition(
        id="btc_exit",
        symbol="BTC-USD",
        direction="exit",
        timeframe="1h",
        entry_rule="",
        exit_rule="close < sma_short",
        category="trend_continuation",
        exit_binding_exempt=True,  # Externally supplied — should be reset
    )
    assert trigger.exit_binding_exempt is False


# =============================================================================
# _is_exit_rule_normal_form — grammar validator unit tests
# =============================================================================

def test_normal_form_canonical_two_terminals():
    assert _is_exit_rule_normal_form("not is_flat and (stop_hit or target_hit)")


def test_normal_form_canonical_three_terminals():
    assert _is_exit_rule_normal_form("not is_flat and (stop_hit or target_hit or force_flatten)")


def test_normal_form_single_terminal():
    assert _is_exit_rule_normal_form("not is_flat and stop_hit")
    assert _is_exit_rule_normal_form("not is_flat and target_hit")


def test_normal_form_backward_compat_aliases():
    assert _is_exit_rule_normal_form("not is_flat and (below_stop or above_target)")


def test_normal_form_rejects_comparison_on_allowed_id():
    """position_age_minutes > 5 uses a Compare node — not a terminal."""
    assert not _is_exit_rule_normal_form("not is_flat and (stop_hit or (position_age_minutes > 5))")


def test_normal_form_rejects_r_tracking_comparison():
    """current_R < 1.0 is a Compare node — advisory, not a valid exit terminal."""
    assert not _is_exit_rule_normal_form("not is_flat and (stop_hit or (r2_reached and current_R < 1.0))")


def test_normal_form_rejects_indicator_condition():
    assert not _is_exit_rule_normal_form("not is_flat and (stop_hit or target_hit or (rsi_14 > 75 and macd_hist < 0))")


def test_normal_form_rejects_nested_and_in_rhs():
    """Boolean obfuscation: nested And inside an Or-of-terminals."""
    assert not _is_exit_rule_normal_form("not is_flat and (stop_hit or (target_hit and position_qty > 0))")


def test_normal_form_rejects_stop_distance_comparison():
    assert not _is_exit_rule_normal_form("not is_flat and stop_distance_pct < 0.001")


def test_normal_form_rejects_churn_rule():
    assert not _is_exit_rule_normal_form("not is_flat and current_R < 0.1")


def test_normal_form_rejects_unparseable():
    assert not _is_exit_rule_normal_form("not is_flat and (stop_hit or ???)")


def test_normal_form_rejects_empty():
    assert not _is_exit_rule_normal_form("")


def test_normal_form_rejects_missing_guard():
    """RHS only, no 'not is_flat' guard."""
    assert not _is_exit_rule_normal_form("stop_hit or target_hit")


# =============================================================================
# sanitize_exit_rules — hard invariant tests
# =============================================================================

def _make_trigger(
    id: str = "t1",
    category: str = "reversal",
    direction: str = "long",
    entry_rule: str = "close > 0",
    exit_rule: str = "",
) -> TriggerCondition:
    return TriggerCondition(
        id=id,
        symbol="ETH-USD",
        timeframe="1h",
        direction=direction,
        category=category,
        entry_rule=entry_rule,
        exit_rule=exit_rule,
    )


def test_sanitize_exit_rules_strips_indicator_conditions():
    """exit_rule with RSI/MACD indicator conditions is replaced with canonical rule."""
    t = _make_trigger(
        exit_rule="not is_flat and (stop_hit or target_hit or (rsi_14 > 75 and macd_hist < 0))",
    )
    results = sanitize_exit_rules([t])
    assert len(results) == 1
    s = results[0]
    assert s.trigger_id == "t1"
    assert s.reason == "invalid_grammar"
    assert "rsi_14" in s.stripped_identifiers
    assert "macd_hist" in s.stripped_identifiers
    assert t.exit_rule == "not is_flat and (stop_hit or target_hit)"


def test_sanitize_exit_rules_clean_rule_untouched():
    """Canonical exit_rule with only stop_hit/target_hit is not modified."""
    t = _make_trigger(exit_rule="not is_flat and (stop_hit or target_hit)")
    results = sanitize_exit_rules([t])
    assert results == []
    assert t.exit_rule == "not is_flat and (stop_hit or target_hit)"


def test_sanitize_exit_rules_r_tracking_comparison_rejected():
    """R-tracking comparisons (current_R < 1.0) are NOT valid exit-rule grammar.

    R-tracking is advisory — it belongs in hold_rule for stop advancement,
    not in exit_rule to close a position.  The grammar rejects any Compare node.
    """
    t = _make_trigger(exit_rule="not is_flat and (stop_hit or (r2_reached and current_R < 1.0))")
    results = sanitize_exit_rules([t])
    assert len(results) == 1
    assert results[0].reason == "invalid_grammar"
    assert t.exit_rule == "not is_flat and (stop_hit or target_hit)"


def test_sanitize_exit_rules_emergency_exit_exempt():
    """Emergency exit triggers bypass exit_rule sanitization."""
    t = _make_trigger(
        category="emergency_exit",
        direction="exit",
        exit_rule="not is_flat and (stop_hit or rsi_14 > 80)",
    )
    results = sanitize_exit_rules([t])
    assert results == []
    # Rule unchanged
    assert "rsi_14" in t.exit_rule


def test_sanitize_exit_rules_no_exit_rule_skipped():
    """Triggers without an exit_rule are skipped cleanly."""
    t = _make_trigger(exit_rule="")
    results = sanitize_exit_rules([t])
    assert results == []


def test_sanitize_exit_rules_multiple_triggers():
    """Sanitizer processes all triggers; only those failing the grammar are corrected."""
    t_bad = _make_trigger("bad", exit_rule="not is_flat and (stop_hit or sma_short > close)")
    t_ok = _make_trigger("ok", exit_rule="not is_flat and (stop_hit or target_hit)")
    results = sanitize_exit_rules([t_bad, t_ok])
    assert len(results) == 1
    assert results[0].trigger_id == "bad"
    assert t_bad.exit_rule == "not is_flat and (stop_hit or target_hit)"
    assert t_ok.exit_rule == "not is_flat and (stop_hit or target_hit)"  # unchanged


def test_sanitize_exit_rules_volume_stripped():
    """volume comparison is invalid exit grammar (Compare node)."""
    t = _make_trigger(exit_rule="not is_flat and (stop_hit or volume > 1000000)")
    results = sanitize_exit_rules([t])
    assert len(results) == 1
    assert results[0].reason == "invalid_grammar"
    assert "volume" in results[0].stripped_identifiers


# --- Adversarial patterns ---

def test_sanitize_adversarial_time_gate():
    """position_age_minutes > 5 exits every trade after 5 min — invalid grammar."""
    t = _make_trigger(exit_rule="not is_flat and (stop_hit or (position_age_minutes > 5))")
    results = sanitize_exit_rules([t])
    assert len(results) == 1
    assert results[0].reason == "invalid_grammar"
    assert t.exit_rule == "not is_flat and (stop_hit or target_hit)"


def test_sanitize_adversarial_churn_r_threshold():
    """current_R < 0.1 exits nearly flat positions — invalid Compare in grammar."""
    t = _make_trigger(exit_rule="not is_flat and (stop_hit or current_R < 0.1)")
    results = sanitize_exit_rules([t])
    assert len(results) == 1
    assert results[0].reason == "invalid_grammar"


def test_sanitize_adversarial_stop_distance_nonsense():
    """stop_distance_pct < 0.001 is semantically nonsense as an exit trigger."""
    t = _make_trigger(exit_rule="not is_flat and stop_distance_pct < 0.001")
    results = sanitize_exit_rules([t])
    assert len(results) == 1
    assert results[0].reason == "invalid_grammar"


def test_sanitize_adversarial_boolean_obfuscation():
    """stop_hit or (target_hit and position_qty > 0) hides a comparison in nested And."""
    t = _make_trigger(exit_rule="not is_flat and (stop_hit or (target_hit and position_qty > 0))")
    results = sanitize_exit_rules([t])
    assert len(results) == 1
    assert results[0].reason == "invalid_grammar"


def test_sanitize_strict_mode_raises():
    """In strict mode, a rule failing normal-form raises TriggerCompilationError."""
    t = _make_trigger(exit_rule="not is_flat and (stop_hit or rsi_14 > 70)")
    with pytest.raises(TriggerCompilationError, match="normal-form"):
        sanitize_exit_rules([t], strict=True)
    # Trigger rule is NOT mutated when strict mode raises
    assert "rsi_14" in t.exit_rule


def test_enforce_plan_quality_includes_sanitization(run_id="sanitize_test"):
    """enforce_plan_quality() integrates sanitize_exit_rules and counts them in total_corrections."""
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trigger = TriggerCondition(
        id="eth_reversal",
        symbol="ETH-USD",
        direction="long",
        timeframe="1h",
        entry_rule="rsi_14 < 35",
        exit_rule="not is_flat and (stop_hit or target_hit or macd_hist < 0)",
        category="reversal",
    )
    plan = StrategyPlan(
        plan_id="plan_sanitize",
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
        sizing_rules=[PositionSizingRule(symbol="ETH-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=5,
    )
    result = enforce_plan_quality(plan, available_timeframes={"1h"})
    assert len(result.exit_rule_sanitizations) == 1
    assert result.exit_rule_sanitizations[0].trigger_id == "eth_reversal"
    assert result.exit_rule_sanitizations[0].reason == "invalid_grammar"
    assert result.total_corrections >= 1


def test_enforce_plan_quality_strict_mode_raises():
    """enforce_plan_quality(strict=True) raises on any exit_rule grammar violation."""
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trigger = TriggerCondition(
        id="bad_trigger",
        symbol="ETH-USD",
        direction="long",
        timeframe="1h",
        entry_rule="close > 0",
        exit_rule="not is_flat and (stop_hit or position_age_minutes > 60)",
        category="reversal",
    )
    plan = StrategyPlan(
        plan_id="plan_strict",
        run_id="strict_test",
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
        sizing_rules=[PositionSizingRule(symbol="ETH-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=5,
    )
    with pytest.raises(TriggerCompilationError, match="normal-form"):
        enforce_plan_quality(plan, available_timeframes={"1h"}, strict=True)


# =============================================================================
# tighten_stop_only — monotonic stop enforcement tests
# =============================================================================

def test_tighten_stop_only_long_advances_stop():
    """For a long, a higher proposed stop is accepted (tightens the floor)."""
    assert tighten_stop_only(100.0, 105.0, "long") == 105.0


def test_tighten_stop_only_long_rejects_lower_stop():
    """For a long, a lower proposed stop is rejected (would widen, not tighten)."""
    assert tighten_stop_only(100.0, 95.0, "long") == 100.0


def test_tighten_stop_only_short_advances_stop():
    """For a short, a lower proposed stop is accepted (tightens the ceiling)."""
    assert tighten_stop_only(200.0, 195.0, "short") == 195.0


def test_tighten_stop_only_short_rejects_higher_stop():
    """For a short, a higher proposed stop is rejected (would widen, not tighten)."""
    assert tighten_stop_only(200.0, 205.0, "short") == 200.0


def test_tighten_stop_only_equal_is_idempotent():
    """Proposing the same stop as current is a no-op."""
    assert tighten_stop_only(150.0, 150.0, "long") == 150.0
    assert tighten_stop_only(150.0, 150.0, "short") == 150.0

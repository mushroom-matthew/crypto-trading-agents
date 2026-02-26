"""Tests for the portfolio meta-risk overlay evaluator (Runbook 60 Phase M4).

Covers:
- symbol concentration condition firing
- cluster concentration condition firing
- portfolio drawdown condition firing
- correlation spike condition firing
- hostile regime condition firing
- disabled policy returns no actions
- only matching condition_id actions are returned
- compute_symbol_concentration helper
- multiple conditions can fire simultaneously
"""
from __future__ import annotations

import pytest

from schemas.position_exit_contract import PortfolioMetaAction, PortfolioMetaRiskPolicy
from services.portfolio_meta_risk_overlay import (
    COND_CLUSTER_CONCENTRATION,
    COND_CORRELATION_SPIKE,
    COND_HOSTILE_REGIME,
    COND_PORTFOLIO_DRAWDOWN,
    COND_SYMBOL_CONCENTRATION,
    OverlayEvaluationResult,
    PortfolioMetrics,
    compute_symbol_concentration,
    evaluate_portfolio_risk_conditions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_policy(**kwargs) -> PortfolioMetaRiskPolicy:
    defaults = dict(policy_id="test-policy-01")
    defaults.update(kwargs)
    return PortfolioMetaRiskPolicy(**defaults)


def _trim_action(condition_id: str, priority: int = 0) -> PortfolioMetaAction:
    return PortfolioMetaAction(
        condition_id=condition_id,
        kind="trim_largest_position_to_cap",
        params={"cap_pct": 20.0},
        priority=priority,
    )


def _freeze_action(condition_id: str) -> PortfolioMetaAction:
    return PortfolioMetaAction(
        condition_id=condition_id,
        kind="freeze_new_entries",
    )


def _metrics(**kwargs) -> PortfolioMetrics:
    defaults = dict(
        total_equity=10_000.0,
        gross_exposure_pct=50.0,
        portfolio_drawdown_pct=5.0,
        symbol_concentrations={},
        cluster_concentrations={},
        max_pairwise_correlation=0.3,
        current_regime="normal",
    )
    defaults.update(kwargs)
    return PortfolioMetrics(**defaults)


# ---------------------------------------------------------------------------
# Disabled policy
# ---------------------------------------------------------------------------


def test_disabled_policy_returns_no_conditions():
    policy = _minimal_policy(enabled=False, max_symbol_concentration_pct=10.0)
    metrics = _metrics(symbol_concentrations={"BTC-USD": 50.0})
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert not result.any_condition_fired
    assert result.triggered_actions == []


# ---------------------------------------------------------------------------
# Symbol concentration
# ---------------------------------------------------------------------------


def test_symbol_concentration_fires_when_exceeded():
    policy = _minimal_policy(
        max_symbol_concentration_pct=30.0,
        actions=[_trim_action(COND_SYMBOL_CONCENTRATION)],
    )
    metrics = _metrics(symbol_concentrations={"BTC-USD": 45.0})
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert result.any_condition_fired
    assert any(c.condition_id == COND_SYMBOL_CONCENTRATION for c in result.conditions_fired)
    assert len(result.triggered_actions) == 1
    assert result.triggered_actions[0].kind == "trim_largest_position_to_cap"


def test_symbol_concentration_does_not_fire_below_threshold():
    policy = _minimal_policy(max_symbol_concentration_pct=50.0)
    metrics = _metrics(symbol_concentrations={"BTC-USD": 30.0})
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert not result.any_condition_fired


def test_symbol_concentration_not_set_never_fires():
    policy = _minimal_policy(max_symbol_concentration_pct=None)
    metrics = _metrics(symbol_concentrations={"BTC-USD": 99.0})
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert not any(c.condition_id == COND_SYMBOL_CONCENTRATION for c in result.conditions_fired)


# ---------------------------------------------------------------------------
# Cluster concentration
# ---------------------------------------------------------------------------


def test_cluster_concentration_fires_when_exceeded():
    policy = _minimal_policy(
        max_sector_or_cluster_concentration_pct=40.0,
        actions=[_trim_action(COND_CLUSTER_CONCENTRATION)],
    )
    metrics = _metrics(cluster_concentrations={"crypto_l1": 60.0})
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert any(c.condition_id == COND_CLUSTER_CONCENTRATION for c in result.conditions_fired)
    assert result.triggered_actions[0].kind == "trim_largest_position_to_cap"


def test_cluster_concentration_does_not_fire_at_threshold():
    policy = _minimal_policy(max_sector_or_cluster_concentration_pct=40.0)
    metrics = _metrics(cluster_concentrations={"crypto_l1": 40.0})
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert not any(c.condition_id == COND_CLUSTER_CONCENTRATION for c in result.conditions_fired)


# ---------------------------------------------------------------------------
# Portfolio drawdown
# ---------------------------------------------------------------------------


def test_portfolio_drawdown_fires_when_exceeded():
    policy = _minimal_policy(
        portfolio_drawdown_reduce_threshold_pct=10.0,
        actions=[_freeze_action(COND_PORTFOLIO_DRAWDOWN)],
    )
    metrics = _metrics(portfolio_drawdown_pct=15.0)
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert any(c.condition_id == COND_PORTFOLIO_DRAWDOWN for c in result.conditions_fired)
    assert result.triggered_actions[0].kind == "freeze_new_entries"


def test_portfolio_drawdown_does_not_fire_below_threshold():
    policy = _minimal_policy(portfolio_drawdown_reduce_threshold_pct=10.0)
    metrics = _metrics(portfolio_drawdown_pct=8.0)
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert not any(c.condition_id == COND_PORTFOLIO_DRAWDOWN for c in result.conditions_fired)


# ---------------------------------------------------------------------------
# Correlation spike
# ---------------------------------------------------------------------------


def test_correlation_spike_fires_when_exceeded():
    policy = _minimal_policy(
        correlation_reduce_threshold=0.7,
        actions=[PortfolioMetaAction(
            condition_id=COND_CORRELATION_SPIKE,
            kind="reduce_gross_exposure_pct",
            params={"reduce_pct": 25.0},
        )],
    )
    metrics = _metrics(max_pairwise_correlation=0.85)
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert any(c.condition_id == COND_CORRELATION_SPIKE for c in result.conditions_fired)
    assert result.triggered_actions[0].kind == "reduce_gross_exposure_pct"


def test_correlation_spike_does_not_fire_at_threshold():
    policy = _minimal_policy(correlation_reduce_threshold=0.7)
    metrics = _metrics(max_pairwise_correlation=0.70)
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert not any(c.condition_id == COND_CORRELATION_SPIKE for c in result.conditions_fired)


# ---------------------------------------------------------------------------
# Hostile regime
# ---------------------------------------------------------------------------


def test_hostile_regime_fires_for_hostile_label():
    policy = _minimal_policy(
        hostile_regime_reduce_enabled=True,
        actions=[PortfolioMetaAction(
            condition_id=COND_HOSTILE_REGIME,
            kind="rebalance_to_cash_pct",
            params={"cash_pct": 50.0},
        )],
    )
    metrics = _metrics(current_regime="hostile")
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert any(c.condition_id == COND_HOSTILE_REGIME for c in result.conditions_fired)
    assert result.triggered_actions[0].kind == "rebalance_to_cash_pct"


def test_hostile_regime_fires_for_crash_label():
    policy = _minimal_policy(hostile_regime_reduce_enabled=True)
    metrics = _metrics(current_regime="crash")
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert any(c.condition_id == COND_HOSTILE_REGIME for c in result.conditions_fired)


def test_hostile_regime_does_not_fire_for_normal():
    policy = _minimal_policy(hostile_regime_reduce_enabled=True)
    metrics = _metrics(current_regime="normal")
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert not any(c.condition_id == COND_HOSTILE_REGIME for c in result.conditions_fired)


def test_hostile_regime_disabled_never_fires():
    policy = _minimal_policy(hostile_regime_reduce_enabled=False)
    metrics = _metrics(current_regime="hostile")
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert not any(c.condition_id == COND_HOSTILE_REGIME for c in result.conditions_fired)


def test_hostile_regime_none_regime_never_fires():
    policy = _minimal_policy(hostile_regime_reduce_enabled=True)
    metrics = _metrics(current_regime=None)
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert not any(c.condition_id == COND_HOSTILE_REGIME for c in result.conditions_fired)


# ---------------------------------------------------------------------------
# Action matching: only fire actions whose condition_id matches
# ---------------------------------------------------------------------------


def test_action_only_fires_for_matching_condition():
    policy = _minimal_policy(
        max_symbol_concentration_pct=30.0,
        portfolio_drawdown_reduce_threshold_pct=50.0,  # high threshold, won't fire
        actions=[
            _trim_action(COND_SYMBOL_CONCENTRATION),
            _freeze_action(COND_PORTFOLIO_DRAWDOWN),
        ],
    )
    metrics = _metrics(symbol_concentrations={"BTC-USD": 60.0}, portfolio_drawdown_pct=5.0)
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert len(result.triggered_actions) == 1
    assert result.triggered_actions[0].condition_id == COND_SYMBOL_CONCENTRATION


def test_no_actions_fired_when_no_conditions_met():
    policy = _minimal_policy(
        max_symbol_concentration_pct=80.0,
        actions=[_trim_action(COND_SYMBOL_CONCENTRATION)],
    )
    metrics = _metrics(symbol_concentrations={"BTC-USD": 10.0})
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert result.triggered_actions == []


def test_multiple_conditions_can_fire():
    policy = _minimal_policy(
        max_symbol_concentration_pct=30.0,
        portfolio_drawdown_reduce_threshold_pct=5.0,
        actions=[
            _trim_action(COND_SYMBOL_CONCENTRATION),
            _freeze_action(COND_PORTFOLIO_DRAWDOWN),
        ],
    )
    metrics = _metrics(
        symbol_concentrations={"BTC-USD": 60.0},
        portfolio_drawdown_pct=10.0,
    )
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    fired_ids = {c.condition_id for c in result.conditions_fired}
    assert COND_SYMBOL_CONCENTRATION in fired_ids
    assert COND_PORTFOLIO_DRAWDOWN in fired_ids
    assert len(result.triggered_actions) == 2


# ---------------------------------------------------------------------------
# Metrics snapshot is captured
# ---------------------------------------------------------------------------


def test_result_includes_metrics_snapshot():
    policy = _minimal_policy()
    metrics = _metrics(total_equity=99_000.0, portfolio_drawdown_pct=3.5)
    result = evaluate_portfolio_risk_conditions(policy, metrics)
    assert result.metrics_snapshot["total_equity"] == pytest.approx(99_000.0)
    assert result.metrics_snapshot["portfolio_drawdown_pct"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# compute_symbol_concentration helper
# ---------------------------------------------------------------------------


def test_compute_symbol_concentration_basic():
    positions = {"BTC-USD": 0.1, "ETH-USD": 2.0}
    prices = {"BTC-USD": 50_000.0, "ETH-USD": 3_000.0}
    total_equity = 11_000.0
    result = compute_symbol_concentration(positions, prices, total_equity)
    # BTC: 0.1 × 50000 = 5000 → 5000/11000 × 100 ≈ 45.45%
    # ETH: 2.0 × 3000 = 6000 → 6000/11000 × 100 ≈ 54.55%
    assert result["BTC-USD"] == pytest.approx(5000 / 11000 * 100, rel=0.01)
    assert result["ETH-USD"] == pytest.approx(6000 / 11000 * 100, rel=0.01)


def test_compute_symbol_concentration_zero_equity_returns_empty():
    result = compute_symbol_concentration({"BTC-USD": 0.1}, {"BTC-USD": 50000.0}, 0.0)
    assert result == {}


def test_compute_symbol_concentration_zero_qty_excluded():
    result = compute_symbol_concentration({"BTC-USD": 0.0}, {"BTC-USD": 50000.0}, 10000.0)
    assert "BTC-USD" not in result


def test_compute_symbol_concentration_missing_price_excluded():
    result = compute_symbol_concentration({"BTC-USD": 0.1}, {}, 10000.0)
    assert "BTC-USD" not in result

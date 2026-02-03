"""Tests for ExperimentSpec schema and experiment-based trade filtering (Runbook 11)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.strategies.risk_engine import RiskEngine
from agents.strategies.trade_risk import TradeRiskEvaluator
from schemas.experiment_spec import ExperimentSpec, ExposureSpec, MetricSpec
from schemas.llm_strategist import (
    IndicatorSnapshot,
    PortfolioState,
    RiskConstraint,
    TriggerCondition,
)
from schemas.strategy_run import LearningBookSettings


def _portfolio() -> PortfolioState:
    return PortfolioState(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        equity=100_000.0,
        cash=100_000.0,
        positions={},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )


def _indicator() -> IndicatorSnapshot:
    return IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=datetime(2024, 1, 1, tzinfo=timezone.utc),
        close=50000.0,
        atr_14=500.0,
    )


def _risk_engine() -> RiskEngine:
    return RiskEngine(
        RiskConstraint(
            max_position_risk_pct=2.0,
            max_symbol_exposure_pct=100.0,
            max_portfolio_exposure_pct=100.0,
            max_daily_loss_pct=10.0,
        ),
        {},
    )


def _learning_settings() -> LearningBookSettings:
    return LearningBookSettings(
        enabled=True,
        notional_usd=500.0,
        daily_risk_budget_pct=10.0,
        max_trades_per_day=20,
    )


def test_experiment_spec_validates_and_roundtrips():
    """ExperimentSpec can be created, serialized, and deserialized."""
    spec = ExperimentSpec(
        experiment_id="exp-001",
        name="Test Experiment",
        description="Testing trend-following on BTC",
        status="draft",
        exposure=ExposureSpec(
            target_symbols=["BTC-USD"],
            trigger_categories=["trend_continuation"],
            max_notional_usd=200.0,
        ),
        metrics=MetricSpec(
            target_metric="sharpe_ratio",
            target_value=1.5,
            min_sample_size=20,
            max_loss_usd=500.0,
        ),
        hypothesis="Trend signals work better with smaller sizing.",
    )

    json_str = spec.to_json()
    restored = ExperimentSpec.from_json(json_str)

    assert restored.experiment_id == "exp-001"
    assert restored.exposure.target_symbols == ["BTC-USD"]
    assert restored.metrics.target_metric == "sharpe_ratio"
    assert restored.status == "draft"


def test_experiment_spec_lifecycle_transitions():
    """Lifecycle transitions follow the allowed state machine."""
    spec = ExperimentSpec(experiment_id="exp-002", name="Lifecycle")

    assert spec.status == "draft"
    assert spec.can_transition("running")
    assert spec.can_transition("cancelled")
    assert not spec.can_transition("completed")
    assert not spec.can_transition("paused")

    spec.transition("running")
    assert spec.status == "running"
    assert spec.can_transition("paused")
    assert spec.can_transition("completed")
    assert spec.can_transition("cancelled")
    assert not spec.can_transition("draft")

    spec.transition("paused")
    assert spec.status == "paused"
    assert spec.can_transition("running")
    assert spec.can_transition("completed")
    assert not spec.can_transition("draft")

    spec.transition("completed")
    assert spec.status == "completed"
    assert not spec.can_transition("running")
    assert not spec.can_transition("draft")

    with pytest.raises(ValueError, match="Cannot transition"):
        spec.transition("running")


def test_experiment_exposure_symbol_filter():
    """When experiment restricts target_symbols, trades for other symbols are blocked."""
    spec = ExperimentSpec(
        experiment_id="exp-003",
        name="BTC-only",
        status="running",
        exposure=ExposureSpec(
            target_symbols=["ETH-USD"],
            max_notional_usd=500.0,
        ),
    )
    evaluator = TradeRiskEvaluator(
        _risk_engine(),
        learning_settings=_learning_settings(),
        experiment_spec=spec,
    )

    # BTC-USD trigger should be blocked (experiment only allows ETH-USD)
    trigger = TriggerCondition(
        id="learn_btc",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        learning_book=True,
        experiment_id="exp-003",
    )
    result = evaluator.evaluate(trigger, "entry", 50000.0, _portfolio(), _indicator())
    assert not result.allowed
    assert result.reason == "experiment_symbol_filter"


def test_experiment_notional_cap():
    """Experiment max_notional_usd caps the trade size even when learning settings allow more."""
    spec = ExperimentSpec(
        experiment_id="exp-004",
        name="Tiny notional",
        status="running",
        exposure=ExposureSpec(
            target_symbols=["BTC-USD"],
            max_notional_usd=50.0,  # Much less than the 500 USD in learning settings
        ),
    )
    settings = _learning_settings()
    settings.notional_usd = 500.0  # Learning allows 500

    evaluator = TradeRiskEvaluator(
        _risk_engine(),
        learning_settings=settings,
        experiment_spec=spec,
    )

    trigger = TriggerCondition(
        id="learn_btc",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        learning_book=True,
        experiment_id="exp-004",
    )
    result = evaluator.evaluate(trigger, "entry", 50000.0, _portfolio(), _indicator())
    assert result.allowed
    # Should be capped at 50 USD notional → qty = 50/50000 = 0.001
    expected_qty = 50.0 / 50000.0
    assert abs(result.quantity - expected_qty) < 1e-9


def test_experiment_category_filter():
    """When experiment restricts trigger_categories, mismatched categories are blocked."""
    spec = ExperimentSpec(
        experiment_id="exp-005",
        name="Reversal-only",
        status="running",
        exposure=ExposureSpec(
            target_symbols=["BTC-USD"],
            trigger_categories=["reversal"],
            max_notional_usd=500.0,
        ),
    )
    evaluator = TradeRiskEvaluator(
        _risk_engine(),
        learning_settings=_learning_settings(),
        experiment_spec=spec,
    )

    # trend_continuation category should be blocked
    trigger = TriggerCondition(
        id="learn_btc",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        learning_book=True,
        experiment_id="exp-005",
    )
    result = evaluator.evaluate(trigger, "entry", 50000.0, _portfolio(), _indicator())
    assert not result.allowed
    assert result.reason == "experiment_category_filter"


def test_non_experiment_learning_trigger_ignores_spec():
    """Learning triggers without experiment_id bypass experiment filtering."""
    spec = ExperimentSpec(
        experiment_id="exp-006",
        name="Restrictive",
        status="running",
        exposure=ExposureSpec(
            target_symbols=["ETH-USD"],  # Only ETH allowed
            max_notional_usd=10.0,
        ),
    )
    evaluator = TradeRiskEvaluator(
        _risk_engine(),
        learning_settings=_learning_settings(),
        experiment_spec=spec,
    )

    # No experiment_id → should bypass experiment filtering
    trigger = TriggerCondition(
        id="learn_btc",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        learning_book=True,
        # No experiment_id
    )
    result = evaluator.evaluate(trigger, "entry", 50000.0, _portfolio(), _indicator())
    assert result.allowed

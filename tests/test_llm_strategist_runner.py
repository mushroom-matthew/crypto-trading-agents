from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from agents.strategies.plan_provider import LLMCostTracker
from agents.strategies.trigger_engine import Order
from backtesting.llm_strategist_runner import LLMStrategistBacktester, PortfolioTracker
from agents.strategies.llm_client import LLMClient
from schemas.llm_strategist import AssetState, IndicatorSnapshot, PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition
from schemas.strategy_run import RiskLimitSettings
from services.strategy_run_registry import StrategyRunRegistry
from tools import execution_tools
from trading_core.execution_engine import ExecutionEngine


class StubPlanProvider:
    def __init__(self, plan: StrategyPlan) -> None:
        self.plan = plan
        self.cost_tracker = LLMCostTracker()
        self.cache_dir = Path(".cache/strategy_plans")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_plan(self, run_id, plan_date, llm_input, prompt_template=None, event_ts=None, emit_events=True):  # noqa: D401
        return self.plan

    def _cache_path(self, run_id, plan_date, llm_input):
        ident = f"{run_id}_{plan_date.isoformat().replace(':', '-')}"
        return self.cache_dir / f"{ident}.json"


def _build_candles(periods: int = 10) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h", tz=timezone.utc)
    data = {
        "timestamp": timestamps,
        "open": [100 + i for i in range(periods)],
        "high": [101 + i for i in range(periods)],
        "low": [99 + i for i in range(periods)],
        "close": [100 + i for i in range(periods)],
        "volume": [1000 + i for i in range(periods)],
    }
    return pd.DataFrame(data).set_index("timestamp")


def _risk_params() -> dict[str, float]:
    return {
        "max_position_risk_pct": 5.0,
        "max_symbol_exposure_pct": 50.0,
        "max_portfolio_exposure_pct": 80.0,
        "max_daily_loss_pct": 3.0,
    }


def test_backtester_executes_trigger(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="buy_breakout",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="not is_flat and (stop_hit or target_hit)",
                stop_loss_pct=99.0,  # Stop at ~1% of price; unreachable on test candles (100-109)
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=5.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0)
        ],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
    run_registry = StrategyRunRegistry(tmp_path / "runs")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=_risk_params(),
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    result = backtester.run(run_id="test-run")
    assert result.final_positions["BTC-USD"] > 0
    assert result.fills.shape[0] == 1
    assert result.daily_reports
    summary = result.daily_reports[-1]
    assert "judge_feedback" in summary
    assert "strategist_constraints" in summary["judge_feedback"]
    assert "plan_limits" in summary
    assert "max_triggers_per_symbol_per_day" in summary["plan_limits"]
    assert "limit_stats" in summary
    assert "risk_limit_hints" in summary["limit_stats"]
    assert "blocked_details" in summary["limit_stats"]
    assert "risk_adjustments" in summary
    assert "overnight_exposure" in summary


def test_rules_refresh_without_replan(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=30),
        global_view="always-in-refresh-test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="always_in_regime",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                category="trend_continuation",
                entry_rule="tf_6h_ema_50 > tf_6h_ema_200 and position == 'flat'",
                exit_rule="not is_flat and (stop_hit or target_hit)",
                # Use price-based stop/target so canonical exit rule can fire.
                # Entry ~hour 6 at close=106; stop=103.88, target=110.24 (2R).
                # Target fires at hour 11 (close=111 > 110.24) — within 18-bar window.
                stop_anchor_type="pct",
                stop_loss_pct=2.0,
                target_anchor_type="r_multiple_2",
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=100.0,
            max_symbol_exposure_pct=100.0,
            max_portfolio_exposure_pct=100.0,
            max_daily_loss_pct=50.0,
            max_daily_risk_budget_pct=100.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=100.0),
        ],
        max_trades_per_day=100,
        max_triggers_per_symbol_per_day=100,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles(periods=18)}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_refresh")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=_risk_params(),
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        replan_on_day_boundary=False,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
        debug_trigger_sample_rate=1.0,
        debug_trigger_max_samples=200,
    )

    refresh_timestamps: list[datetime] = []

    def _dynamic_asset_states(timestamp: datetime):
        refresh_timestamps.append(timestamp)
        hour = int((timestamp - now).total_seconds() // 3600)
        if hour < 6:
            ema50, ema200 = 90.0, 100.0
        elif hour < 12:
            ema50, ema200 = 110.0, 100.0
        else:
            ema50, ema200 = 90.0, 100.0
        snapshot_1h = IndicatorSnapshot(
            symbol="BTC-USD",
            timeframe="1h",
            as_of=timestamp,
            close=100.0 + hour,
            ema_50=100.0,
            ema_200=100.0,
            atr_14=1.0,
        )
        snapshot_6h = IndicatorSnapshot(
            symbol="BTC-USD",
            timeframe="6h",
            as_of=timestamp,
            close=100.0 + hour,
            ema_50=ema50,
            ema_200=ema200,
        )
        return {
            "BTC-USD": AssetState(
                symbol="BTC-USD",
                indicators=[snapshot_1h, snapshot_6h],
                trend_state="uptrend",
                vol_state="normal",
            )
        }

    monkeypatch.setattr(backtester, "_asset_states", _dynamic_asset_states)
    result = backtester.run(run_id="refresh-no-replan")
    strategy_metrics = result.summary["strategy_metrics"]

    assert strategy_metrics["trade_count"] >= 1
    assert result.fills.shape[0] >= 2
    assert len(refresh_timestamps) > 1
    assert result.summary["run_summary"]["stale_context_bars"] == 0

    samples = result.summary.get("trigger_evaluation_samples") or []
    entry_samples = [s for s in samples if s.get("rule_type") == "entry"]
    assert entry_samples
    ema_pairs = {
        (
            (s.get("context_values") or {}).get("tf_6h_ema_50"),
            (s.get("context_values") or {}).get("tf_6h_ema_200"),
        )
        for s in entry_samples
    }
    assert len(ema_pairs) > 1


def test_carry_forward_exit_triggers_for_open_positions():
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    previous_plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="btc_mean_reversion_long_1",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                category="mean_reversion",
                entry_rule="rsi_14 < 40 and position == 'flat'",
                exit_rule="close > sma_short",
                stop_loss_pct=2.0,
            ),
            TriggerCondition(
                id="eth_mean_reversion_long_1",
                symbol="ETH-USD",
                direction="long",
                timeframe="1h",
                category="mean_reversion",
                entry_rule="rsi_14 < 40 and position == 'flat'",
                exit_rule="close > sma_short",
                stop_loss_pct=2.0,
            ),
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=5.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0),
            PositionSizingRule(symbol="ETH-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0),
        ],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD", "ETH-USD"],
        allowed_directions=["long", "short", "exit"],
        allowed_trigger_categories=["trend_continuation", "mean_reversion"],
    )
    current_plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="btc_trend_continuation_short_1",
                symbol="BTC-USD",
                direction="short",
                timeframe="1h",
                category="trend_continuation",
                entry_rule="rsi_14 > 60 and position == 'flat'",
                exit_rule="rsi_14 < 50",
                stop_loss_pct=2.0,
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=5.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0),
            PositionSizingRule(symbol="ETH-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0),
        ],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD", "ETH-USD"],
        allowed_directions=["long", "short", "exit"],
        allowed_trigger_categories=["trend_continuation", "mean_reversion"],
    )

    backtester = LLMStrategistBacktester.__new__(LLMStrategistBacktester)  # type: ignore
    backtester.portfolio = PortfolioTracker(
        initial_cash=1000.0,
        fee_rate=0.0,
    )
    backtester.portfolio.positions = {"BTC-USD": 1.0, "ETH-USD": 2.0}
    backtester.portfolio.position_meta = {
        "BTC-USD": {"entry_category": "mean_reversion", "entry_trigger_id": "btc_mean_reversion_long_1"},
        "ETH-USD": {"entry_category": "mean_reversion", "entry_trigger_id": "eth_mean_reversion_long_1"},
    }

    updated, carried = backtester._carry_forward_exit_triggers(current_plan, previous_plan)
    assert len(carried) == 2
    carried_ids = {entry["carried_trigger_id"] for entry in carried}
    updated_ids = {t.id for t in updated.triggers}
    assert carried_ids.issubset(updated_ids)
    # ensure carried triggers are exit-only
    exit_only = [t for t in updated.triggers if t.id in carried_ids]
    assert all(t.entry_rule == "false" for t in exit_only)


@pytest.mark.parametrize("strict_fixed_caps", [True, False])
def test_cap_state_reports_policy_vs_derived(monkeypatch, tmp_path, strict_fixed_caps):
    monkeypatch.setenv("STRATEGIST_STRICT_FIXED_CAPS", "true" if strict_fixed_caps else "false")
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="buy_breakout",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="not is_flat and close < 0",
                category="trend_continuation",
                stop_loss_pct=2.0,
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=1.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
            max_daily_risk_budget_pct=10.0,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=30,
        max_triggers_per_symbol_per_day=40,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
    run_registry = StrategyRunRegistry(tmp_path / f"runs_cap_state_{strict_fixed_caps}")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    risk_limits = RiskLimitSettings(
        max_position_risk_pct=1.0,
        max_symbol_exposure_pct=50.0,
        max_portfolio_exposure_pct=80.0,
        max_daily_loss_pct=3.0,
        max_daily_risk_budget_pct=10.0,
    )
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=risk_limits,
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    result = backtester.run(run_id=f"cap-state-{strict_fixed_caps}")
    summary = result.daily_reports[-1]
    cap_state = summary.get("cap_state") or {}
    policy = cap_state.get("policy") or {}
    derived = cap_state.get("derived") or {}
    resolved = cap_state.get("resolved") or {}
    flags = cap_state.get("flags") or {}
    assert policy["max_trades_per_day"] == 30
    assert policy["max_triggers_per_symbol_per_day"] == 40
    assert derived["max_trades_per_day"] == 10
    assert derived["max_triggers_per_symbol_per_day"] == 10
    if strict_fixed_caps:
        assert resolved["max_trades_per_day"] == 30
        assert resolved["max_triggers_per_symbol_per_day"] == 40
        assert flags.get("strict_fixed_caps") is True
    else:
        assert resolved["max_trades_per_day"] == 10
        assert resolved["max_triggers_per_symbol_per_day"] == 10
        assert flags.get("strict_fixed_caps") is False
    # Session caps should be derived from resolved caps when present; none configured here.
    assert cap_state.get("session_caps") == {}


def test_exit_orders_map_to_plan_triggers(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="buy_breakout",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="not is_flat and (stop_hit or target_hit)",
                category="trend_continuation",
                # Entry at close=100; stop=98 (2%), target=104 (2R).
                # Target fires at close=105 (candle 6) so exit routing is exercised.
                stop_anchor_type="pct",
                stop_loss_pct=2.0,
                target_anchor_type="r_multiple_2",
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=5.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0)
        ],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long", "flat"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_exit")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=_risk_params(),
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    result = backtester.run(run_id="test-run-exit")
    assert result.fills.shape[0] >= 1
    exit_fills = [reason for reason in result.fills["reason"].tolist() if reason.endswith("_exit")]
    assert exit_fills, "Expected at least one exit fill routed through the execution engine"


def test_flatten_daily_zeroes_overnight_exposure(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="buy_breakout",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="not is_flat and close < 0",
                stop_loss_pct=2.0,
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=5.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0)
        ],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_flatten")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=_risk_params(),
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
        flatten_positions_daily=True,
    )
    result = backtester.run(run_id="test-run-flatten")
    report = result.daily_reports[-1]
    assert report["flatten_positions_daily"] is True
    assert all(abs(entry["quantity"]) < 1e-9 for entry in report["overnight_exposure"].values())


def test_factor_exposures_in_reports(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="buy_breakout",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="not is_flat and close < 0",
                category="trend_continuation",
                stop_loss_pct=2.0,
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=5.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0)
        ],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
    factor_index = list(market_data["BTC-USD"]["1h"].index)
    factor_df = pd.DataFrame({"market": [0.0] * len(factor_index)}, index=factor_index)
    run_registry = StrategyRunRegistry(tmp_path / "runs_factor")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=_risk_params(),
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
        factor_data=factor_df,
        auto_hedge_market=True,
    )
    result = backtester.run(run_id="test-run-factor")
    summary = result.daily_reports[-1]
    assert "factor_exposures" in summary
    assert result.summary["run_summary"].get("factor_exposures") == summary["factor_exposures"]


def test_fee_aware_sizing_allows_full_fraction_entry(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="fee-aware-sizing",
        regime="range",
        triggers=[
            TriggerCondition(
                id="always_in_long",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h' and position == 'flat'",
                exit_rule="not is_flat and close < 0",
                category="trend_continuation",
                stop_loss_pct=2.0,
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=100.0,
            max_symbol_exposure_pct=100.0,
            max_portfolio_exposure_pct=100.0,
            max_daily_loss_pct=100.0,
            max_daily_risk_budget_pct=100.0,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=100.0)],
        max_trades_per_day=100,
        max_triggers_per_symbol_per_day=100,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles(periods=20)}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_fee_aware")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.001,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=RiskLimitSettings(
            max_position_risk_pct=100.0,
            max_symbol_exposure_pct=100.0,
            max_portfolio_exposure_pct=100.0,
            max_daily_loss_pct=100.0,
            max_daily_risk_budget_pct=100.0,
        ),
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    result = backtester.run(run_id="fee-aware-sizing")
    assert result.fills.shape[0] >= 1
    first_fill = result.fills.iloc[0]
    notional = float(first_fill["qty"]) * float(first_fill["price"])
    assert notional <= (1000.0 / 1.001) + 1e-6


def test_failed_portfolio_execution_is_blocked_not_executed(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="execution-atomicity",
        regime="range",
        triggers=[],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=10.0,
            max_symbol_exposure_pct=100.0,
            max_portfolio_exposure_pct=100.0,
            max_daily_loss_pct=100.0,
            max_daily_risk_budget_pct=100.0,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=10.0)],
        max_trades_per_day=100,
        max_triggers_per_symbol_per_day=100,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles(periods=20)}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_execution_atomicity")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.001,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=RiskLimitSettings(
            max_position_risk_pct=10.0,
            max_symbol_exposure_pct=100.0,
            max_portfolio_exposure_pct=100.0,
            max_daily_loss_pct=100.0,
            max_daily_risk_budget_pct=100.0,
        ),
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )

    def _reject_execute(order, market_structure_entry=None):
        backtester.portfolio.last_reject_reason = "insufficient_cash_fee"
        return False

    monkeypatch.setattr(backtester.portfolio, "execute", _reject_execute)
    ts = market_data["BTC-USD"]["1h"].index[0]
    backtester.portfolio.mark_to_market(ts, {"BTC-USD": float(market_data["BTC-USD"]["1h"].iloc[0]["close"])})
    state = backtester.portfolio.portfolio_state(ts)
    order = Order(
        symbol="BTC-USD",
        side="buy",
        quantity=0.5,
        price=100.0,
        timeframe="1h",
        reason="test_rejected_order",
        timestamp=ts,
    )
    executed = backtester._process_orders_with_limits(
        run_id="execution-atomicity",
        day_key=ts.date().isoformat(),
        orders=[order],
        portfolio_state=state,
        plan_payload=None,
        compiled_payload=None,
    )
    assert executed == []
    assert backtester.portfolio.fills == []
    limit_entry = backtester.limit_enforcement_by_day[ts.date().isoformat()]
    assert limit_entry["trades_executed"] == 0
    assert len(limit_entry["executed_details"]) == 0
    assert any(entry.get("reason") == "insufficient_cash_fee" for entry in limit_entry["blocked_details"])

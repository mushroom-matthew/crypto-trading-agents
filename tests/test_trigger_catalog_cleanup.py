from __future__ import annotations

from datetime import datetime, timezone

from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import StrategyPlanProvider
from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition


class _DummyAsset:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


class _DummyInput:
    def __init__(self, symbols: list[str]) -> None:
        self.assets = [_DummyAsset(symbol) for symbol in symbols]


def _risk_params() -> dict[str, float]:
    return {
        "max_position_risk_pct": 2.0,
        "max_symbol_exposure_pct": 25.0,
        "max_portfolio_exposure_pct": 80.0,
        "max_daily_loss_pct": 3.0,
        "max_daily_risk_budget_pct": 3.75,
    }


def test_prunes_dead_exit_variants(tmp_path):
    provider = StrategyPlanProvider(llm_client=LLMClient(allow_fallback=True), cache_dir=tmp_path / "plans", llm_calls_per_day=1)
    plan = StrategyPlan(
        generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        valid_until=datetime(2024, 1, 2, tzinfo=timezone.utc),
        global_view="cleanup",
        regime="range",
        triggers=[
            TriggerCondition(
                id="btc_mean_reversion",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="True",
                exit_rule="False",
                category="mean_reversion",
            ),
            TriggerCondition(
                id="btc_mean_reversion_exit_exit",
                symbol="BTC-USD",
                direction="exit",
                timeframe="1h",
                entry_rule="True",
                exit_rule="True",
                category="mean_reversion",
            ),
        ],
        risk_constraints=RiskConstraint(**_risk_params()),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=None,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long", "exit"],
        allowed_trigger_categories=["mean_reversion"],
    )
    dummy_input = _DummyInput(symbols=["BTC-USD"])
    cleaned = provider._enrich_plan(plan, dummy_input)  # type: ignore[arg-type]
    trigger_ids = {trigger.id for trigger in cleaned.triggers}
    assert "btc_mean_reversion_exit_exit" not in trigger_ids
    assert "btc_mean_reversion" in trigger_ids

from unittest.mock import MagicMock

import pandas as pd

from backtesting.llm_backtest_engine import (
    BacktestPortfolio,
    call_llm_strategy_planner,
    run_backtest,
)
from schemas.strategy_plan import StrategyPlan, LookbackConfig, RiskManagementConfig, EntryRule, ExitRule, Condition, ReplanTriggers, LLMMetadata


def _make_plan(symbol: str = "BTC-USD") -> StrategyPlan:
    return StrategyPlan(
        plan_id="test",
        created_at=pd.Timestamp.utcnow().to_pydatetime(),
        symbol=symbol,
        timeframe="1h",
        lookback=LookbackConfig(preferred_bars=5, min_bars=3, max_bars=10),
        risk_management=RiskManagementConfig(
            max_position_pct=0.2,
            max_daily_loss_pct=0.05,
            max_total_drawdown_pct=0.3,
            per_trade_risk_pct=0.01,
        ),
        entry_rules=[EntryRule(id="trend", direction="long", conditions=[Condition(indicator="ema_fast", operator=">", value="ema_slow")])],
        exit_rules=[ExitRule(id="trend_exit", conditions=[Condition(indicator="ema_fast", operator="<", value="ema_slow")])],
        replan_triggers=ReplanTriggers(),
        llm_metadata=LLMMetadata(model_name="gpt-test", prompt_version="1"),
    )


def test_call_llm_strategy_planner_parses_json(monkeypatch):
    snapshot = {"llm_model": "fake", "symbol": "BTC-USD"}
    fake_response = MagicMock()
    fake_response.output = [MagicMock(content=[MagicMock(text=_make_plan().model_dump_json())])]
    fake_client = MagicMock()
    fake_client.responses.create.return_value = fake_response
    plan = call_llm_strategy_planner(fake_client, snapshot)
    assert plan.symbol == "BTC-USD"


def test_run_backtest_without_future_data(monkeypatch):
    prices = [100 + (i % 20) for i in range(120)]
    data = pd.DataFrame(
        [
            {"open": price - 0.5, "high": price + 1, "low": price - 1, "close": price, "volume": 1000 + i}
            for i, price in enumerate(prices)
        ]
    )
    fake_plan = _make_plan()
    fake_client = MagicMock()
    fake_plan = _make_plan()
    fake_plan.lookback = fake_plan.lookback.model_copy(update={"preferred_bars": 30, "min_bars": 30, "max_bars": 60})
    fake_plan_dict = fake_plan.model_dump_json()
    fake_client.responses.create.return_value = MagicMock(
        output=[MagicMock(content=[MagicMock(text=fake_plan_dict)])]
    )
    portfolio = BacktestPortfolio(cash=1000)
    result = run_backtest(
        market_data=data,
        initial_portfolio=portfolio,
        llm_client=fake_client,
        symbol="BTC-USD",
        timeframe="1h",
        config_bounds={"min_lookback_bars": 30, "max_lookback_bars": 60},
    )
    assert fake_client.responses.create.called
    assert result.final_portfolio.cash <= 1000

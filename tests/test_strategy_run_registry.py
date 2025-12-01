from __future__ import annotations

import pytest

from schemas.judge_feedback import JudgeFeedback, JudgeConstraints
from schemas.strategy_run import StrategyRunConfig
from services.strategy_run_registry import StrategyRunRegistry
from tools import strategy_run_tools


def _config() -> StrategyRunConfig:
    return StrategyRunConfig(
        symbols=["BTC-USD", "ETH-USD"],
        timeframes=["1h", "4h"],
        history_window_days=14,
        plan_cadence_hours=12,
        metadata={"risk_profile": "balanced"},
    )


def test_create_get_and_update_strategy_run(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(_config())
    assert run.run_id.startswith("run_")
    fetched = registry.get_strategy_run(run.run_id)
    assert fetched.config.symbols == ["BTC-USD", "ETH-USD"]

    fetched.current_plan_id = "plan_123"
    registry.update_strategy_run(fetched)
    updated = registry.get_strategy_run(run.run_id)
    assert updated.current_plan_id == "plan_123"
    assert updated.created_at == fetched.created_at
    assert updated.updated_at >= fetched.updated_at


def test_locking_prevents_mutation(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(_config())
    locked = registry.lock_run(run.run_id)
    assert locked.is_locked
    locked.latest_judge_feedback = JudgeFeedback(
        score=55.0,
        constraints=JudgeConstraints(max_trades_per_day=5, risk_mode="normal"),
    )
    with pytest.raises(ValueError):
        registry.update_strategy_run(locked)


def test_strategy_run_tools_create_and_lock(tmp_path, monkeypatch):
    tool_registry = StrategyRunRegistry(tmp_path / "runs")
    monkeypatch.setattr(strategy_run_tools, "registry", tool_registry)

    payload = {
        "symbols": ["BTC-USD"],
        "timeframes": ["1h"],
        "history_window_days": 7,
        "plan_cadence_hours": 24,
        "notes": "test run",
    }
    created = strategy_run_tools.create_strategy_run_tool(payload)
    assert created["run_id"].startswith("run_")
    run_id = created["run_id"]

    fetched = strategy_run_tools.get_strategy_run_tool(run_id)
    assert fetched["config"]["symbols"] == ["BTC-USD"]

    fetched["current_plan_id"] = "plan_abc"
    updated = strategy_run_tools.update_strategy_run_tool(fetched)
    assert updated["current_plan_id"] == "plan_abc"

    locked = strategy_run_tools.lock_run_tool(run_id)
    assert locked["is_locked"] is True

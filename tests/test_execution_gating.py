from datetime import datetime, timedelta, timezone

from agents.execution_config import ExecutionGatingConfig
from agents.workflows.execution_agent_workflow import (
    SymbolDecisionState,
    should_call_llm,
)


def _make_cfg(
    min_move: float = 0.5,
    max_stale: int = 1800,
    max_calls: int = 3,
) -> ExecutionGatingConfig:
    return ExecutionGatingConfig(
        min_price_move_pct=min_move,
        max_staleness_seconds=max_stale,
        max_calls_per_hour_per_symbol=max_calls,
    )


def test_should_call_llm_bootstrap_allows_call() -> None:
    now = datetime.now(timezone.utc)
    state = SymbolDecisionState()
    decision, reason = should_call_llm(100.0, now, state, _make_cfg())
    assert decision
    assert reason == "BOOTSTRAP"


def test_should_call_llm_blocks_small_move() -> None:
    now = datetime.now(timezone.utc)
    state = SymbolDecisionState(
        last_eval_price=100.0,
        last_eval_time=(now - timedelta(minutes=1)).timestamp(),
    )
    decision, reason = should_call_llm(100.1, now, state, _make_cfg(min_move=1.0))
    assert not decision
    assert reason.startswith("PRICE_DELTA")


def test_should_call_llm_triggers_on_price_move() -> None:
    now = datetime.now(timezone.utc)
    state = SymbolDecisionState(
        last_eval_price=100.0,
        last_eval_time=(now - timedelta(minutes=1)).timestamp(),
    )
    decision, reason = should_call_llm(105.0, now, state, _make_cfg(min_move=1.0))
    assert decision
    assert reason == "PRICE_MOVE"


def test_should_call_llm_triggers_on_staleness() -> None:
    now = datetime.now(timezone.utc)
    state = SymbolDecisionState(
        last_eval_price=100.0,
        last_eval_time=(now - timedelta(hours=1)).timestamp(),
    )
    decision, reason = should_call_llm(
        100.0, now, state, _make_cfg(max_stale=60)
    )
    assert decision
    assert reason == "STALE"


def test_should_call_llm_rate_limits_until_window_resets() -> None:
    now = datetime.now(timezone.utc)
    cfg = _make_cfg(max_calls=1)
    state = SymbolDecisionState(
        last_eval_price=100.0,
        last_eval_time=now.timestamp(),
        calls_in_current_window=1,
        current_window_start=now.timestamp(),
    )
    decision, reason = should_call_llm(101.0, now, state, cfg)
    assert not decision
    assert reason == "RATE_LIMIT"

    # Move window forward to ensure reset
    past = now - timedelta(hours=2)
    state.current_window_start = past.timestamp()
    decision, reason = should_call_llm(101.0, now, state, cfg)
    assert decision
    assert reason in {"BOOTSTRAP", "PRICE_MOVE", "STALE"}

from __future__ import annotations

import pytest

from schemas.judge_feedback import DisplayConstraints, JudgeFeedback
from schemas.strategy_run import RiskAdjustmentState, RiskLimitSettings, StrategyRun, StrategyRunConfig
from services.risk_adjustment_service import (
    apply_judge_risk_feedback,
    effective_risk_limits,
    multiplier_from_instruction,
    build_risk_profile,
)


def _run() -> StrategyRun:
    return StrategyRun(
        run_id="run_test",
        config=StrategyRunConfig(
            symbols=["BTC-USD"],
            timeframes=["1h"],
            history_window_days=7,
            risk_limits=RiskLimitSettings(max_position_risk_pct=2.0),
        ),
    )


def test_apply_feedback_tracks_and_restores_adjustments():
    run = _run()
    feedback = JudgeFeedback(
        strategist_constraints=DisplayConstraints(
            sizing_adjustments={"BTC-USD": "Cut risk by 25% until two winning days post drawdown."}
        )
    )
    changed = apply_judge_risk_feedback(run, feedback, winning_day=False)
    assert changed is True
    assert "BTC-USD" in run.risk_adjustments
    state = run.risk_adjustments["BTC-USD"]
    assert state.multiplier == pytest.approx(0.75)
    limits = effective_risk_limits(run)
    assert limits.max_position_risk_pct == pytest.approx(1.5)
    no_instruction_feedback = JudgeFeedback(strategist_constraints=DisplayConstraints())
    apply_judge_risk_feedback(run, no_instruction_feedback, winning_day=True)
    assert run.risk_adjustments["BTC-USD"].wins_progress == 1
    apply_judge_risk_feedback(run, no_instruction_feedback, winning_day=True)
    assert "BTC-USD" not in run.risk_adjustments
    assert effective_risk_limits(run).max_position_risk_pct == pytest.approx(2.0)


def test_multiplier_parser_handles_cap():
    assert multiplier_from_instruction("Cap risk at 10% until calm returns.") == pytest.approx(0.10)
    assert multiplier_from_instruction("Allow full allocation for grade A setups.") == pytest.approx(1.0)


def test_structured_multiplier_overrides_instruction():
    run = _run()
    feedback = JudgeFeedback(
        constraints={"symbol_risk_multipliers": {"BTC-USD": 0.8}},
        strategist_constraints=DisplayConstraints(
            sizing_adjustments={"BTC-USD": "Cut risk by 50% until two winning days post drawdown."}
        ),
    )
    apply_judge_risk_feedback(run, feedback, winning_day=False)
    state = run.risk_adjustments["BTC-USD"]
    assert state.multiplier == pytest.approx(0.8)
    assert feedback.constraints.symbol_risk_multipliers["BTC-USD"] == pytest.approx(0.8)


def test_multiplier_clamped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JUDGE_RISK_MULTIPLIER_MIN", "0.25")
    monkeypatch.setenv("JUDGE_RISK_MULTIPLIER_MAX", "3.0")

    run = _run()
    feedback = JudgeFeedback(
        constraints={"symbol_risk_multipliers": {"BTC-USD": 0.01}},
        strategist_constraints=DisplayConstraints(),
    )
    apply_judge_risk_feedback(run, feedback, winning_day=False)
    assert run.risk_adjustments["BTC-USD"].multiplier == pytest.approx(0.25)
    assert feedback.constraints.symbol_risk_multipliers["BTC-USD"] == pytest.approx(0.25)


def test_build_risk_profile_maps_adjustments():
    run = _run()
    run.risk_adjustments = {
        "BTC-USD": RiskAdjustmentState(multiplier=0.5, instruction="Cut"),
        "ETH-USD": RiskAdjustmentState(multiplier=0.8, instruction="Trim"),
    }
    profile = build_risk_profile(run)
    assert profile.global_multiplier == pytest.approx(0.5)
    assert profile.multiplier_for("BTC-USD") == pytest.approx(0.5)
    assert profile.multiplier_for("ETH-USD") == pytest.approx(0.8)
    assert profile.multiplier_for("LTC-USD") == pytest.approx(0.5)

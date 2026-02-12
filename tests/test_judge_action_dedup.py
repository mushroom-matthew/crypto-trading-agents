"""Runbook 36 — Judge action de-duplication tests.

Verifies that when multiple judge actions arrive from the same eval window,
only the last one persists and earlier ones are superseded with an event.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from schemas.judge_feedback import JudgeAction, JudgeConstraints, DisplayConstraints


def _make_action(
    action_id: str = "a1",
    source_eval_id: str = "2024-01-01",
    recommended_action: str = "hold",
    status: str = "applied",
    ttl_evals: int = 3,
    evals_remaining: int = 3,
) -> JudgeAction:
    return JudgeAction(
        action_id=action_id,
        source_eval_id=source_eval_id,
        recommended_action=recommended_action,
        constraints=JudgeConstraints(),
        strategist_constraints=DisplayConstraints(),
        status=status,
        ttl_evals=ttl_evals,
        evals_remaining=evals_remaining,
        scope="intraday",
    )


class FakeRunner:
    """Minimal stub mimicking LLMStrategistBacktester for _dedup_judge_action."""

    def __init__(self):
        self.active_judge_action: JudgeAction | None = None
        self.events: list[tuple[str, dict]] = []
        self.persisted: list[JudgeAction] = []

    def _persist_judge_action(self, run_id: str, action: JudgeAction) -> None:
        self.persisted.append(action)

    def _emit_event(
        self,
        event_type: str,
        payload: dict,
        *,
        run_id: str,
        correlation_id: str | None = None,
        event_ts: datetime | None = None,
    ) -> None:
        self.events.append((event_type, payload))

    # Import the real method we're testing
    from backtesting.llm_strategist_runner import LLMStrategistBacktester
    _dedup_judge_action = LLMStrategistBacktester._dedup_judge_action


def test_same_eval_last_action_wins():
    """Two actions from same eval window — last one supersedes the first."""
    runner = FakeRunner()

    # First action is already active
    action1 = _make_action(action_id="a1", source_eval_id="day-2024-01-01")
    runner.active_judge_action = action1

    # Second action from same eval
    action2 = _make_action(action_id="a2", source_eval_id="day-2024-01-01", recommended_action="replan")

    superseded = runner._dedup_judge_action(action2, run_id="run1")
    assert superseded is not None
    assert superseded.action_id == "a1"
    assert superseded.status == "expired"
    assert superseded.evals_remaining == 0
    # Active action should be cleared (will be set by _apply_judge_action later)
    assert runner.active_judge_action is None


def test_different_eval_both_apply():
    """Actions from different evals don't conflict — no dedup."""
    runner = FakeRunner()

    action1 = _make_action(action_id="a1", source_eval_id="day-2024-01-01")
    runner.active_judge_action = action1

    action2 = _make_action(action_id="a2", source_eval_id="day-2024-01-02")

    superseded = runner._dedup_judge_action(action2, run_id="run1")
    assert superseded is None
    # Original action untouched
    assert runner.active_judge_action.action_id == "a1"
    assert runner.active_judge_action.status == "applied"


def test_superseded_event_emitted():
    """Superseded action generates a judge_action_superseded event."""
    runner = FakeRunner()

    action1 = _make_action(action_id="a1", source_eval_id="eval-42", recommended_action="hold")
    runner.active_judge_action = action1

    action2 = _make_action(action_id="a2", source_eval_id="eval-42", recommended_action="replan")

    runner._dedup_judge_action(action2, run_id="run1")

    # Check event was emitted
    assert len(runner.events) == 1
    event_type, payload = runner.events[0]
    assert event_type == "judge_action_superseded"
    assert payload["superseded_action_id"] == "a1"
    assert payload["new_action_id"] == "a2"
    assert payload["source_eval_id"] == "eval-42"
    assert payload["superseded_recommended"] == "hold"
    assert payload["new_recommended"] == "replan"


def test_no_dedup_when_no_active_action():
    """No active action means no dedup."""
    runner = FakeRunner()
    runner.active_judge_action = None

    action = _make_action(action_id="a1", source_eval_id="eval-1")
    superseded = runner._dedup_judge_action(action, run_id="run1")

    assert superseded is None
    assert len(runner.events) == 0


def test_no_dedup_when_active_action_expired():
    """Expired active action should not trigger dedup even with same eval."""
    runner = FakeRunner()

    expired = _make_action(action_id="a0", source_eval_id="eval-1", status="expired", evals_remaining=0)
    runner.active_judge_action = expired

    action = _make_action(action_id="a1", source_eval_id="eval-1")
    superseded = runner._dedup_judge_action(action, run_id="run1")

    assert superseded is None
    assert len(runner.events) == 0

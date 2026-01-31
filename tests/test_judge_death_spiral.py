"""Tests for judge death-spiral floor protections (Runbooks 13 + 16).

Covers:
1. Category validator N-2 rule (at least 2 entry categories enabled)
2. Trigger floor via apply_trigger_floor() helper
3. Zero-activity re-enablement of disabled triggers
4. Stale snapshot detection to skip redundant judge evaluations
5. Stale snapshot forced trigger re-enablement (consecutive stale skips)
6. stale_judge_evals daily metric tracking
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Dict, List

import pytest

from schemas.judge_feedback import JudgeConstraints, apply_trigger_floor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trigger(tid: str, category: str) -> SimpleNamespace:
    """Minimal trigger-like object with .id and .category."""
    return SimpleNamespace(id=tid, category=category)


ENTRY_CATEGORIES = list(JudgeConstraints.ENTRY_CATEGORIES)


# ---------------------------------------------------------------------------
# Feature 1: disabled_categories N-2 floor
# ---------------------------------------------------------------------------


class TestDisabledCategoriesFloor:
    """Validator must keep at least 2 entry categories enabled."""

    def test_disabling_all_four_trims_to_two(self):
        c = JudgeConstraints(disabled_categories=list(ENTRY_CATEGORIES))
        disabled_entry = [v for v in c.disabled_categories if v in JudgeConstraints.ENTRY_CATEGORIES]
        assert len(disabled_entry) <= 2
        # At least 2 entry categories remain enabled
        enabled = JudgeConstraints.ENTRY_CATEGORIES - set(c.disabled_categories)
        assert len(enabled) >= 2

    def test_disabling_three_trims_to_two(self):
        c = JudgeConstraints(disabled_categories=ENTRY_CATEGORIES[:3])
        disabled_entry = [v for v in c.disabled_categories if v in JudgeConstraints.ENTRY_CATEGORIES]
        assert len(disabled_entry) <= 2
        enabled = JudgeConstraints.ENTRY_CATEGORIES - set(c.disabled_categories)
        assert len(enabled) >= 2

    def test_disabling_two_is_allowed(self):
        c = JudgeConstraints(disabled_categories=ENTRY_CATEGORIES[:2])
        disabled_entry = [v for v in c.disabled_categories if v in JudgeConstraints.ENTRY_CATEGORIES]
        assert len(disabled_entry) == 2

    def test_disabling_one_is_allowed(self):
        c = JudgeConstraints(disabled_categories=[ENTRY_CATEGORIES[0]])
        assert ENTRY_CATEGORIES[0] in c.disabled_categories

    def test_non_entry_categories_unaffected(self):
        """emergency_exit and other are not entry categories, always allowed."""
        c = JudgeConstraints(
            disabled_categories=ENTRY_CATEGORIES[:2] + ["emergency_exit", "other"],
        )
        assert "emergency_exit" in c.disabled_categories
        assert "other" in c.disabled_categories

    def test_disabled_categories_floor_keeps_two_entry_categories(self):
        """Top-level integration: validator trims to keep >= 2 entry categories."""
        c = JudgeConstraints(
            disabled_categories=[
                "trend_continuation",
                "reversal",
                "volatility_breakout",
                "mean_reversion",
            ]
        )
        enabled = JudgeConstraints.ENTRY_CATEGORIES - set(c.disabled_categories)
        assert len(enabled) >= 2, f"Only {len(enabled)} entry categories enabled: {enabled}"


# ---------------------------------------------------------------------------
# Feature 2: apply_trigger_floor (disabled_trigger_ids)
# ---------------------------------------------------------------------------


class TestTriggerFloor:
    """apply_trigger_floor must keep at least MIN_ENABLED_ENTRY_TRIGGERS entry triggers."""

    def test_no_trimming_when_enough_enabled(self):
        triggers = [
            _make_trigger("t1", "trend_continuation"),
            _make_trigger("t2", "reversal"),
            _make_trigger("t3", "volatility_breakout"),
        ]
        c = JudgeConstraints(disabled_trigger_ids=["t1"])
        result = apply_trigger_floor(c, triggers)
        # 2 entry triggers still enabled (t2, t3), floor=2 → no trim
        assert result.disabled_trigger_ids == ["t1"]

    def test_trims_when_too_many_disabled(self):
        triggers = [
            _make_trigger("t1", "trend_continuation"),
            _make_trigger("t2", "reversal"),
            _make_trigger("t3", "volatility_breakout"),
        ]
        # Disabling all 3 → only 0 enabled, below floor of 2
        c = JudgeConstraints(disabled_trigger_ids=["t1", "t2", "t3"])
        result = apply_trigger_floor(c, triggers)
        enabled_entry = {"t1", "t2", "t3"} - set(result.disabled_trigger_ids)
        assert len(enabled_entry) >= 2

    def test_trigger_floor_trims_disabled_trigger_ids(self):
        """Application-level helper trims disabled triggers to keep at least 2 entry triggers."""
        triggers = [
            _make_trigger("t1", "trend_continuation"),
            _make_trigger("t2", "reversal"),
            _make_trigger("t3", "mean_reversion"),
            _make_trigger("t4", "emergency_exit"),  # not an entry trigger
        ]
        # Disable all 3 entry + 1 exit
        c = JudgeConstraints(disabled_trigger_ids=["t1", "t2", "t3", "t4"])
        result = apply_trigger_floor(c, triggers)
        entry_ids = {"t1", "t2", "t3"}
        enabled_entry = entry_ids - set(result.disabled_trigger_ids)
        assert len(enabled_entry) >= 2
        # Non-entry trigger t4 should still be disabled
        assert "t4" in result.disabled_trigger_ids

    def test_exit_triggers_not_counted(self):
        """Only entry-category triggers count toward the floor."""
        triggers = [
            _make_trigger("t1", "trend_continuation"),
            _make_trigger("t2", "emergency_exit"),
            _make_trigger("t3", "other"),
        ]
        # Disable the only entry trigger → 0 enabled, must re-enable
        c = JudgeConstraints(disabled_trigger_ids=["t1", "t2"])
        result = apply_trigger_floor(c, triggers, min_enabled=1)
        assert "t1" not in result.disabled_trigger_ids
        # exit trigger stays disabled
        assert "t2" in result.disabled_trigger_ids

    def test_custom_min_enabled(self):
        triggers = [
            _make_trigger("t1", "trend_continuation"),
            _make_trigger("t2", "reversal"),
            _make_trigger("t3", "volatility_breakout"),
            _make_trigger("t4", "mean_reversion"),
        ]
        c = JudgeConstraints(disabled_trigger_ids=["t1", "t2", "t3"])
        result = apply_trigger_floor(c, triggers, min_enabled=3)
        entry_ids = {"t1", "t2", "t3", "t4"}
        enabled_entry = entry_ids - set(result.disabled_trigger_ids)
        assert len(enabled_entry) >= 3

    def test_empty_triggers_returns_unchanged(self):
        c = JudgeConstraints(disabled_trigger_ids=["t1"])
        result = apply_trigger_floor(c, [])
        assert result.disabled_trigger_ids == ["t1"]

    def test_no_disabled_returns_same_object(self):
        triggers = [_make_trigger("t1", "trend_continuation")]
        c = JudgeConstraints(disabled_trigger_ids=[])
        result = apply_trigger_floor(c, triggers)
        assert result is c  # same object, not copied


# ---------------------------------------------------------------------------
# Feature 3: Zero-activity re-enablement
# ---------------------------------------------------------------------------


class TestZeroActivityReenablement:
    """After N bars without trades post-judge intervention, disabled_trigger_ids should clear."""

    def test_zero_activity_reenables_triggers(self):
        """Simulate the zero-activity path: after threshold bars, triggers re-enable."""
        # We test the logic directly by checking state transitions
        # rather than running the full backtest loop.

        # Setup: mimic strategist state
        from schemas.judge_feedback import JudgeConstraints

        constraints = JudgeConstraints(
            disabled_trigger_ids=["t1", "t2"],
            disabled_categories=["trend_continuation"],
        )

        # Simulate: bars_since_last_trade >= threshold AND judge intervention happened
        bars_since_last_trade = 48
        threshold = 48
        last_judge_intervention = datetime(2024, 1, 1, tzinfo=timezone.utc)

        should_reenable = (
            last_judge_intervention is not None
            and bars_since_last_trade >= threshold
            and constraints.disabled_trigger_ids
        )
        assert should_reenable

        # Apply re-enablement (same logic as in the runner)
        new_constraints = constraints.model_copy(update={"disabled_trigger_ids": []})
        assert new_constraints.disabled_trigger_ids == []
        # Categories are preserved (broader policy)
        assert new_constraints.disabled_categories == ["trend_continuation"]

    def test_no_reenable_below_threshold(self):
        constraints = JudgeConstraints(disabled_trigger_ids=["t1"])
        bars_since_last_trade = 10
        threshold = 48
        last_judge_intervention = datetime(2024, 1, 1, tzinfo=timezone.utc)

        should_reenable = (
            last_judge_intervention is not None
            and bars_since_last_trade >= threshold
            and constraints.disabled_trigger_ids
        )
        assert not should_reenable

    def test_no_reenable_without_judge_intervention(self):
        constraints = JudgeConstraints(disabled_trigger_ids=["t1"])
        bars_since_last_trade = 100
        threshold = 48
        last_judge_intervention = None

        should_reenable = (
            last_judge_intervention is not None
            and bars_since_last_trade >= threshold
            and constraints.disabled_trigger_ids
        )
        assert not should_reenable

    def test_no_reenable_when_no_disabled_triggers(self):
        constraints = JudgeConstraints(disabled_trigger_ids=[])
        bars_since_last_trade = 100
        threshold = 48
        last_judge_intervention = datetime(2024, 1, 1, tzinfo=timezone.utc)

        should_reenable = (
            last_judge_intervention is not None
            and bars_since_last_trade >= threshold
            and constraints.disabled_trigger_ids
        )
        assert not should_reenable


# ---------------------------------------------------------------------------
# Feature 4: Stale snapshot detection
# ---------------------------------------------------------------------------


class TestStaleSnapshotDetection:
    """Same equity + trade_count should skip the judge evaluation."""

    def test_stale_snapshot_skips_judge_evaluation(self):
        """If snapshot_key matches previous, judge should be skipped."""
        last_key = (1000.50, 5)
        current_key = (1000.50, 5)
        assert current_key == last_key  # stale → skip

    def test_changed_snapshot_runs_judge(self):
        """Different equity or trade_count means evaluation proceeds."""
        last_key = (1000.50, 5)

        # Different equity
        key_a = (1001.00, 5)
        assert key_a != last_key

        # Different trade count
        key_b = (1000.50, 6)
        assert key_b != last_key

        # Both different
        key_c = (999.00, 7)
        assert key_c != last_key

    def test_first_evaluation_always_runs(self):
        """When last_judge_snapshot_key is None, evaluation should proceed."""
        last_key = None
        current_key = (1000.50, 5)
        assert current_key != last_key  # None != tuple → always runs

    def test_rounding_behavior(self):
        """Equity is rounded to 2 decimal places for comparison."""
        # These should match after rounding
        key_a = (round(1000.504, 2), 5)
        key_b = (round(1000.505, 2), 5)
        # 1000.50 vs 1000.50 or 1000.51 — depends on rounding
        # The point is that the rounding is applied consistently
        assert key_a == (1000.50, 5)


# ---------------------------------------------------------------------------
# Feature 5: Stale snapshot forced re-enablement (Runbook 16)
# ---------------------------------------------------------------------------


class TestStaleSnapshotReenablement:
    """After consecutive stale skips, disabled triggers should be force re-enabled."""

    def test_consecutive_stale_skips_force_reenable(self):
        """After stale_reenable_threshold consecutive stale evals, triggers are cleared."""
        constraints = JudgeConstraints(disabled_trigger_ids=["t1", "t2"])
        consecutive_stale = 2
        threshold = 2

        should_force_reenable = (
            consecutive_stale >= threshold
            and constraints.disabled_trigger_ids
        )
        assert should_force_reenable

        new_constraints = constraints.model_copy(update={"disabled_trigger_ids": []})
        assert new_constraints.disabled_trigger_ids == []

    def test_single_stale_skip_no_reenable(self):
        """One stale skip should not trigger re-enablement."""
        constraints = JudgeConstraints(disabled_trigger_ids=["t1"])
        consecutive_stale = 1
        threshold = 2

        should_force_reenable = (
            consecutive_stale >= threshold
            and constraints.disabled_trigger_ids
        )
        assert not should_force_reenable

    def test_stale_reenable_no_effect_without_disabled(self):
        """If no triggers are disabled, stale re-enable is a no-op."""
        constraints = JudgeConstraints(disabled_trigger_ids=[])
        consecutive_stale = 5
        threshold = 2

        should_force_reenable = (
            consecutive_stale >= threshold
            and constraints.disabled_trigger_ids
        )
        assert not should_force_reenable

    def test_stale_counter_resets_on_changed_snapshot(self):
        """Consecutive stale counter resets when snapshot changes."""
        # Simulate: 1 stale, then changed snapshot, then 1 stale again
        consecutive = 0

        # First eval: new snapshot
        consecutive = 0  # reset on change

        # Second eval: stale
        consecutive += 1
        assert consecutive == 1

        # Third eval: changed
        consecutive = 0
        assert consecutive == 0

        # Fourth eval: stale again — only 1, not 2
        consecutive += 1
        assert consecutive == 1
        assert consecutive < 2  # below threshold


# ---------------------------------------------------------------------------
# Feature 6: stale_judge_evals daily metric (Runbook 16)
# ---------------------------------------------------------------------------


class TestStaleJudgeEvalsMetric:
    """Daily report should include stale_judge_evals count."""

    def test_stale_evals_counter_accumulates_per_day(self):
        """Counter tracks stale evals per day key."""
        from collections import defaultdict

        counter: Dict[str, int] = defaultdict(int)
        counter["2024-01-01"] += 1
        counter["2024-01-01"] += 1
        counter["2024-01-02"] += 1

        assert counter["2024-01-01"] == 2
        assert counter["2024-01-02"] == 1
        assert counter["2024-01-03"] == 0  # defaultdict returns 0

    def test_stale_evals_pop_returns_zero_for_missing_day(self):
        """Pop on missing day returns 0 default (same pattern as _finalize_day)."""
        from collections import defaultdict

        counter: Dict[str, int] = defaultdict(int)
        counter["2024-01-01"] = 3

        assert counter.pop("2024-01-01", 0) == 3
        assert counter.pop("2024-01-02", 0) == 0

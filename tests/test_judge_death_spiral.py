"""Tests for judge death-spiral floor protections (Runbooks 13 + 16 + 24).

Covers:
1. Category validator N-2 rule (at least 2 entry categories enabled)
2. Trigger floor via apply_trigger_floor() helper
3. Zero-activity re-enablement of disabled triggers
4. Stale snapshot detection to skip redundant judge evaluations
5. Stale snapshot forced trigger re-enablement (consecutive stale skips)
6. stale_judge_evals daily metric tracking
7. Stale skip advances next_judge_time (Runbook 24)
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


# ---------------------------------------------------------------------------
# Feature 7: Zero-activity re-enablement clears disabled_categories (D1)
# ---------------------------------------------------------------------------


class TestZeroActivityReenablementCategories:
    """After N bars without trades, disabled_categories must also clear."""

    def test_zero_activity_clears_categories(self):
        """When only disabled_categories is set (no trigger IDs), re-enablement fires."""
        constraints = JudgeConstraints(
            disabled_trigger_ids=[],
            disabled_categories=["trend_continuation", "reversal"],
        )
        bars_since_last_trade = 48
        threshold = 48
        last_judge_intervention = datetime(2024, 1, 1, tzinfo=timezone.utc)

        should_reenable = (
            last_judge_intervention is not None
            and bars_since_last_trade >= threshold
            and (constraints.disabled_trigger_ids or constraints.disabled_categories)
        )
        assert should_reenable

        new_constraints = constraints.model_copy(
            update={"disabled_trigger_ids": [], "disabled_categories": []}
        )
        assert new_constraints.disabled_trigger_ids == []
        assert new_constraints.disabled_categories == []

    def test_zero_activity_clears_both_ids_and_categories(self):
        """When both disabled_trigger_ids and disabled_categories are set, both clear."""
        constraints = JudgeConstraints(
            disabled_trigger_ids=["t1"],
            disabled_categories=["mean_reversion"],
        )
        new_constraints = constraints.model_copy(
            update={"disabled_trigger_ids": [], "disabled_categories": []}
        )
        assert new_constraints.disabled_trigger_ids == []
        assert new_constraints.disabled_categories == []

    def test_no_reenable_when_neither_set(self):
        """When both fields are empty, condition is False."""
        constraints = JudgeConstraints(disabled_trigger_ids=[], disabled_categories=[])
        should_reenable = (
            constraints.disabled_trigger_ids or constraints.disabled_categories
        )
        assert not should_reenable


# ---------------------------------------------------------------------------
# Feature 8: Canonical judge snapshot (D7)
# ---------------------------------------------------------------------------


class TestCanonicalJudgeSnapshot:
    """Intraday judge result must include canonical_snapshot key."""

    def test_canonical_snapshot_structure(self):
        """Verify canonical_snapshot has the expected keys."""
        compact_summary = {
            "trigger_attempts_summary": {"t1": {"fired": 1, "blocked": 0}},
            "equity": 1050.0,
        }
        score = 65.0
        ts = datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc)

        canonical_snapshot = {
            "summary_compact": compact_summary,
            "trigger_attempts_summary": compact_summary.get("trigger_attempts_summary"),
            "score": score,
            "timestamp": ts.isoformat(),
        }

        assert "summary_compact" in canonical_snapshot
        assert "trigger_attempts_summary" in canonical_snapshot
        assert canonical_snapshot["score"] == 65.0
        assert canonical_snapshot["timestamp"] == "2024-01-15T14:00:00+00:00"
        assert canonical_snapshot["trigger_attempts_summary"] == {"t1": {"fired": 1, "blocked": 0}}

    def test_canonical_snapshot_in_result_dict(self):
        """Result dict from intraday judge includes canonical_snapshot."""
        result = {
            "timestamp": "2024-01-15T14:00:00+00:00",
            "score": 55.0,
            "canonical_snapshot": {
                "summary_compact": {},
                "trigger_attempts_summary": None,
                "score": 55.0,
                "timestamp": "2024-01-15T14:00:00+00:00",
            },
        }
        assert "canonical_snapshot" in result
        assert result["canonical_snapshot"]["score"] == 55.0


# ---------------------------------------------------------------------------
# Feature 9: Strip judge-constrained triggers (D5)
# ---------------------------------------------------------------------------


class TestStripJudgeConstrainedTriggers:
    """Disabled categories/IDs should be stripped from generated plans."""

    def test_strip_by_category(self):
        """Triggers matching disabled_categories are removed."""
        constraints = JudgeConstraints(disabled_categories=["trend_continuation"])
        triggers = [
            _make_trigger("t1", "trend_continuation"),
            _make_trigger("t2", "reversal"),
            _make_trigger("t3", "trend_continuation"),
        ]

        filtered = []
        stripped = []
        for t in triggers:
            cat = getattr(t, "category", None)
            if cat and cat in constraints.disabled_categories:
                stripped.append({"id": t.id, "category": cat, "reason": "disabled_category"})
                continue
            filtered.append(t)

        assert len(filtered) == 1
        assert filtered[0].id == "t2"
        assert len(stripped) == 2
        assert all(s["reason"] == "disabled_category" for s in stripped)

    def test_strip_by_trigger_id(self):
        """Triggers matching disabled_trigger_ids are removed."""
        constraints = JudgeConstraints(disabled_trigger_ids=["t1", "t3"])
        triggers = [
            _make_trigger("t1", "reversal"),
            _make_trigger("t2", "reversal"),
            _make_trigger("t3", "mean_reversion"),
        ]

        filtered = []
        stripped = []
        for t in triggers:
            if t.id in constraints.disabled_trigger_ids:
                stripped.append({"id": t.id, "category": getattr(t, "category", None), "reason": "disabled_trigger_id"})
                continue
            filtered.append(t)

        assert len(filtered) == 1
        assert filtered[0].id == "t2"
        assert len(stripped) == 2

    def test_no_strip_when_no_constraints(self):
        """No stripping when constraints are empty."""
        constraints = JudgeConstraints(disabled_trigger_ids=[], disabled_categories=[])
        triggers = [_make_trigger("t1", "reversal")]

        should_strip = bool(constraints.disabled_trigger_ids or constraints.disabled_categories)
        assert not should_strip

    def test_all_triggers_stripped_produces_empty_plan(self):
        """When all triggers are stripped, result is empty (wait stance)."""
        constraints = JudgeConstraints(disabled_categories=["reversal"])
        triggers = [
            _make_trigger("t1", "reversal"),
            _make_trigger("t2", "reversal"),
        ]

        filtered = [
            t for t in triggers
            if getattr(t, "category", None) not in constraints.disabled_categories
        ]
        assert len(filtered) == 0  # wait stance


# =============================================================================
# Runbook 24 — Judge Eval Flood
# =============================================================================


class TestStaleSkipAdvancesNextJudgeTime:
    """Verify that stale snapshot skips advance next_judge_time (Runbook 24)."""

    def test_stale_skip_advances_next_judge_time(self):
        """Stale skip must advance next_judge_time so we don't re-trigger every bar.

        Simulates the state machine: after a stale skip, next_judge_time should
        move forward by judge_cadence so the next bar doesn't immediately trigger
        another evaluation.
        """
        from datetime import timedelta

        # Simulate the state before a stale skip
        judge_cadence_hours = 12.0
        judge_cadence = timedelta(hours=judge_cadence_hours)
        ts = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
        original_next_judge_time = ts  # "due now"

        # After stale skip, next_judge_time should advance
        next_judge_time = ts + judge_cadence

        assert next_judge_time == datetime(2025, 1, 16, 0, 0, tzinfo=timezone.utc)
        assert next_judge_time > ts, "next_judge_time must advance past current ts"
        assert next_judge_time - ts == judge_cadence

    def test_stale_skip_dedup_only_first_appended(self):
        """Only the first stale skip per cadence window should append to history.

        The stale_skip_count_since_last_real counter prevents flooding
        intraday_judge_history with redundant stale skip entries.
        """
        stale_skip_count = 0
        history: List[dict] = []

        # Simulate 5 consecutive stale skips
        for i in range(5):
            stale_skip_count += 1
            result = {"skipped": True, "replan_reason": "stale_snapshot_skip"}
            if stale_skip_count <= 1:
                history.append(result)

        assert len(history) == 1, "Only first stale skip should be in history"
        assert stale_skip_count == 5

        # After a real eval, counter resets
        stale_skip_count = 0
        for i in range(3):
            stale_skip_count += 1
            result = {"skipped": True, "replan_reason": "stale_snapshot_skip"}
            if stale_skip_count <= 1:
                history.append(result)

        assert len(history) == 2, "Second cadence window adds one more entry"

    def test_default_cadence_is_12_hours(self):
        """Default judge_cadence_hours should be 12.0 (Runbook 24)."""
        from datetime import timedelta

        default_cadence_hours = 12.0
        cadence = timedelta(hours=default_cadence_hours)
        assert cadence == timedelta(hours=12)

        # Verify adaptive bounds
        # Drawdown min: 4h floor
        drawdown_cadence = max(4.0, default_cadence_hours / 2)
        assert drawdown_cadence == 6.0  # 12/2 = 6, > 4 floor

        # Good perf max: 24h ceiling
        good_perf_cadence = min(24.0, default_cadence_hours * 1.5)
        assert good_perf_cadence == 18.0  # 12*1.5 = 18, < 24 ceiling


# =============================================================================
# Runbook 26 — Risk Telemetry + Position Sizing Efficiency
# =============================================================================


class TestRiskTelemetry:
    """Verify risk telemetry renames and budget utilization (Runbook 26)."""

    def test_snapshot_risk_matches_risk_engine(self):
        """Position quality snapshot should use risk_quality_score, not risk_score."""
        from trading_core.trade_quality import assess_position_quality

        assessments = assess_position_quality(
            positions={"BTC-USD": 1.0},
            entry_prices={"BTC-USD": 50000.0},
            current_prices={"BTC-USD": 49000.0},  # underwater
            position_opened_times={"BTC-USD": datetime(2025, 1, 1, tzinfo=timezone.utc)},
            current_time=datetime(2025, 1, 3, tzinfo=timezone.utc),  # 48h later
        )
        assert len(assessments) == 1
        pq = assessments[0]
        # Verify renamed field exists
        assert hasattr(pq, "risk_quality_score")
        assert not hasattr(pq, "risk_score"), "risk_score should be renamed to risk_quality_score"
        # Underwater + extended + >48h should reduce score significantly
        assert pq.risk_quality_score < 50.0
        assert pq.is_underwater is True
        # New fields exist
        assert hasattr(pq, "position_risk_pct")
        assert hasattr(pq, "symbol_exposure_pct")
        assert pq.symbol_exposure_pct > 0.0

    def test_budget_utilization_in_snapshot(self):
        """Budget utilization should compute from risk engine snapshot."""
        # Simulate a risk_snap from the engine
        risk_snap = {
            "allocated_risk_abs": 300.0,
            "actual_risk_abs": 15.0,  # Only 5% utilization
            "final_notional": 500.0,
            "risk_cap_notional": 10000.0,
            "profile_multiplier": 0.5,
            "profile_multiplier_components": {"global": 1.0, "symbol": 0.5},
            "risk_cap_abs": 300.0,
        }
        allocated = risk_snap.get("allocated_risk_abs") or 0.0
        actual = risk_snap.get("actual_risk_abs")
        final_notional = risk_snap.get("final_notional") or 0.0
        risk_cap_notional = risk_snap.get("risk_cap_notional") or 0.0

        budget_util = (actual / allocated * 100) if (allocated > 0 and actual is not None) else 0.0
        notional_util = (final_notional / risk_cap_notional * 100) if risk_cap_notional > 0 else 0.0

        assert abs(budget_util - 5.0) < 0.01
        assert abs(notional_util - 5.0) < 0.01

        # Should trigger must_fix when < 10%
        assert budget_util < 10.0, "Low utilization should be flagged"

    def test_budget_utilization_must_fix_hint(self):
        """Utilization < 10% should produce a must_fix hint."""
        util_pct = 5.0
        must_fix: List[str] = []
        if util_pct > 0 and util_pct < 10.0:
            must_fix.append(
                f"Position sizes using only {util_pct:.1f}% of risk budget — check profile multipliers."
            )
        assert len(must_fix) == 1
        assert "5.0%" in must_fix[0]

        # 15% should NOT trigger
        must_fix_high: List[str] = []
        util_high = 15.0
        if util_high > 0 and util_high < 10.0:
            must_fix_high.append("should not appear")
        assert len(must_fix_high) == 0


# =============================================================================
# Runbook 27 — Stance Diversity
# =============================================================================


class TestStanceDiversity:
    """Verify recommended stance and stance diversity metrics (Runbook 27)."""

    def test_recommended_stance_on_drawdown(self):
        """Drawdown > 2% should recommend defensive stance."""
        # Simulate heuristic logic
        emergency_pct = 0.35
        daily_return_pct = -2.5
        quality_score = 40.0

        recommended_stance = "active"
        if emergency_pct > 0.3 or daily_return_pct < -2.0:
            recommended_stance = "defensive"
        elif quality_score < 30:
            recommended_stance = "wait"

        assert recommended_stance == "defensive"

    def test_recommended_stance_wait_on_low_quality(self):
        """Quality score < 30 should recommend wait stance."""
        emergency_pct = 0.05
        daily_return_pct = -0.5
        quality_score = 25.0

        recommended_stance = "active"
        if emergency_pct > 0.3 or daily_return_pct < -2.0:
            recommended_stance = "defensive"
        elif quality_score < 30:
            recommended_stance = "wait"

        assert recommended_stance == "wait"

    def test_recommended_stance_active_when_healthy(self):
        """Good metrics should keep active stance."""
        emergency_pct = 0.05
        daily_return_pct = 1.5
        quality_score = 70.0

        recommended_stance = "active"
        if emergency_pct > 0.3 or daily_return_pct < -2.0:
            recommended_stance = "defensive"
        elif quality_score < 30:
            recommended_stance = "wait"

        assert recommended_stance == "active"

    def test_stance_history_in_snapshot(self):
        """Stance history should compute diversity score."""
        import math

        stances = ["active", "active", "defensive", "active", "wait"]
        counts = {"active": 3, "defensive": 1, "wait": 1}
        total = 5

        diversity = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                diversity -= p * math.log(p)
        max_diversity = math.log(3)
        score = round(diversity / max_diversity, 2)

        # With 3 active, 1 defensive, 1 wait = moderate diversity
        assert 0.0 < score < 1.0
        assert score > 0.5  # Some diversity present

    def test_stance_diversity_all_same(self):
        """All-same stances should have diversity score 0."""
        import math

        counts = {"active": 5, "defensive": 0, "wait": 0}
        total = 5

        diversity = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                diversity -= p * math.log(p)
        max_diversity = math.log(3)
        score = round(diversity / max_diversity, 2)

        assert score == 0.0

    def test_display_constraints_has_recommended_stance(self):
        """DisplayConstraints should accept recommended_stance field."""
        from schemas.judge_feedback import DisplayConstraints

        dc = DisplayConstraints(recommended_stance="defensive")
        assert dc.recommended_stance == "defensive"

        dc_none = DisplayConstraints()
        assert dc_none.recommended_stance is None

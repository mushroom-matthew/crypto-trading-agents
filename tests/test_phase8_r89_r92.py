"""Unit tests for Phase 8: R89 best-of-N and R92 challenger debate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from schemas.judge_feedback import DeliberationVerdict
from services.judge_validation_service import JudgePlanValidationService
from services.plan_deliberation_service import DeliberationService, _score_plan, _find_divergence_points


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trigger(symbol: str = "BTC-USD", direction: str = "long") -> "TriggerCondition":
    from schemas.llm_strategist import TriggerCondition
    return TriggerCondition(
        id=str(uuid4()),
        symbol=symbol,
        direction=direction,
        timeframe="5m",
        category="trend_continuation",
        entry_rule="price > sma_20",
        exit_rule="price < sma_20",
        stop_loss_pct=0.02,
    )


def _make_plan(
    regime: str = "bull",
    with_triggers: bool = True,
    playbook_id: str | None = None,
    allowed_directions: list[str] | None = None,
    stance: str | None = "active",
) -> "StrategyPlan":
    from schemas.llm_strategist import StrategyPlan, RiskConstraint, PositionSizingRule
    now = datetime.now(timezone.utc)
    kwargs: dict = dict(
        generated_at=now,
        valid_until=now + timedelta(hours=4),
        global_view="test plan",
        regime=regime,
        triggers=[_make_trigger()] if with_triggers else [],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=2.0,
            max_symbol_exposure_pct=25.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=2.0)
        ],
        allowed_directions=allowed_directions or [],
        playbook_id=playbook_id,
    )
    if stance is not None:
        kwargs["stance"] = stance
    return StrategyPlan(**kwargs)


# ---------------------------------------------------------------------------
# R89 — JudgePlanValidationService.batch_validate
# ---------------------------------------------------------------------------

class TestBatchValidate:
    def test_batch_validate_returns_one_verdict_per_plan(self):
        svc = JudgePlanValidationService()
        plans = [_make_plan("bull"), _make_plan("bear"), _make_plan("range")]
        verdicts = svc.batch_validate(plans)
        assert len(verdicts) == 3

    def test_batch_validate_empty_list(self):
        svc = JudgePlanValidationService()
        assert svc.batch_validate([]) == []

    def test_batch_validate_all_have_confidence_scores(self):
        svc = JudgePlanValidationService()
        plans = [_make_plan("bull"), _make_plan("range")]
        verdicts = svc.batch_validate(plans)
        for v in verdicts:
            assert 0.0 <= v.judge_confidence_score <= 1.0

    def test_batch_validate_empty_trigger_plan_gets_structural_reject(self):
        """Empty trigger list → structural reject (not a crash)."""
        svc = JudgePlanValidationService()
        empty_trigger_plan = _make_plan("bull", with_triggers=False)
        verdicts = svc.batch_validate([empty_trigger_plan])
        assert len(verdicts) == 1
        assert verdicts[0].decision == "reject"


# ---------------------------------------------------------------------------
# R92 — DeliberationVerdict schema
# ---------------------------------------------------------------------------

class TestDeliberationVerdictSchema:
    def test_primary_wins_verdict(self):
        v = DeliberationVerdict(outcome="primary_wins", confidence_margin=0.15, primary_score=0.9, challenger_score=0.75)
        assert v.outcome == "primary_wins"
        assert v.confidence_margin == pytest.approx(0.15)

    def test_challenger_wins_verdict(self):
        v = DeliberationVerdict(outcome="challenger_wins", confidence_margin=-0.1, primary_score=0.6, challenger_score=0.7)
        assert v.outcome == "challenger_wins"

    def test_inconclusive_verdict(self):
        v = DeliberationVerdict(outcome="inconclusive", confidence_margin=0.02)
        assert v.outcome == "inconclusive"
        assert v.divergence_points == []

    def test_divergence_points_accepted(self):
        v = DeliberationVerdict(
            outcome="primary_wins",
            divergence_points=["regime: bull vs bear", "stance: aggressive vs flat"],
        )
        assert len(v.divergence_points) == 2


# ---------------------------------------------------------------------------
# R92 — DeliberationService
# ---------------------------------------------------------------------------

class TestDeliberationService:
    def test_primary_wins_when_challenger_has_more_findings(self):
        svc = DeliberationService()
        primary = _make_plan("bull")   # clean plan
        challenger = _make_plan("bear")  # also clean — scores same
        verdict = svc.deliberate(primary, challenger)
        # Both score roughly equally on a clean plan — should be inconclusive or primary_wins
        assert verdict.outcome in ("primary_wins", "inconclusive")

    def test_returns_deliberation_verdict_type(self):
        svc = DeliberationService()
        primary = _make_plan("bull")
        challenger = _make_plan("bear")
        verdict = svc.deliberate(primary, challenger)
        assert isinstance(verdict, DeliberationVerdict)
        assert 0.0 <= verdict.primary_score <= 1.0
        assert 0.0 <= verdict.challenger_score <= 1.0

    def test_divergence_points_captured(self):
        primary = _make_plan("bull", allowed_directions=["long"])
        challenger = _make_plan("bear", allowed_directions=["short"])
        svc = DeliberationService()
        verdict = svc.deliberate(primary, challenger)
        # regime and allowed_directions differ — should appear in divergence_points
        assert len(verdict.divergence_points) >= 1
        joined = " ".join(verdict.divergence_points)
        assert "bull" in joined or "bear" in joined or "long" in joined or "short" in joined

    def test_score_plan_returns_float_in_range(self):
        plan = _make_plan("bull")
        score = _score_plan(plan)
        assert 0.0 <= score <= 1.0

    def test_score_plan_handles_none_plan_gracefully(self):
        # Should return 0.5 neutral when scorer fails
        score = _score_plan(SimpleNamespace())
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# R92 — validate_plan with deliberation_verdict escalation
# ---------------------------------------------------------------------------

class TestDeliberationEscalation:
    def test_inconclusive_debate_adds_revise_reason(self):
        svc = JudgePlanValidationService()
        plan = _make_plan("bull")
        verdict_no_debate = svc.validate_plan(plan)

        debate = DeliberationVerdict(outcome="inconclusive", confidence_margin=0.02)
        verdict_with_debate = svc.validate_plan(plan, deliberation_verdict=debate)

        # The debate should add a REVISE: reason about the escalation
        debate_reasons = [r for r in verdict_with_debate.reasons if "R92" in r]
        assert len(debate_reasons) >= 1

    def test_primary_wins_debate_adds_no_escalation(self):
        svc = JudgePlanValidationService()
        plan = _make_plan("bull")
        debate = DeliberationVerdict(outcome="primary_wins", confidence_margin=0.25)
        verdict = svc.validate_plan(plan, deliberation_verdict=debate)
        debate_reasons = [r for r in verdict.reasons if "R92" in r]
        assert len(debate_reasons) == 0  # no penalty when primary wins clearly

    def test_challenger_wins_adds_revise_reason(self):
        svc = JudgePlanValidationService()
        plan = _make_plan("bull")
        debate = DeliberationVerdict(outcome="challenger_wins", confidence_margin=-0.15)
        verdict = svc.validate_plan(plan, deliberation_verdict=debate)
        debate_reasons = [r for r in verdict.reasons if "R92" in r]
        assert len(debate_reasons) >= 1
        assert any("challenger_wins" in r for r in debate_reasons)

    def test_validate_plan_without_debate_unchanged(self):
        svc = JudgePlanValidationService()
        plan = _make_plan("bull")
        verdict = svc.validate_plan(plan, deliberation_verdict=None)
        assert not any("R92" in r for r in verdict.reasons)


# ---------------------------------------------------------------------------
# R92 — _find_divergence_points
# ---------------------------------------------------------------------------

class TestFindDivergencePoints:
    def test_same_regime_no_regime_divergence(self):
        primary = _make_plan("bull")
        challenger = _make_plan("bull")
        points = _find_divergence_points(primary, challenger)
        assert not any("regime" in p for p in points)

    def test_different_regime_captured(self):
        primary = _make_plan("bull")
        challenger = _make_plan("bear")
        points = _find_divergence_points(primary, challenger)
        assert any("regime" in p for p in points)

    def test_different_directions_captured(self):
        primary = _make_plan("bull", allowed_directions=["long"])
        challenger = _make_plan("bull", allowed_directions=["short"])
        points = _find_divergence_points(primary, challenger)
        assert any("directions" in p for p in points)

    def test_max_four_points(self):
        primary = _make_plan("bull", allowed_directions=["long"], stance="active")
        challenger = _make_plan("bear", allowed_directions=["short"], stance="defensive")
        points = _find_divergence_points(primary, challenger)
        assert len(points) <= 4

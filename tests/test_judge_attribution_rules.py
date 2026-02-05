"""Tests for judge attribution computation rules.

Verifies:
- Attribution layer selection based on metrics
- Evidence collection from heuristics
- Secondary factor detection
- Confidence level assignment
"""

import pytest

from schemas.judge_feedback import JudgeAttribution, AttributionEvidence
from services.judge_feedback_service import JudgeFeedbackService, HeuristicAnalysis
from trading_core.trade_quality import TradeMetrics


def make_trade_metrics(
    win_rate: float = 0.5,
    profit_factor: float = 1.0,
    total_trades: int = 10,
    emergency_exit_pct: float = 0.0,
    max_consecutive_losses: int = 2,
) -> TradeMetrics:
    """Helper to create TradeMetrics with specified values."""
    winning = int(total_trades * win_rate)
    losing = total_trades - winning
    gross_profit = 100.0 if profit_factor >= 1 else 50.0
    gross_loss = gross_profit / profit_factor if profit_factor > 0 else 100.0
    return TradeMetrics(
        total_trades=total_trades,
        winning_trades=winning,
        losing_trades=losing,
        win_rate=win_rate,
        profit_factor=profit_factor,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        avg_win=10.0,
        avg_loss=5.0,
        quality_score=50.0,
        emergency_exit_pct=emergency_exit_pct,
        emergency_exit_count=int(total_trades * emergency_exit_pct),
        max_consecutive_losses=max_consecutive_losses,
    )


def make_heuristics(base_score: float = 50.0, with_evidence: bool = True) -> HeuristicAnalysis:
    """Helper to create HeuristicAnalysis with at least minimal evidence."""
    heuristics = HeuristicAnalysis(base_score=base_score)
    if with_evidence:
        # Add a minimal score adjustment to ensure evidence is present
        heuristics.score_adjustments.append({
            "reason": f"Base score evaluation",
            "delta": 0.0,
        })
    return heuristics


class TestSafetyAttribution:
    """Test safety layer attribution detection."""

    def test_high_emergency_exit_rate_triggers_safety(self):
        """Emergency exit rate > 30% attributes to safety."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=40.0)
        metrics = make_trade_metrics(emergency_exit_pct=0.35)

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.primary_attribution == "safety"
        assert attribution.confidence == "high"
        assert attribution.recommended_action == "hold"
        assert "emergency_exit_pct" in str(attribution.evidence.metrics)

    def test_moderate_emergency_exit_does_not_trigger_safety(self):
        """Emergency exit rate < 30% does not attribute to safety."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=50.0)
        metrics = make_trade_metrics(emergency_exit_pct=0.20)

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.primary_attribution != "safety"


class TestPolicyAttribution:
    """Test policy layer attribution detection."""

    def test_decent_win_rate_poor_profit_factor(self):
        """Good direction but losing money suggests policy issue."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=45.0)
        # Win rate >= 45% but profit factor < 1.0
        metrics = make_trade_metrics(win_rate=0.50, profit_factor=0.8)

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.primary_attribution == "policy"
        assert attribution.recommended_action == "policy_adjust"

    def test_win_rate_near_threshold_poor_pf(self):
        """Win rate at threshold with poor profit factor suggests policy."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=50.0)
        # Win rate at 45% threshold, poor profit factor
        metrics = make_trade_metrics(
            win_rate=0.45,
            profit_factor=0.7,
        )

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.primary_attribution == "policy"


class TestTriggerAttribution:
    """Test trigger layer attribution detection."""

    def test_low_win_rate_triggers_trigger_attribution(self):
        """Win rate < 35% suggests trigger signal quality issues."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=40.0)
        metrics = make_trade_metrics(win_rate=0.30)

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.primary_attribution == "trigger"
        assert attribution.recommended_action == "replan"
        assert "win_rate" in str(attribution.evidence.metrics)

    def test_high_consecutive_losses_triggers_trigger_attribution(self):
        """4+ consecutive losses suggests trigger timing issues."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=45.0)
        metrics = make_trade_metrics(
            win_rate=0.40,  # Not low enough alone
            max_consecutive_losses=5,
        )

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.primary_attribution == "trigger"

    def test_trigger_red_flags_in_heuristics(self):
        """Red flags mentioning triggers attribute to trigger layer."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=45.0)
        heuristics.red_flags.append("Low win rate (<40%); review trigger accuracy.")
        metrics = make_trade_metrics(win_rate=0.40)

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.primary_attribution == "trigger"


class TestPlanAttribution:
    """Test plan layer attribution detection."""

    def test_very_low_score_triggers_plan_attribution(self):
        """Very poor heuristic score suggests plan-level issues."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=30.0)  # Final score < 35
        metrics = make_trade_metrics(win_rate=0.45, profit_factor=1.1)

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.primary_attribution == "plan"
        assert attribution.recommended_action == "replan"

    def test_regime_red_flags_trigger_plan_attribution(self):
        """Regime-related red flags suggest plan issues."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=45.0)
        # Use a red flag that mentions regime but not "trigger" to avoid trigger attribution
        heuristics.red_flags.append("Possible regime mismatch; strategy direction wrong.")
        metrics = make_trade_metrics(win_rate=0.50, profit_factor=1.0)

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.primary_attribution == "plan"

    def test_consecutive_losses_regime_red_flag(self):
        """Consecutive losses flag suggests regime/plan issues."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=40.0)
        heuristics.red_flags.append("Streak of 5 consecutive losses; possible regime mismatch.")
        # Make sure trigger attribution doesn't catch it first
        metrics = make_trade_metrics(
            win_rate=0.45,  # Decent win rate
            max_consecutive_losses=2,  # Low
        )

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.primary_attribution == "plan"


class TestHoldAction:
    """Test hold action assignment."""

    def test_good_score_results_in_hold(self):
        """High heuristic score results in hold action."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=65.0)  # Final score >= 60
        metrics = make_trade_metrics(win_rate=0.55, profit_factor=1.3)

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.recommended_action == "hold"


class TestEvidenceCollection:
    """Test evidence collection from heuristics."""

    def test_score_adjustments_become_metrics(self):
        """Score adjustments are captured as evidence metrics."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=50.0, with_evidence=False)
        heuristics.score_adjustments.append({
            "reason": "Strong win rate (>=60%)",
            "delta": 5.0,
        })
        heuristics.score_adjustments.append({
            "reason": "Good profit factor (>=1.5)",
            "delta": 5.0,
        })
        metrics = make_trade_metrics()

        attribution = service.compute_attribution(heuristics, metrics)

        assert len(attribution.evidence.metrics) >= 2
        assert any("win rate" in m.lower() for m in attribution.evidence.metrics)

    def test_red_flags_become_notes(self):
        """Red flags are captured in evidence notes."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=40.0, with_evidence=False)
        heuristics.red_flags.append("HIGH emergency exit rate (35%)")
        heuristics.red_flags.append("Low win rate; review triggers")
        metrics = make_trade_metrics(emergency_exit_pct=0.35)

        attribution = service.compute_attribution(heuristics, metrics)

        assert attribution.evidence.notes is not None
        assert "emergency" in attribution.evidence.notes.lower() or "win rate" in attribution.evidence.notes.lower()

    def test_evidence_metrics_limited_to_ten(self):
        """Evidence metrics are limited to prevent bloat."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=50.0, with_evidence=False)
        # Add many adjustments
        for i in range(15):
            heuristics.score_adjustments.append({
                "reason": f"Adjustment {i}",
                "delta": 1.0,
            })
        metrics = make_trade_metrics()

        attribution = service.compute_attribution(heuristics, metrics)

        assert len(attribution.evidence.metrics) <= 10


class TestAttributionPrecedence:
    """Test attribution layer precedence ordering."""

    def test_safety_takes_precedence_over_policy(self):
        """Safety attribution trumps policy when both apply."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=40.0)
        # Both safety and policy conditions met
        metrics = make_trade_metrics(
            emergency_exit_pct=0.35,  # Safety trigger
            win_rate=0.50,  # Policy might apply
            profit_factor=0.8,  # Policy trigger
        )

        attribution = service.compute_attribution(heuristics, metrics)

        # Safety should win
        assert attribution.primary_attribution == "safety"

    def test_policy_takes_precedence_over_trigger(self):
        """Policy attribution checked before trigger when both might apply."""
        service = JudgeFeedbackService()
        heuristics = make_heuristics(base_score=45.0)
        # Policy condition: decent win rate but poor profit factor
        metrics = make_trade_metrics(
            win_rate=0.48,  # Above trigger threshold
            profit_factor=0.7,  # Policy trigger
        )

        attribution = service.compute_attribution(heuristics, metrics)

        # Policy should be detected
        assert attribution.primary_attribution == "policy"

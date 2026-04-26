"""Unit tests for Phase 9: R94 regime drift detection, R95 plan audit score."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from schemas.reasoning_cadence import CadenceConfig, RegimeDriftSignal, get_cadence_config
from services.cadence_governor import CadenceGovernor


# ---------------------------------------------------------------------------
# R94 — RegimeDriftSignal schema
# ---------------------------------------------------------------------------

class TestRegimeDriftSignalSchema:
    def test_valid_drift_signal(self):
        sig = RegimeDriftSignal(
            prior_regime="bull",
            current_regime="bear",
            cosine_distance=0.42,
            detected_at_bar=48,
        )
        assert sig.cosine_distance == pytest.approx(0.42)
        assert sig.prior_regime == "bull"

    def test_minimal_drift_signal(self):
        sig = RegimeDriftSignal(cosine_distance=0.37)
        assert sig.prior_regime is None
        assert sig.current_regime is None
        assert sig.detected_at_bar is None

    def test_confidence_defaults_to_one(self):
        sig = RegimeDriftSignal(cosine_distance=0.5)
        assert sig.confidence == 1.0


# ---------------------------------------------------------------------------
# R94 — CadenceConfig drift fields
# ---------------------------------------------------------------------------

class TestCadenceConfigDriftFields:
    def test_has_drift_threshold(self):
        cfg = CadenceConfig()
        assert isinstance(cfg.regime_drift_threshold, float)
        assert cfg.regime_drift_threshold > 0.0

    def test_has_min_cycles_between_refresh(self):
        cfg = CadenceConfig()
        assert isinstance(cfg.regime_drift_min_cycles_between_refresh, int)
        assert cfg.regime_drift_min_cycles_between_refresh > 0

    def test_default_drift_threshold_is_0_35(self):
        cfg = CadenceConfig()
        assert cfg.regime_drift_threshold == pytest.approx(0.35)

    def test_default_min_cycles_is_12(self):
        cfg = CadenceConfig()
        assert cfg.regime_drift_min_cycles_between_refresh == 12


# ---------------------------------------------------------------------------
# R94 — CadenceGovernor.detect_regime_drift
# ---------------------------------------------------------------------------

_FP_BULL = {"vol_percentile": 0.2, "atr_percentile": 0.15, "volume_percentile": 0.8}
_FP_BEAR = {"vol_percentile": 0.9, "atr_percentile": 0.85, "volume_percentile": 0.1}
_FP_SIMILAR = {"vol_percentile": 0.22, "atr_percentile": 0.16, "volume_percentile": 0.78}


class TestDetectRegimeDrift:
    def test_returns_none_when_no_prior_fingerprint(self):
        result = CadenceGovernor.detect_regime_drift({}, _FP_BULL)
        assert result is None

    def test_returns_none_when_no_current_fingerprint(self):
        result = CadenceGovernor.detect_regime_drift(_FP_BULL, {})
        assert result is None

    def test_no_drift_when_fingerprints_similar(self):
        result = CadenceGovernor.detect_regime_drift(_FP_BULL, _FP_SIMILAR, threshold=0.35)
        assert result is None  # similar fingerprints → distance < 0.35

    def test_drift_detected_when_fingerprints_diverge(self):
        result = CadenceGovernor.detect_regime_drift(_FP_BULL, _FP_BEAR, threshold=0.10)
        assert result is not None
        assert isinstance(result, RegimeDriftSignal)
        assert result.cosine_distance > 0.10

    def test_drift_signal_contains_regime_labels(self):
        result = CadenceGovernor.detect_regime_drift(
            _FP_BULL, _FP_BEAR,
            threshold=0.10,
            prior_regime="bull",
            current_regime="bear",
        )
        assert result is not None
        assert result.prior_regime == "bull"
        assert result.current_regime == "bear"

    def test_drift_signal_contains_cycle_count(self):
        result = CadenceGovernor.detect_regime_drift(
            _FP_BULL, _FP_BEAR, threshold=0.10, cycle_count=42
        )
        assert result is not None
        assert result.detected_at_bar == 42

    def test_threshold_respected(self):
        # Distance between BULL and BEAR should exceed threshold=0.10
        result_low = CadenceGovernor.detect_regime_drift(_FP_BULL, _FP_BEAR, threshold=0.10)
        assert result_low is not None
        # Same pair should not fire at very high threshold
        result_high = CadenceGovernor.detect_regime_drift(_FP_BULL, _FP_BEAR, threshold=0.99)
        assert result_high is None

    def test_cosine_distance_is_bounded(self):
        result = CadenceGovernor.detect_regime_drift(_FP_BULL, _FP_BEAR, threshold=0.01)
        assert result is not None
        assert 0.0 <= result.cosine_distance <= 1.0

    def test_handles_no_shared_keys(self):
        fp_a = {"metric_a": 0.5}
        fp_b = {"metric_b": 0.5}
        result = CadenceGovernor.detect_regime_drift(fp_a, fp_b, threshold=0.10)
        assert result is None  # no shared keys → no comparison possible


# ---------------------------------------------------------------------------
# R95 — PlanConfidenceScore schema
# ---------------------------------------------------------------------------

class TestPlanConfidenceScore:
    def test_default_score(self):
        from schemas.judge_feedback import PlanConfidenceScore
        score = PlanConfidenceScore(aggregate_log_reward=0.0)
        assert score.reject_count == 0
        assert score.revise_count == 0
        assert score.field_uncertainty_mean is None
        assert score.field_logprobs_flagged == []

    def test_score_with_findings(self):
        from schemas.judge_feedback import PlanConfidenceScore
        score = PlanConfidenceScore(
            aggregate_log_reward=-3.22,
            reject_count=1,
            revise_count=2,
            interpretation="Has structural violations.",
            field_logprobs_flagged=["regime"],
        )
        assert score.reject_count == 1
        assert score.field_logprobs_flagged == ["regime"]


# ---------------------------------------------------------------------------
# R95 — PlanHallucinationScorer.aggregate_score
# ---------------------------------------------------------------------------

class TestAggregateScore:
    def _make_report(self, reject: int = 0, revise: int = 0):
        from schemas.judge_feedback import PlanHallucinationReport, SectionHallucinationFinding
        findings = []
        for i in range(reject):
            findings.append(SectionHallucinationFinding(
                section_id=f"section_r{i}",
                hallucination_type="fabrication",
                severity="REJECT",
                detail="test reject",
            ))
        for i in range(revise):
            findings.append(SectionHallucinationFinding(
                section_id=f"section_v{i}",
                hallucination_type="context_inconsistency",
                severity="REVISE",
                detail="test revise",
            ))
        return PlanHallucinationReport(findings=findings)

    def test_clean_plan_scores_zero(self):
        from services.plan_hallucination_scorer import PlanHallucinationScorer
        report = self._make_report()
        score = PlanHallucinationScorer().aggregate_score(report)
        assert score.aggregate_log_reward == pytest.approx(0.0)
        assert score.reject_count == 0
        assert score.revise_count == 0

    def test_reject_finding_penalizes_score(self):
        from services.plan_hallucination_scorer import PlanHallucinationScorer
        report = self._make_report(reject=1)
        score = PlanHallucinationScorer().aggregate_score(report)
        # log(1 - 0.90) = log(0.10) ≈ -2.303
        assert score.aggregate_log_reward < 0
        assert score.aggregate_log_reward == pytest.approx(math.log(0.10), rel=0.01)
        assert score.reject_count == 1

    def test_revise_finding_penalizes_less_than_reject(self):
        from services.plan_hallucination_scorer import PlanHallucinationScorer
        r_report = self._make_report(reject=1)
        v_report = self._make_report(revise=1)
        r_score = PlanHallucinationScorer().aggregate_score(r_report)
        v_score = PlanHallucinationScorer().aggregate_score(v_report)
        assert r_score.aggregate_log_reward < v_score.aggregate_log_reward

    def test_multiple_findings_add_up(self):
        from services.plan_hallucination_scorer import PlanHallucinationScorer
        report = self._make_report(reject=2, revise=1)
        score = PlanHallucinationScorer().aggregate_score(report)
        expected = 2 * math.log(0.10) + math.log(0.40)
        assert score.aggregate_log_reward == pytest.approx(expected, rel=0.01)
        assert score.reject_count == 2
        assert score.revise_count == 1

    def test_field_uncertainty_mean_computed(self):
        from services.plan_hallucination_scorer import PlanHallucinationScorer
        report = self._make_report()
        score = PlanHallucinationScorer().aggregate_score(
            report,
            field_uncertainty={"regime": 0.8, "stance": 0.6},
        )
        assert score.field_uncertainty_mean == pytest.approx(0.7, abs=0.01)

    def test_field_logprobs_flagged_below_threshold(self):
        from services.plan_hallucination_scorer import PlanHallucinationScorer
        report = self._make_report()
        score = PlanHallucinationScorer().aggregate_score(
            report,
            field_logprobs={"regime": -0.5, "stop_loss_pct": -1.8},
        )
        assert "stop_loss_pct" in score.field_logprobs_flagged
        assert "regime" not in score.field_logprobs_flagged

    def test_interpretation_describes_clean_plan(self):
        from services.plan_hallucination_scorer import PlanHallucinationScorer
        score = PlanHallucinationScorer().aggregate_score(self._make_report())
        assert "No hallucination" in score.interpretation

    def test_interpretation_describes_reject(self):
        from services.plan_hallucination_scorer import PlanHallucinationScorer
        score = PlanHallucinationScorer().aggregate_score(self._make_report(reject=1))
        assert "REJECT" in score.interpretation


# ---------------------------------------------------------------------------
# R94 — SessionIntent.drift_triggered field
# ---------------------------------------------------------------------------

def _sym_intent(symbol: str = "BTC-USD") -> "SymbolIntent":
    from schemas.session_intent import SymbolIntent
    return SymbolIntent(
        symbol=symbol,
        direction_bias="long",
        risk_budget_fraction=1.0,
        thesis_summary="test",
        expected_hold_horizon="intraday",
        opportunity_score_norm=0.75,
    )


class TestSessionIntentDriftTriggered:
    def test_drift_triggered_defaults_to_false(self):
        from schemas.session_intent import SessionIntent
        intent = SessionIntent(
            selected_symbols=["BTC-USD"],
            symbol_intents=[_sym_intent()],
        )
        assert intent.drift_triggered is False

    def test_drift_triggered_can_be_set_true(self):
        from schemas.session_intent import SessionIntent
        intent = SessionIntent(
            selected_symbols=["BTC-USD"],
            symbol_intents=[_sym_intent()],
            drift_triggered=True,
        )
        assert intent.drift_triggered is True

"""Tests for WorldState schema and world_state_manager (R80)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from schemas.world_state import (
    ConfidenceCalibration,
    EpisodeDigest,
    RegimeFingerprintPoint,
    RegimeTrajectory,
    StructureDigest,
    WorldState,
)
from services.world_state_manager import (
    apply_judge_guidance,
    get_playbook_bonuses,
    get_playbook_penalties,
    get_risk_multiplier,
    get_symbol_vetoes,
    update_episode_digest,
    update_policy_state,
    update_regime,
    update_structure_digest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_fingerprint(rsi: float = 0.5, vol: float = 0.3) -> dict:
    return {"rsi_norm": rsi, "vol_norm": vol, "trend_norm": 0.6}


# ---------------------------------------------------------------------------
# 1. WorldState default construction
# ---------------------------------------------------------------------------


class TestWorldStateDefaults:
    def test_construct_with_defaults(self):
        ws = WorldState()
        assert ws.world_state_id is not None
        assert ws.regime_fingerprint is None
        assert ws.judge_guidance is None
        assert ws.policy_state is None
        assert isinstance(ws.confidence_calibration, ConfidenceCalibration)
        assert isinstance(ws.regime_trajectory, RegimeTrajectory)
        assert isinstance(ws.structure_digest, StructureDigest)
        assert isinstance(ws.episode_digest, EpisodeDigest)

    def test_to_dict_and_from_dict(self):
        ws = WorldState()
        d = ws.to_dict()
        assert isinstance(d, dict)
        assert "world_state_id" in d
        ws2 = WorldState.from_dict(d)
        assert ws2.world_state_id == ws.world_state_id

    def test_extra_fields_forbidden(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            WorldState(unknown_field="x")

    def test_confidence_calibration_defaults(self):
        cal = ConfidenceCalibration()
        assert cal.regime_assessment_confidence == 1.0
        assert cal.stop_placement_confidence == 1.0
        assert cal.updated_by == "init"

    def test_regime_trajectory_defaults(self):
        traj = RegimeTrajectory()
        assert traj.snapshots == []
        assert traj.velocity_scalar == 0.0
        assert traj.stability_score == 1.0
        assert traj.window_size == 20


# ---------------------------------------------------------------------------
# 2. update_regime — adds to trajectory and recomputes velocity
# ---------------------------------------------------------------------------


class TestUpdateRegime:
    def test_single_update(self):
        ws = WorldState()
        fp = _make_fingerprint(0.6, 0.4)
        ws2 = update_regime(ws, fp, trend_state="uptrend", bar_index=1)

        assert ws2.regime_fingerprint == fp
        assert ws2.regime_fingerprint_meta == {"trend_state": "uptrend"}
        assert len(ws2.regime_trajectory.snapshots) == 1
        # With only one snapshot, velocity is 0 and stability is 1
        assert ws2.regime_trajectory.velocity_scalar == 0.0
        assert ws2.regime_trajectory.stability_score == 1.0

    def test_immutability(self):
        ws = WorldState()
        fp = _make_fingerprint()
        ws2 = update_regime(ws, fp, bar_index=1)
        # Original unchanged
        assert ws.regime_fingerprint is None
        assert len(ws.regime_trajectory.snapshots) == 0

    def test_world_state_id_changes_on_update(self):
        ws = WorldState()
        ws2 = update_regime(ws, _make_fingerprint(), bar_index=1)
        assert ws2.world_state_id != ws.world_state_id

    def test_velocity_increases_after_multiple_updates(self):
        ws = WorldState()
        # Feed progressively changing fingerprints to build trajectory
        for i in range(5):
            fp = {"rsi_norm": 0.1 * i, "vol_norm": 0.05 * i}
            ws = update_regime(ws, fp, bar_index=i)

        # With 5 points all moving in the same direction, velocity > 0
        assert ws.regime_trajectory.velocity_scalar > 0.0

    def test_stability_decreases_with_volatile_regime(self):
        ws = WorldState()
        # Alternating high/low values should produce low stability
        for i in range(6):
            val = 0.9 if i % 2 == 0 else 0.1
            fp = {"rsi_norm": val}
            ws = update_regime(ws, fp, bar_index=i)

        assert ws.regime_trajectory.stability_score < 1.0

    def test_trajectory_window_capped(self):
        ws = WorldState()
        for i in range(25):
            ws = update_regime(ws, {"x": float(i)}, bar_index=i)
        # Window size is 20 — should not exceed it
        assert len(ws.regime_trajectory.snapshots) <= 20

    def test_meta_only_includes_provided_states(self):
        ws = WorldState()
        ws2 = update_regime(ws, _make_fingerprint(), vol_state="high_vol", bar_index=1)
        # Only vol_state provided — trend_state should not appear
        assert "vol_state" in ws2.regime_fingerprint_meta
        assert "trend_state" not in ws2.regime_fingerprint_meta

    def test_existing_meta_preserved_when_no_new_meta(self):
        ws = WorldState()
        ws = update_regime(ws, _make_fingerprint(), trend_state="bear", bar_index=1)
        # Second update with no meta — original meta preserved
        ws2 = update_regime(ws, _make_fingerprint(0.3, 0.7), bar_index=2)
        assert ws2.regime_fingerprint_meta.get("trend_state") == "bear"


# ---------------------------------------------------------------------------
# 3. apply_judge_guidance — updates WorldState and calibration
# ---------------------------------------------------------------------------


class TestApplyJudgeGuidance:
    def test_basic_guidance_application(self):
        ws = WorldState()
        guidance = {
            "risk_multiplier": 0.7,
            "playbook_penalties": {"rsi_extremes": 0.5},
            "symbol_vetoes": ["DOGE-USD"],
            "confidence_adjustments": {
                "regime_assessment": 0.6,
                "stop_placement": 0.8,
            },
        }
        ws2 = apply_judge_guidance(ws, guidance)
        assert ws2.judge_guidance == guidance
        assert ws2.confidence_calibration.regime_assessment_confidence == 0.6
        assert ws2.confidence_calibration.stop_placement_confidence == 0.8
        assert ws2.confidence_calibration.updated_by == "judge_evaluation"

    def test_immutability(self):
        ws = WorldState()
        guidance = {"risk_multiplier": 0.5}
        ws2 = apply_judge_guidance(ws, guidance)
        assert ws.judge_guidance is None  # original unchanged

    def test_world_state_id_changes(self):
        ws = WorldState()
        ws2 = apply_judge_guidance(ws, {"risk_multiplier": 1.0})
        assert ws2.world_state_id != ws.world_state_id

    def test_empty_guidance_does_not_break(self):
        ws = WorldState()
        ws2 = apply_judge_guidance(ws, {})
        assert ws2.judge_guidance == {}
        # Calibration retains defaults
        assert ws2.confidence_calibration.entry_timing_confidence == 1.0

    def test_only_provided_confidence_fields_updated(self):
        ws = WorldState()
        guidance = {
            "confidence_adjustments": {"entry_timing": 0.5},
        }
        ws2 = apply_judge_guidance(ws, guidance)
        assert ws2.confidence_calibration.entry_timing_confidence == 0.5
        # Other fields unchanged
        assert ws2.confidence_calibration.stop_placement_confidence == 1.0


# ---------------------------------------------------------------------------
# 4. get_risk_multiplier — returns 1.0 when no guidance
# ---------------------------------------------------------------------------


class TestGetRiskMultiplier:
    def test_returns_1_when_no_world_state(self):
        assert get_risk_multiplier(None) == 1.0

    def test_returns_1_when_no_guidance(self):
        ws = WorldState()
        assert get_risk_multiplier(ws) == 1.0

    def test_returns_guidance_value(self):
        ws = WorldState()
        ws2 = apply_judge_guidance(ws, {"risk_multiplier": 0.7})
        assert get_risk_multiplier(ws2) == pytest.approx(0.7)

    def test_returns_1_when_guidance_has_no_risk_multiplier(self):
        ws = WorldState()
        ws2 = apply_judge_guidance(ws, {"symbol_vetoes": ["BTC-USD"]})
        assert get_risk_multiplier(ws2) == 1.0

    def test_get_playbook_penalties_empty_when_no_guidance(self):
        ws = WorldState()
        assert get_playbook_penalties(ws) == {}
        assert get_playbook_penalties(None) == {}

    def test_get_playbook_bonuses_empty_when_no_guidance(self):
        assert get_playbook_bonuses(None) == {}

    def test_get_symbol_vetoes_empty_when_no_guidance(self):
        assert get_symbol_vetoes(None) == []

    def test_get_symbol_vetoes_with_guidance(self):
        ws = WorldState()
        ws2 = apply_judge_guidance(ws, {"symbol_vetoes": ["SOL-USD", "DOGE-USD"]})
        vetoes = get_symbol_vetoes(ws2)
        assert "SOL-USD" in vetoes
        assert "DOGE-USD" in vetoes


# ---------------------------------------------------------------------------
# 5. JudgeGuidanceVector construction and serialization
# ---------------------------------------------------------------------------


class TestJudgeGuidanceVector:
    def test_construct_with_defaults(self):
        from schemas.judge_feedback import JudgeGuidanceVector
        gv = JudgeGuidanceVector()
        assert gv.risk_multiplier == 1.0
        assert gv.playbook_penalties == {}
        assert gv.playbook_bonuses == {}
        assert gv.symbol_vetoes == []
        assert gv.confidence_adjustments == {}
        assert gv.direction_bias is None
        assert gv.expires_at_eval == 3

    def test_serialize_to_dict(self):
        from schemas.judge_feedback import JudgeGuidanceVector
        gv = JudgeGuidanceVector(
            risk_multiplier=0.8,
            playbook_penalties={"rsi_extremes": 0.5},
            symbol_vetoes=["ETH-USD"],
            summary="Reduced sizing due to drawdown.",
        )
        d = gv.model_dump()
        assert d["risk_multiplier"] == pytest.approx(0.8)
        assert d["playbook_penalties"] == {"rsi_extremes": 0.5}
        assert "ETH-USD" in d["symbol_vetoes"]

    def test_risk_multiplier_bounds(self):
        from schemas.judge_feedback import JudgeGuidanceVector
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            JudgeGuidanceVector(risk_multiplier=0.0)  # below ge=0.1
        with pytest.raises(ValidationError):
            JudgeGuidanceVector(risk_multiplier=3.0)  # above le=2.0

    def test_expires_at_eval_minimum(self):
        from schemas.judge_feedback import JudgeGuidanceVector
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            JudgeGuidanceVector(expires_at_eval=0)  # below ge=1

    def test_direction_bias_valid_values(self):
        from schemas.judge_feedback import JudgeGuidanceVector
        from pydantic import ValidationError
        gv = JudgeGuidanceVector(direction_bias="long_only")
        assert gv.direction_bias == "long_only"
        with pytest.raises(ValidationError):
            JudgeGuidanceVector(direction_bias="buy_only")  # invalid literal

    def test_extra_fields_forbidden(self):
        from schemas.judge_feedback import JudgeGuidanceVector
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            JudgeGuidanceVector(unknown="x")


# ---------------------------------------------------------------------------
# 6. update_structure_digest and update_episode_digest
# ---------------------------------------------------------------------------


class TestUpdateDigests:
    def test_update_structure_digest(self):
        ws = WorldState()
        ws2 = update_structure_digest(
            ws,
            snapshot_id="snap-001",
            symbol="BTC-USD",
            nearest_support_pct=0.5,
            nearest_resistance_pct=1.2,
            active_level_count=7,
        )
        d = ws2.structure_digest
        assert d.snapshot_id == "snap-001"
        assert d.symbol == "BTC-USD"
        assert d.nearest_support_pct == pytest.approx(0.5)
        assert d.active_level_count == 7
        assert d.computed_at is not None

    def test_update_episode_digest(self):
        ws = WorldState()
        ws2 = update_episode_digest(
            ws,
            bundle_id="bundle-abc",
            win_count=5,
            loss_count=2,
            dominant_failure_mode="early_entry",
            avg_r_achieved=1.4,
        )
        d = ws2.episode_digest
        assert d.bundle_id == "bundle-abc"
        assert d.win_count == 5
        assert d.loss_count == 2
        assert d.dominant_failure_mode == "early_entry"
        assert d.avg_r_achieved == pytest.approx(1.4)

    def test_update_policy_state(self):
        ws = WorldState()
        ws2 = update_policy_state(ws, "THESIS_ARMED")
        assert ws2.policy_state == "THESIS_ARMED"
        assert ws.policy_state is None  # original unchanged


# ---------------------------------------------------------------------------
# 7. JudgeFeedbackService.build_guidance_vector
# ---------------------------------------------------------------------------


class TestBuildGuidanceVector:
    def _make_service(self):
        from services.judge_feedback_service import JudgeFeedbackService, HeuristicAnalysis
        svc = JudgeFeedbackService(transport=lambda _: '{"score": 50}')
        return svc, HeuristicAnalysis

    def test_low_score_reduces_risk_multiplier(self):
        from services.judge_feedback_service import JudgeFeedbackService, HeuristicAnalysis
        svc = JudgeFeedbackService(transport=lambda _: '{"score": 50}')
        h = HeuristicAnalysis(base_score=35.0)  # final_score = 35 (< 40)
        gv = svc.build_guidance_vector(h)
        assert gv.risk_multiplier == pytest.approx(0.6)

    def test_below_50_moderate_reduction(self):
        from services.judge_feedback_service import JudgeFeedbackService, HeuristicAnalysis
        svc = JudgeFeedbackService(transport=lambda _: '{"score": 50}')
        h = HeuristicAnalysis(base_score=45.0)  # final_score = 45 (< 50, >= 40)
        gv = svc.build_guidance_vector(h)
        assert gv.risk_multiplier == pytest.approx(0.8)

    def test_above_70_increases_risk_multiplier(self):
        from services.judge_feedback_service import JudgeFeedbackService, HeuristicAnalysis
        svc = JudgeFeedbackService(transport=lambda _: '{"score": 50}')
        h = HeuristicAnalysis(base_score=75.0)  # final_score = 75 (>= 70)
        gv = svc.build_guidance_vector(h)
        assert gv.risk_multiplier == pytest.approx(1.2)

    def test_neutral_score_keeps_multiplier_at_1(self):
        from services.judge_feedback_service import JudgeFeedbackService, HeuristicAnalysis
        svc = JudgeFeedbackService(transport=lambda _: '{"score": 50}')
        h = HeuristicAnalysis(base_score=60.0)  # final_score = 60 (neutral range)
        gv = svc.build_guidance_vector(h)
        assert gv.risk_multiplier == pytest.approx(1.0)

    def test_guidance_vector_has_guidance_id(self):
        from services.judge_feedback_service import JudgeFeedbackService, HeuristicAnalysis
        svc = JudgeFeedbackService(transport=lambda _: '{"score": 50}')
        h = HeuristicAnalysis(base_score=50.0)
        gv = svc.build_guidance_vector(h)
        assert gv.guidance_id is not None
        assert len(gv.guidance_id) > 0

    def test_generate_feedback_sets_last_guidance_vector(self):
        from services.judge_feedback_service import JudgeFeedbackService
        from schemas.judge_feedback import JudgeGuidanceVector
        import json

        # Use transport shim to avoid LLM call
        def shim(payload):
            return json.dumps({
                "score": 65.0,
                "constraints": {
                    "max_trades_per_day": None,
                    "max_triggers_per_symbol_per_day": None,
                    "risk_mode": "normal",
                    "disabled_trigger_ids": [],
                    "disabled_categories": [],
                },
                "strategist_constraints": {
                    "must_fix": [],
                    "vetoes": [],
                    "boost": [],
                    "regime_correction": None,
                    "sizing_adjustments": {},
                },
            })

        svc = JudgeFeedbackService(transport=shim)
        svc.generate_feedback({"return_pct": 0.5, "trade_count": 5})
        assert svc.last_guidance_vector is not None
        assert isinstance(svc.last_guidance_vector, JudgeGuidanceVector)

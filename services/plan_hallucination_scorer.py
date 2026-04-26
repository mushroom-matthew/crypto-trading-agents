"""R90: Deterministic fine-grained hallucination scorer for StrategyPlan outputs.

Covers the six FG-PRM hallucination types adapted to the trading domain.
All checks are pure Python — no LLM calls, no I/O.  Runs as Layer 0 before
the existing 5-layer judge gate in judge_validation_service.py.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import math

from schemas.judge_feedback import PlanConfidenceScore, PlanHallucinationReport, SectionHallucinationFinding

logger = logging.getLogger(__name__)

# Six FG-PRM hallucination type labels (trading-domain adaptation).
CONTEXT_INCONSISTENCY = "context_inconsistency"
LOGICAL_INCONSISTENCY = "logical_inconsistency"
INSTRUCTION_INCONSISTENCY = "instruction_inconsistency"
CALCULATION_ERROR = "calculation_error"
FACTUAL_INCONSISTENCY = "factual_inconsistency"
FABRICATION = "fabrication"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_regime(r: str) -> str:
    """Map asset-level regime labels onto plan-level labels for comparison."""
    return {"volatile": "high_vol", "uncertain": "mixed"}.get(r, r)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class PlanHallucinationScorer:
    """Score a StrategyPlan against the 6-type hallucination taxonomy."""

    def score(
        self,
        plan,
        llm_input=None,
        judge_constraints=None,
        risk_params: Optional[dict] = None,
    ) -> PlanHallucinationReport:
        """Run all six checks and return a combined report."""
        report = PlanHallucinationReport()
        try:
            self._check_context_inconsistency(plan, llm_input, report)
        except Exception:
            logger.debug("Hallucination check context_inconsistency failed (non-fatal)", exc_info=True)
        try:
            self._check_logical_inconsistency(plan, report)
        except Exception:
            logger.debug("Hallucination check logical_inconsistency failed (non-fatal)", exc_info=True)
        try:
            self._check_instruction_inconsistency(plan, judge_constraints, report)
        except Exception:
            logger.debug("Hallucination check instruction_inconsistency failed (non-fatal)", exc_info=True)
        try:
            self._check_calculation_error(plan, risk_params, report)
        except Exception:
            logger.debug("Hallucination check calculation_error failed (non-fatal)", exc_info=True)
        try:
            self._check_factual_inconsistency(plan, llm_input, report)
        except Exception:
            logger.debug("Hallucination check factual_inconsistency failed (non-fatal)", exc_info=True)
        try:
            self._check_fabrication(plan, report)
        except Exception:
            logger.debug("Hallucination check fabrication failed (non-fatal)", exc_info=True)
        return report

    # ------------------------------------------------------------------
    # Check 1 — Context Inconsistency
    # ------------------------------------------------------------------

    def _check_context_inconsistency(self, plan, llm_input, report: PlanHallucinationReport) -> None:
        """Plan regime contradicts majority-vote from asset-level regime assessments."""
        if llm_input is None:
            return
        assets = getattr(llm_input, "assets", None) or []
        assessments = []
        for asset in assets:
            ra = getattr(asset, "regime_assessment", None)
            if ra is not None:
                r = getattr(ra, "regime", None)
                if r:
                    assessments.append(_normalize_regime(r))
        if len(assessments) < 2:
            return
        from collections import Counter
        counts = Counter(assessments)
        majority_regime, majority_count = counts.most_common(1)[0]
        if majority_count < len(assessments) * 0.6:
            return  # No clear majority — inconclusive
        plan_regime = getattr(plan, "regime", None)
        if plan_regime and plan_regime != majority_regime:
            report.findings.append(SectionHallucinationFinding(
                section_id="regime",
                hallucination_type=CONTEXT_INCONSISTENCY,
                severity="REVISE",
                detail=(
                    f"plan.regime='{plan_regime}' contradicts asset majority "
                    f"regime='{majority_regime}' ({majority_count}/{len(assessments)} assets)"
                ),
            ))

    # ------------------------------------------------------------------
    # Check 2 — Logical Inconsistency
    # ------------------------------------------------------------------

    def _check_logical_inconsistency(self, plan, report: PlanHallucinationReport) -> None:
        """Opposite-direction entry triggers on the same symbol without mutual exclusion."""
        triggers = getattr(plan, "triggers", None) or []
        from collections import defaultdict
        by_symbol: dict[str, list] = defaultdict(list)
        entry_cats = {"trend_continuation", "reversal", "volatility_breakout", "mean_reversion"}
        for t in triggers:
            cat = getattr(t, "category", None) or ""
            direction = getattr(t, "direction", None) or ""
            symbol = getattr(t, "symbol", None) or ""
            if direction in ("long", "short") and cat in entry_cats:
                by_symbol[symbol].append(t)

        for symbol, sym_triggers in by_symbol.items():
            directions = {getattr(t, "direction", "") for t in sym_triggers}
            if "long" not in directions or "short" not in directions:
                continue
            # Check if entry rules are mutually exclusive via regime/condition guard
            # Simple heuristic: if any trigger lacks an explicit condition guard that
            # the other direction's trigger would also pass, flag it.
            long_rules = [getattr(t, "entry_rule", "") or "" for t in sym_triggers if getattr(t, "direction", "") == "long"]
            short_rules = [getattr(t, "entry_rule", "") or "" for t in sym_triggers if getattr(t, "direction", "") == "short"]
            # If any long rule is a subset string of a short rule or vice versa, they can co-exist.
            # Simplified check: if neither set has an exclusive condition keyword, flag.
            exclusive_keywords = {"bull", "bear", "breakout_confirmed", "above_ema", "below_ema",
                                   "htf_daily_trend", "expansion_flag", "compression_flag"}
            long_has_guard = any(
                any(kw in rule.lower() for kw in exclusive_keywords) for rule in long_rules
            )
            short_has_guard = any(
                any(kw in rule.lower() for kw in exclusive_keywords) for rule in short_rules
            )
            if not long_has_guard and not short_has_guard:
                report.findings.append(SectionHallucinationFinding(
                    section_id=f"triggers.{symbol}",
                    hallucination_type=LOGICAL_INCONSISTENCY,
                    severity="REVISE",
                    detail=(
                        f"Symbol '{symbol}' has both long and short entry triggers "
                        f"without clear mutually exclusive conditions"
                    ),
                ))

    # ------------------------------------------------------------------
    # Check 3 — Instruction Inconsistency
    # ------------------------------------------------------------------

    def _check_instruction_inconsistency(self, plan, judge_constraints, report: PlanHallucinationReport) -> None:
        """Trigger category violates disabled_categories from judge constraints."""
        if judge_constraints is None:
            return
        disabled = set(getattr(judge_constraints, "disabled_categories", None) or [])
        if not disabled:
            return
        triggers = getattr(plan, "triggers", None) or []
        violating = [
            getattr(t, "id", str(i))
            for i, t in enumerate(triggers)
            if (getattr(t, "category", None) or "") in disabled
        ]
        if violating:
            report.findings.append(SectionHallucinationFinding(
                section_id="triggers.category",
                hallucination_type=INSTRUCTION_INCONSISTENCY,
                severity="REJECT",
                detail=(
                    f"Triggers {violating} use disabled categories: "
                    f"{disabled & {getattr(t, 'category', '') for t in triggers}}"
                ),
            ))

    # ------------------------------------------------------------------
    # Check 4 — Calculation Error
    # ------------------------------------------------------------------

    def _check_calculation_error(self, plan, risk_params: Optional[dict], report: PlanHallucinationReport) -> None:
        """stop_loss_pct outside allowed band or R:R below floor."""
        max_pos_risk = (risk_params or {}).get("max_position_risk_pct", 10.0)
        min_rr = (risk_params or {}).get("min_rr_ratio", 1.0)
        triggers = getattr(plan, "triggers", None) or []
        for t in triggers:
            direction = getattr(t, "direction", "") or ""
            if direction not in ("long", "short"):
                continue
            tid = getattr(t, "id", "?")
            stop_pct = getattr(t, "stop_loss_pct", None)
            if stop_pct is not None and stop_pct > max_pos_risk:
                report.findings.append(SectionHallucinationFinding(
                    section_id=f"triggers.{tid}.stop_loss_pct",
                    hallucination_type=CALCULATION_ERROR,
                    severity="REVISE",
                    detail=(
                        f"Trigger '{tid}' stop_loss_pct={stop_pct:.2f}% "
                        f"exceeds max_position_risk_pct={max_pos_risk:.2f}%"
                    ),
                ))
            r_multiple = getattr(t, "r_multiple", None)
            if r_multiple is not None and r_multiple < min_rr:
                report.findings.append(SectionHallucinationFinding(
                    section_id=f"triggers.{tid}.r_multiple",
                    hallucination_type=CALCULATION_ERROR,
                    severity="REVISE",
                    detail=(
                        f"Trigger '{tid}' r_multiple={r_multiple:.2f} "
                        f"below min_rr_ratio={min_rr:.2f}"
                    ),
                ))

    # ------------------------------------------------------------------
    # Check 5 — Factual Inconsistency
    # ------------------------------------------------------------------

    def _check_factual_inconsistency(self, plan, llm_input, report: PlanHallucinationReport) -> None:
        """plan.allowed_symbols contains a symbol not present in LLMInput.assets."""
        if llm_input is None:
            return
        input_symbols = {getattr(a, "symbol", None) for a in (getattr(llm_input, "assets", None) or [])}
        input_symbols.discard(None)
        allowed = set(getattr(plan, "allowed_symbols", None) or [])
        phantom = allowed - input_symbols
        if phantom:
            report.findings.append(SectionHallucinationFinding(
                section_id="allowed_symbols",
                hallucination_type=FACTUAL_INCONSISTENCY,
                severity="REVISE",
                detail=(
                    f"allowed_symbols contains symbols not in LLMInput.assets: {sorted(phantom)}"
                ),
            ))

    # ------------------------------------------------------------------
    # Check 6 — Fabrication
    # ------------------------------------------------------------------

    def _check_fabrication(self, plan, report: PlanHallucinationReport) -> None:
        """plan.playbook_id not in PlaybookRegistry."""
        playbook_id = getattr(plan, "playbook_id", None)
        if not playbook_id:
            return
        try:
            from services.playbook_registry import PlaybookRegistry
            registry = PlaybookRegistry()
            known_ids = {pb.playbook_id for pb in registry.list_eligible("unknown") or []}
            # list_eligible filters by regime; get all by loading the full registry
            all_ids: set = set()
            if hasattr(registry, "_playbooks"):
                all_ids = {pb.playbook_id for pb in registry._playbooks.values()}
            elif hasattr(registry, "playbooks"):
                all_ids = {pb.playbook_id for pb in registry.playbooks.values()}
            else:
                # Fallback: run list_eligible for all known regimes
                for regime in ("bull", "bear", "range", "high_vol", "mixed", "unknown"):
                    try:
                        all_ids.update(pb.playbook_id for pb in (registry.list_eligible(regime) or []))
                    except Exception:
                        pass
            if all_ids and playbook_id not in all_ids:
                report.findings.append(SectionHallucinationFinding(
                    section_id="playbook_id",
                    hallucination_type=FABRICATION,
                    severity="REJECT",
                    detail=f"plan.playbook_id='{playbook_id}' not found in PlaybookRegistry",
                ))
        except Exception:
            logger.debug("Fabrication check skipped (PlaybookRegistry unavailable)", exc_info=True)

    # ------------------------------------------------------------------
    # R93 — Uncertainty cross-check
    # ------------------------------------------------------------------

    def uncertainty_pass(
        self,
        plan,
        memory_bundle=None,
    ) -> "list[SectionHallucinationFinding]":
        """Cross-reference LLM-stated confidence against memory cluster win-rate.

        High stated confidence + memory showing mostly losses for this regime →
        FieldUncertaintyFinding (REVISE severity).  All failures are non-fatal.
        """
        findings: list[SectionHallucinationFinding] = []
        try:
            field_uncertainty = getattr(plan, "field_uncertainty", None) or {}
            if not field_uncertainty:
                return findings

            # Compute cluster support from memory bundle (loss rate for current regime)
            cluster_support: float | None = None
            if memory_bundle is not None:
                wins = len(getattr(memory_bundle, "winning_contexts", None) or [])
                losses = len(getattr(memory_bundle, "losing_contexts", None) or [])
                total = wins + losses
                if total >= 3:
                    cluster_support = wins / total  # fraction that are wins

            for field, lm_confidence in field_uncertainty.items():
                if not isinstance(lm_confidence, (int, float)):
                    continue
                lm_confidence = float(lm_confidence)
                if lm_confidence <= 0.7:
                    continue  # only flag when LLM is highly confident
                if cluster_support is None:
                    continue  # no memory data — can't cross-check
                if cluster_support >= 0.4:
                    continue  # memory supports the trade — no conflict
                # High LLM confidence + low memory win-rate = suspicious
                findings.append(SectionHallucinationFinding(
                    section_id=field,
                    hallucination_type=FACTUAL_INCONSISTENCY,
                    severity="REVISE",
                    detail=(
                        f"R93: field='{field}' lm_confidence={lm_confidence:.2f} "
                        f"cluster_support={cluster_support:.2f} (memory win-rate below 40%)"
                    ),
                ))
        except Exception:
            logger.debug("R93 uncertainty_pass failed (non-fatal)", exc_info=True)
        return findings

    # ------------------------------------------------------------------
    # R97 — Logprob-based factual inconsistency
    # ------------------------------------------------------------------

    def logprob_pass(
        self,
        field_logprobs: "dict[str, float]",
    ) -> "list[SectionHallucinationFinding]":
        """Flag high-entropy fields: mean logprob < -1.5 → Factual Inconsistency.

        Threshold: per-field mean logprob < -1.5 (≈ 22% avg probability per token).
        Severity is REVISE — informs risk scaling but does not hard-reject.
        """
        findings: list[SectionHallucinationFinding] = []
        try:
            for field, mean_lp in (field_logprobs or {}).items():
                if not isinstance(mean_lp, (int, float)):
                    continue
                if float(mean_lp) < -1.5:
                    findings.append(SectionHallucinationFinding(
                        section_id=field,
                        hallucination_type=FACTUAL_INCONSISTENCY,
                        severity="REVISE",
                        detail=(
                            f"R97: field='{field}' mean_logprob={mean_lp:.3f} "
                            f"(<-1.5 threshold — model was uncertain about this value)"
                        ),
                    ))
        except Exception:
            logger.debug("R97 logprob_pass failed (non-fatal)", exc_info=True)
        return findings

    # ------------------------------------------------------------------
    # R95 — Aggregate confidence score (FG-PRM log-sum)
    # ------------------------------------------------------------------

    def aggregate_score(
        self,
        report: PlanHallucinationReport,
        field_uncertainty: "dict[str, float] | None" = None,
        field_logprobs: "dict[str, float] | None" = None,
    ) -> PlanConfidenceScore:
        """Aggregate all signals into a single PlanConfidenceScore.

        Score = Σ log(1 - p_hallucination_i) over all findings.
        Each REJECT finding contributes log(1 - 0.90) = -2.30.
        Each REVISE finding contributes log(1 - 0.60) = -0.92.
        A plan with no findings scores 0.0 (maximum).
        """
        try:
            # Per-section penalties
            _REJECT_P = 0.90  # probability of being a true hallucination
            _REVISE_P = 0.60
            total_log_reward = 0.0
            per_section: list[dict] = []
            reject_count = sum(1 for f in report.findings if f.severity == "REJECT")
            revise_count = sum(1 for f in report.findings if f.severity == "REVISE")

            for f in report.findings:
                p = _REJECT_P if f.severity == "REJECT" else _REVISE_P
                contribution = math.log(1.0 - p)
                total_log_reward += contribution
                per_section.append({
                    "section": f.section_id,
                    "type": f.hallucination_type,
                    "severity": f.severity,
                    "log_contribution": round(contribution, 4),
                })

            # Field uncertainty mean (R93)
            uncertainty_mean: float | None = None
            if field_uncertainty:
                vals = [float(v) for v in field_uncertainty.values() if isinstance(v, (int, float))]
                if vals:
                    uncertainty_mean = round(sum(vals) / len(vals), 4)

            # Logprob-flagged fields (R97)
            flagged_logprob_fields = [
                k for k, v in (field_logprobs or {}).items()
                if isinstance(v, (int, float)) and float(v) < -1.5
            ]

            # Human-readable interpretation
            if total_log_reward == 0.0 and not flagged_logprob_fields:
                interpretation = "No hallucination signals detected. Plan appears internally consistent."
            elif reject_count > 0:
                interpretation = f"Plan has {reject_count} REJECT finding(s) — likely structurally invalid."
            elif revise_count > 0:
                interpretation = f"Plan has {revise_count} REVISE finding(s) — revision recommended."
            else:
                interpretation = "Logprob uncertainty flagged but no structural violations."

            return PlanConfidenceScore(
                aggregate_log_reward=round(total_log_reward, 4),
                per_section_scores=per_section,
                interpretation=interpretation,
                reject_count=reject_count,
                revise_count=revise_count,
                field_uncertainty_mean=uncertainty_mean,
                field_logprobs_flagged=flagged_logprob_fields,
            )
        except Exception as exc:
            logger.debug("aggregate_score failed (non-fatal): %s", exc)
            return PlanConfidenceScore(
                aggregate_log_reward=0.0,
                interpretation="Score computation failed (non-fatal).",
            )

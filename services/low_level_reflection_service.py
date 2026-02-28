"""Policy-level (fast) reflection service (Runbook 50).

Runs immediately after strategist proposal generation and before policy
freeze / judge validation.  Performs deterministic checks first; optional
LLM consolidation is bounded and skippable.

Invocation contract
-------------------
- ONLY called during policy loop events (regime change, position opened/closed,
  heartbeat).  NOT called during THESIS_ARMED activation ticks or HOLD_LOCK
  deterministic management ticks — unless an explicit invalidation/safety
  override reopens a policy boundary.
- All checks must complete within the configured latency budget.
- Deterministic checks run first; any block result short-circuits the rest.
- Optional LLM consolidation cannot introduce new triggers, new feature
  requirements, or new structural hypotheses/playbooks.  It may only
  summarize, validate, or request revision of the strategist proposal.

Environment variables
---------------------
POLICY_REFLECTION_ENABLED   — "true" to enable (default: false)
POLICY_REFLECTION_LLM       — "true" to enable optional LLM step (default: false)
POLICY_REFLECTION_MAX_MS    — latency budget in ms (default: 500)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

from schemas.reflection import (
    PolicyLevelReflectionRequest,
    PolicyLevelReflectionResult,
    ReflectionInvocationMeta,
)

logger = logging.getLogger(__name__)

_ENABLED_DEFAULT = False
_LLM_ENABLED_DEFAULT = False
_MAX_LATENCY_MS_DEFAULT = 500

_FORBIDDEN_SLOW_LOOP_TASKS = [
    "re-cluster regimes",
    "update playbook priors",
    "rewrite playbooks",
    "tune memory distance",
    "multi-episode statistical reanalysis",
]


def is_enabled() -> bool:
    """Check whether policy-level reflection is enabled via env var."""
    return os.environ.get("POLICY_REFLECTION_ENABLED", "").lower() in ("true", "1", "yes")


def _max_latency_ms() -> int:
    try:
        return int(os.environ.get("POLICY_REFLECTION_MAX_MS", str(_MAX_LATENCY_MS_DEFAULT)))
    except (ValueError, TypeError):
        return _MAX_LATENCY_MS_DEFAULT


def _llm_enabled() -> bool:
    return os.environ.get("POLICY_REFLECTION_LLM", "").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Deterministic invariant checks
# ---------------------------------------------------------------------------


def _check_invariants(
    req: PolicyLevelReflectionRequest,
) -> tuple[List[str], bool]:
    """Return (findings, hard_block)."""
    findings: List[str] = []
    hard_block = False

    if req.kill_switch_active:
        findings.append("BLOCK: kill switch is active — no new proposals permitted")
        hard_block = True

    if req.is_activation_window_tick and not req.is_hold_lock_tick:
        findings.append(
            "BLOCK: policy-level reflection must not run during THESIS_ARMED activation "
            "window ticks without explicit safety override"
        )
        hard_block = True

    if req.is_hold_lock_tick:
        findings.append(
            "BLOCK: policy-level reflection must not run during HOLD_LOCK ticks without "
            "explicit safety override"
        )
        hard_block = True

    if not req.risk_constraints_present:
        findings.append("WARN: proposal lacks risk_constraints — expected for any live proposal")

    return findings, hard_block


# ---------------------------------------------------------------------------
# Coherence checks
# ---------------------------------------------------------------------------


def _check_coherence(
    req: PolicyLevelReflectionRequest,
) -> List[str]:
    """Return list of coherence findings (non-blocking by default)."""
    findings: List[str] = []

    if req.trigger_count == 0:
        findings.append("REVISE: no triggers in proposal — strategist produced an empty plan")

    # Direction consistency: if a single direction summary is given, all
    # allowed_directions should be compatible
    if req.direction_summary and req.allowed_directions:
        if req.direction_summary in ("long", "short"):
            if req.direction_summary not in req.allowed_directions:
                findings.append(
                    f"REVISE: trigger direction_summary='{req.direction_summary}' not in "
                    f"allowed_directions={req.allowed_directions}"
                )

    # Playbook declared but no template
    if req.playbook_id and not req.template_id:
        findings.append(
            f"WARN: playbook_id='{req.playbook_id}' is set but template_id is absent — "
            "template binding may be incomplete (R47)"
        )

    # Regime / rationale basic checks
    if not req.regime:
        findings.append("WARN: regime is absent from proposal — expected for regime-tagged plans")

    return findings


# ---------------------------------------------------------------------------
# Contrastive memory checks
# ---------------------------------------------------------------------------


def _check_memory(
    req: PolicyLevelReflectionRequest,
) -> List[str]:
    """Return list of memory-based findings (non-blocking)."""
    findings: List[str] = []

    if not req.memory_bundle_id:
        # No memory available — note it but don't block
        return []

    # Flag specific failure modes that directly contradict common conditions
    direction = req.direction_summary or ""
    for mode in req.memory_failure_modes:
        if mode == "false_breakout_reversion" and direction == "long":
            findings.append(
                f"MEMORY: failure mode '{mode}' is present in retrieved episodes — "
                "long breakout proposals in this regime have historically reversed"
            )
        elif mode == "late_entry_poor_r_multiple":
            findings.append(
                f"MEMORY: failure mode '{mode}' is present — check entry conditions "
                "against playbook activation refinement rules"
            )
        elif mode == "stop_too_tight_noise_out":
            findings.append(
                f"MEMORY: failure mode '{mode}' is present — verify stop placement "
                "uses structural anchors (R42/R56)"
            )
        elif mode == "macro_news_whipsaw":
            findings.append(
                f"MEMORY: failure mode '{mode}' is present — high news sensitivity; "
                "consider reducing conviction or deferring entry"
            )
        else:
            findings.append(
                f"MEMORY: failure mode '{mode}' is present in retrieved losing episodes"
            )

    if req.memory_losing_count > req.memory_winning_count and req.memory_losing_count >= 3:
        findings.append(
            f"MEMORY: retrieved episodes skew losing ({req.memory_losing_count} losses "
            f"vs {req.memory_winning_count} wins) — proposal may be swimming against "
            "recent regime pattern"
        )

    return findings


# ---------------------------------------------------------------------------
# Expectation calibration checks
# ---------------------------------------------------------------------------


def _check_expectations(
    req: PolicyLevelReflectionRequest,
) -> List[str]:
    """Return list of expectation calibration findings (non-blocking)."""
    findings: List[str] = []

    if req.stated_conviction == "high" and req.memory_losing_count >= 3:
        findings.append(
            "EXPECTATION: stated conviction is 'high' but memory bundle shows "
            f"{req.memory_losing_count} losing episodes — conviction may be overfit"
        )

    if (
        req.playbook_expected_hold_bars_p50 is not None
        and req.playbook_expected_hold_bars_p50 < 2
        and req.trigger_count > 5
    ):
        findings.append(
            f"EXPECTATION: playbook P50 hold time is {req.playbook_expected_hold_bars_p50:.1f} bars "
            f"but {req.trigger_count} triggers present — may generate excessive churn"
        )

    return findings


# ---------------------------------------------------------------------------
# Derive overall status from findings
# ---------------------------------------------------------------------------


def _derive_status(
    hard_block: bool,
    invariant_findings: List[str],
    coherence_findings: List[str],
) -> Literal["pass", "revise", "block"]:  # type: ignore[valid-type]
    if hard_block or any(f.startswith("BLOCK:") for f in invariant_findings):
        return "block"
    revise_triggers = [
        f
        for f in invariant_findings + coherence_findings
        if f.startswith("REVISE:")
    ]
    if revise_triggers:
        return "revise"
    return "pass"


# ---------------------------------------------------------------------------
# Public service entry point
# ---------------------------------------------------------------------------


class PolicyLevelReflectionService:
    """Fast policy-boundary reflection service.

    Usage::

        svc = PolicyLevelReflectionService()
        result = svc.reflect(request)
    """

    def reflect(
        self,
        request: PolicyLevelReflectionRequest,
    ) -> PolicyLevelReflectionResult:
        """Run all reflection checks and return a typed result.

        Deterministic invariants run first.  Any BLOCK result short-circuits
        further checks.  Optional LLM consolidation runs last if enabled and
        within latency budget.
        """
        t0 = time.perf_counter()

        # 1. Deterministic invariant checks (cheap — run first, block early)
        invariant_findings, hard_block = _check_invariants(request)

        if hard_block:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return PolicyLevelReflectionResult(
                status="block",
                invariant_findings=invariant_findings,
                latency_ms=elapsed_ms,
            )

        # 2. Structural / coherence checks
        coherence_findings = _check_coherence(request)

        # 3. Contrastive memory scan (bounded — uses pre-fetched summary)
        memory_findings = _check_memory(request)

        # 4. Expectation calibration
        expectation_findings = _check_expectations(request)

        # Derive final status
        status = _derive_status(hard_block, invariant_findings, coherence_findings)

        # Build requested_revisions list
        requested_revisions = [
            f[len("REVISE: "):]
            for f in invariant_findings + coherence_findings
            if f.startswith("REVISE:")
        ]

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # 4b. Optional LLM consolidation (timeout-bounded, only if still within budget)
        max_ms = _max_latency_ms()
        if _llm_enabled() and elapsed_ms < max_ms and status != "block":
            try:
                llm_findings = self._llm_consolidate(
                    request,
                    budget_ms=max_ms - elapsed_ms,
                )
                # LLM may only add findings — it may not introduce new triggers/features
                coherence_findings.extend(llm_findings)
            except Exception:
                logger.debug(
                    "Optional LLM consolidation failed in policy-level reflection",
                    exc_info=True,
                )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(
            "PolicyLevelReflection plan=%s status=%s latency_ms=%d",
            request.plan_id,
            status,
            elapsed_ms,
        )

        return PolicyLevelReflectionResult(
            status=status,
            coherence_findings=coherence_findings,
            invariant_findings=invariant_findings,
            memory_findings=memory_findings,
            expectation_findings=expectation_findings,
            requested_revisions=requested_revisions,
            latency_ms=elapsed_ms,
        )

    def _llm_consolidate(
        self,
        request: PolicyLevelReflectionRequest,
        budget_ms: int,
    ) -> List[str]:
        """Optional LLM-backed consolidation step.

        Role-boundary constraint:
        - May summarize, validate, or request revision of the strategist proposal.
        - May NOT introduce new triggers, feature requirements, or structural
          hypotheses/playbooks.
        - If budget_ms is insufficient, returns empty list.

        This is a stub — wire to the real LLM client when enabling.
        """
        if budget_ms < 50:
            return []
        # TODO(R54): wire to OpenAI with the low_level_reflection.txt prompt template
        #            when POLICY_REFLECTION_LLM=true is set in production.
        return []


# ---------------------------------------------------------------------------
# Convenience: build a request from plan_provider context
# ---------------------------------------------------------------------------


def build_reflection_request(
    *,
    plan_id: Optional[str],
    playbook_id: Optional[str],
    template_id: Optional[str],
    trigger_count: int,
    allowed_directions: List[str],
    regime: Optional[str],
    rationale_excerpt: Optional[str],
    risk_constraints_present: bool,
    policy_state: Optional[str] = None,
    is_activation_window_tick: bool = False,
    is_hold_lock_tick: bool = False,
    snapshot_id: Optional[str] = None,
    snapshot_hash: Optional[str] = None,
    memory_failure_modes: Optional[List[str]] = None,
    memory_winning_count: int = 0,
    memory_losing_count: int = 0,
    memory_bundle_id: Optional[str] = None,
    disabled_trigger_ids: Optional[List[str]] = None,
    disabled_categories: Optional[List[str]] = None,
    kill_switch_active: bool = False,
    playbook_expected_hold_bars_p50: Optional[float] = None,
    playbook_mae_budget_pct: Optional[float] = None,
    stated_conviction: Optional[str] = None,
    source: str = "plan_provider",
    policy_event_type: Optional[str] = None,
) -> PolicyLevelReflectionRequest:
    return PolicyLevelReflectionRequest(
        snapshot_id=snapshot_id,
        snapshot_hash=snapshot_hash,
        plan_id=plan_id,
        playbook_id=playbook_id,
        template_id=template_id,
        direction_summary=allowed_directions[0] if len(allowed_directions) == 1 else (
            "long" if allowed_directions == ["long"] else
            "short" if allowed_directions == ["short"] else
            "mixed"
        ),
        trigger_count=trigger_count,
        allowed_directions=allowed_directions,
        regime=regime,
        rationale_excerpt=rationale_excerpt,
        policy_state=policy_state,
        is_activation_window_tick=is_activation_window_tick,
        is_hold_lock_tick=is_hold_lock_tick,
        memory_failure_modes=memory_failure_modes or [],
        memory_winning_count=memory_winning_count,
        memory_losing_count=memory_losing_count,
        memory_bundle_id=memory_bundle_id,
        risk_constraints_present=risk_constraints_present,
        disabled_trigger_ids=disabled_trigger_ids or [],
        disabled_categories=disabled_categories or [],
        kill_switch_active=kill_switch_active,
        playbook_expected_hold_bars_p50=playbook_expected_hold_bars_p50,
        playbook_mae_budget_pct=playbook_mae_budget_pct,
        stated_conviction=stated_conviction,
        meta=ReflectionInvocationMeta(
            invoked_at=datetime.now(tz=timezone.utc),
            source=source,
            policy_event_type=policy_event_type,
            reflection_kind="policy",
        ),
    )

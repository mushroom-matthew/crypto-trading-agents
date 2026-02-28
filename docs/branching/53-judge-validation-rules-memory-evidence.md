# Runbook 53: Judge Validation Rules (Memory-Backed)

## Purpose

Upgrade the judge loop from primarily risk/logic validation to evidence-based plan
validation using diversified memory retrieval and historical cluster evidence.

The judge should not merely ask "Is this plan syntactically valid and within risk caps?"
It should also ask:

- "Does memory retrieval show known failure patterns matching this proposal?"
- "Is the claimed conviction supported by comparable historical clusters?"
- "Is the strategist overconfident relative to available evidence?"

Cadence boundary rule:

- judge validation in this runbook is a **policy-boundary** function
- tick/bar execution remains deterministic in the trigger engine

## Dependencies / Context

- Builds on **Runbook 20** (attribution rubric)
- Builds on **Runbooks 28–31** (judge action contract + enforcement)
- Uses **Runbook 51** memory retrieval bundles
- Uses **Runbook 52** playbook expectation metadata

## Scope

1. **`schemas/judge_feedback.py`** — extend validation verdicts / evidence fields
2. **`services/judge_feedback_service.py`** — memory-backed plan review and rejection rules
3. **`agents/judge_agent_client.py`** — route revise/reject outcomes back to strategist
4. **`agents/strategies/plan_provider.py`** (or orchestration layer) — revision loop handling
5. **`schemas/llm_strategist.py`** — ensure plan carries evidence references needed by judge
6. **Telemetry/events** — persist judge evidence refs, failure pattern matches, confidence mismatches
7. **Tests** for rejection criteria, revise loops, and unsupported assumption detection

## Out of Scope

- Replacing deterministic risk checks (they remain mandatory)
- Autonomous judge editing of playbooks or memory labels
- Mid-trade micromanagement beyond existing judge action contracts
- Execution-layer slippage attribution changes (covered by existing signal/execution telemetry)

## Judge Validation Model (Post-Upgrade)

Judge validation should be a layered gate:

1. **Deterministic constraints** (risk/invariants/policy/kill switches)
2. **Playbook consistency** (selected playbook, regime eligibility, invalidation completeness)
   - includes policy stability constraints / mutation cooldown checks
3. **Memory failure-pattern scan** (contrastive evidence from Runbook 51)
4. **Historical cluster support** (similar contexts: win/loss balance, expectancy, hold profile)
5. **Confidence calibration** (proposal confidence vs evidence strength)

The judge can return:

- `approve`
- `revise`
- `reject`
- `stand_down` (existing action path when risk/safety conditions demand no trade)

## Validation Finding Classes (Must Remain Distinct)

Judge findings must distinguish these classes explicitly:

1. **Structural violation** (hard reject)
   - deterministic invariant or playbook contract violation
2. **Statistical suspicion** (soft revise)
   - evidence weak/mixed, confidence too high, expectation mismatch
3. **Memory contradiction** (explain-or-revise)
   - nearest losers / failure patterns materially contradict the plan unless the strategist
     provides a concrete divergence explanation

This prevents flattening everything into binary approve/reject and improves auditability.

## Policy Boundary Enforcement (Required)

Judge must enforce policy mutation stability:

- no playbook switching inside the active policy cooldown window unless:
  - structural invalidation is triggered, or
  - safety override path is active
- memory contradiction overrides are evaluated only at policy boundaries (not per tick)
- judge does not micromanage tick-level stop/target actions handled by the deterministic engine
- judge does not reopen thesis evaluation during `THESIS_ARMED` activation-window ticks unless
  invalidation or safety override creates a new policy boundary
- judge does not permit target re-optimization during `HOLD_LOCK` unless explicitly allowed
  by playbook policy-stability rules and triggered via a policy boundary event

## Explicit Rejection / Revision Criteria (Required)

Document these criteria in code and prompt contracts. Example baseline set:

### `reject` (hard stop)

- Plan violates deterministic invariants or risk policy
- Selected playbook is ineligible for current regime
- Memory retrieval shows matching failure pattern with strong recurrence and no offsetting evidence
- Stop/target/invalidation logic is internally contradictory
- Plan cites assumptions not present in `PolicySnapshot` or retrieval evidence ("hallucinated inputs")
- Proposed playbook switch violates policy mutation cooldown with no invalidation/safety exception
- Proposed thesis re-evaluation during `THESIS_ARMED` activation window without a valid policy-boundary trigger
- Proposed target re-optimization during `HOLD_LOCK` without an allowed policy-boundary exception

### `revise` (repairable)

- Evidence is mixed and proposal confidence is too high
- Plan omits invalidation condition required by playbook
- Memory bundle includes recent losses in same regime that are not addressed in rationale
- Expected holding time is outside playbook P90 without justification
- MAE/MFE expectations are inconsistent with proposed stop sizing
- Memory contradiction exists but can be addressed with a concrete divergence explanation
- Policy mutation is requested before cooldown expiry but may be legal if reframed as an
  allowed mutation or tied to a documented invalidation trigger
- A boundary request arrives during `THESIS_ARMED`/`HOLD_LOCK` and must be reframed as a
  valid invalidation/safety event before policy re-evaluation can proceed

### `approve`

- Deterministic checks pass
- Playbook and regime alignment pass
- Memory evidence is supportive or mixed-but-addressed
- Confidence is calibrated to evidence strength

If approval proceeds despite contradictory loser memory, the judge must explicitly state
why this case diverges from the nearest loser set.

## Judge Accountability Requirements

Every non-approve-neutral verdict should include:

- explicit citation of retrieved episode IDs used in the argument
- judge confidence score (not only label)
- divergence explanation when overruling contradictory memory evidence

## Implementation Steps

### Step 1: Extend judge verdict schema

Add a typed validation section in `schemas/judge_feedback.py`, for example:

```python
class JudgeValidationVerdict(BaseModel):
    decision: Literal["approve", "revise", "reject", "stand_down"]
    finding_class: Literal["structural_violation", "statistical_suspicion", "memory_contradiction", "none"]
    reasons: list[str]
    judge_confidence_score: float          # 0.0-1.0
    memory_evidence_refs: list[str] = []
    cited_episode_ids: list[str] = []
    failure_pattern_matches: list[str] = []
    cluster_support_summary: str | None = None
    confidence_calibration: Literal["supported", "weakly_supported", "unsupported"]
    divergence_from_nearest_losers: str | None = None
    requested_revisions: list[str] = []
```

### Step 2: Implement memory-backed checks in `services/judge_feedback_service.py`

Judge review should consume:

- `PolicySnapshot` (policy-boundary snapshot)
- strategist proposal + playbook metadata
- diversified memory bundle (Runbook 51)
- playbook expectations (Runbook 52)

It should emit explicit evidence references, not just prose.
When memory contradictions exist, it should emit cited episode IDs and either:

- a revision request (default), or
- an explicit divergence explanation if approving despite contradictory memory

### Step 3: Add strategist revision loop handling

When judge returns `revise`, route structured revision requests back to strategist with:

- exact failing criteria
- cited memory failure patterns
- expectation mismatches to fix

Cap revision attempts per **policy event** (e.g., `max_revisions_per_policy_event`) to
avoid infinite loops.

Recommended default:

- `max_revisions_per_policy_event = 2`
- if exceeded, auto-emit `stand_down` (or equivalent no-trade verdict) for that policy event
  with explicit reason `revision_budget_exhausted`

### Step 4: Telemetry and audit trails

Persist:

- judge decision (`approve/revise/reject`)
- cited failure patterns and cluster evidence refs
- confidence calibration result
- revision count per strategist invocation

This allows later analysis of false rejections and weak judge enforcement.

## Acceptance Criteria

- [ ] Judge validation includes memory-backed checks in addition to risk/logic validation
- [ ] Judge outputs explicit `approve/revise/reject/stand_down` verdict with typed reasons
- [ ] Judge findings explicitly distinguish structural violation vs statistical suspicion vs memory contradiction
- [ ] Rejection criteria are documented and implemented explicitly (not prompt-only)
- [ ] Judge validation is enforced at policy boundaries; tick-level execution remains deterministic
- [ ] Judge enforces playbook/policy mutation cooldown and blocks illegal mid-trade switching
- [ ] Judge blocks thesis re-evaluation during `THESIS_ARMED` activation ticks unless a valid policy-boundary trigger exists
- [ ] Judge blocks target re-optimization during `HOLD_LOCK` unless explicitly allowed by playbook rules and boundary conditions
- [ ] Judge can cite failure-pattern matches from diversified memory retrieval
- [ ] Judge cites retrieved episode IDs used as evidence in revise/reject decisions
- [ ] Judge emits confidence score and confidence calibration result
- [ ] Judge requires explicit divergence explanation when approving against contradictory loser-memory evidence
- [ ] Judge checks proposal confidence against evidence strength / cluster support
- [ ] Revision loop passes structured requests back to strategist and caps retries
- [ ] Revision retries are capped per policy event (recommended default: 2) with auto-stand-down on exhaustion
- [ ] Telemetry records verdicts, evidence refs, and revision counts
- [ ] Tests cover hard reject, revise, and approve scenarios

## Test Plan

```bash
# Judge validation schema extensions
uv run pytest tests/test_judge_validation_verdict_schema.py -vv

# Memory-backed judge review rules
uv run pytest tests/test_judge_memory_validation.py -vv

# Accountability fields (confidence score, cited episodes, divergence explanation)
uv run pytest -k "judge_validation and accountability" -vv

# Revision loop orchestration
uv run pytest tests/test_judge_revision_loop.py -vv

# Policy-boundary cooldown enforcement / no mid-trade playbook switching
uv run pytest -k "judge and policy_cooldown" -vv

# Revision budget exhaustion -> stand_down
uv run pytest -k "judge and revision_budget_exhausted" -vv

# THESIS_ARMED / HOLD_LOCK boundary enforcement
uv run pytest -k "judge and thesis_armed_boundary" -vv
uv run pytest -k "judge and hold_lock_target_reopt" -vv

# Regression: attribution and action gating still hold
uv run pytest tests/test_judge_attribution_rules.py tests/test_judge_replan_gating.py -vv
```

## Human Verification Evidence

```text
1. Memory failure-pattern contradiction → revise:
   TestMemoryFailurePatternScan::test_strong_failure_mode_triggers_revise
   - Bundle: 0W/4L with false_breakout_reversion x3 → verdict.decision in {revise, reject}
   - verdict.failure_pattern_matches contains "false_breakout_reversion" ✅

2. Structural playbook violation → hard reject:
   TestPlaybookConsistency::test_ineligible_regime_rejects
   - Plan regime="bear", playbook_regime_tags=["bull","range"]
   - verdict.decision == "reject", finding_class == "structural_violation" ✅

3. Policy cooldown + playbook switch → reject:
   TestDeterministicHardReject::test_policy_cooldown_playbook_switch_no_exception_rejects
   - policy_cooldown_active=True, is_playbook_switch=True, no exceptions
   - verdict.decision == "reject", "cooldown" in reasons ✅

4. THESIS_ARMED boundary enforcement:
   TestDeterministicHardReject::test_thesis_armed_no_invalidation_rejects
   - is_thesis_armed=True, no invalidation/safety → hard reject ✅
   TestDeterministicHardReject::test_thesis_armed_with_invalidation_allows_through
   - With invalidation trigger → no THESIS_ARMED structural block ✅

5. HOLD_LOCK target reopt enforcement:
   TestDeterministicHardReject::test_hold_lock_no_override_rejects
   - is_hold_lock=True, no safety override → hard reject ✅
   TestDeterministicHardReject::test_hold_lock_with_safety_override_passes
   - With safety override → no HOLD_LOCK structural block ✅

6. Strong-evidence approve path:
   TestFullApprovePath::test_clean_plan_approves
   - Bundle 4W/1L, regime in playbook_regime_tags, stated_conviction="medium"
   - verdict.decision == "approve", confidence_calibration == "supported",
     judge_confidence_score > 0.5 ✅

7. Revision budget exhaustion → stand_down:
   TestRevisionBudgetExhaustion::test_budget_exhausted_after_max_revisions
   - AlwaysReviseService + max_revisions=2 → decision=="stand_down",
     revision_budget_exhausted==True ✅
   TestRevisionBudgetExhaustion::test_callback_called_max_revisions_times
   - Callback called exactly 2 times ✅
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-22 | Runbook created — judge validation rules upgraded with memory and cluster evidence | Codex |
| 2026-02-28 | Implemented: schemas/judge_feedback.py (JudgeValidationVerdict, JudgePlanRevisionRequest, RevisionLoopResult), services/judge_validation_service.py (JudgePlanValidationService — 5-layer gate), services/judge_revision_loop.py (JudgePlanRevisionLoopOrchestrator — max 2 revisions, auto-stand-down), 3 test files (77 tests, all passing) | Claude |

## Test Evidence (append results before commit)

```text
uv run pytest tests/test_judge_validation_verdict_schema.py \
              tests/test_judge_memory_validation.py \
              tests/test_judge_revision_loop.py -vv

77 passed in 23.88s

Regression check (existing judge tests):
uv run pytest tests/test_judge_attribution_rules.py \
              tests/test_judge_replan_gating.py \
              tests/test_judge_attribution_schema.py -vv

62 passed in 12.48s  (no regressions)
```

## Worktree Setup

```bash
git worktree add -b feat/judge-validation-memory-evidence ../wt-judge-memory-validation main
cd ../wt-judge-memory-validation

# When finished
git worktree remove ../wt-judge-memory-validation
```

## Git Workflow

```bash
git checkout -b feat/judge-validation-memory-evidence

git add schemas/judge_feedback.py \
  services/judge_feedback_service.py \
  agents/judge_agent_client.py \
  agents/strategies/plan_provider.py \
  schemas/llm_strategist.py \
  tests/test_judge_validation_verdict_schema.py \
  tests/test_judge_memory_validation.py \
  tests/test_judge_revision_loop.py \
  docs/branching/53-judge-validation-rules-memory-evidence.md \
  docs/branching/README.md

uv run pytest tests/test_judge_validation_verdict_schema.py tests/test_judge_memory_validation.py -vv

# Commit only after Test Evidence + Human Verification Evidence are recorded
git commit -m "feat: add memory-backed judge validation and rejection rules (Runbook 53)"
```

# Runbook 50: Dual Reflection Templates

## Purpose

Implement a dual-level reflection framework for reasoning agents:

- **Policy-Level (fast) reflection** in the event-driven policy loop
- **High-Level (slow) reflection** on a scheduled batch cadence

Plus a separate **tick-level deterministic validation layer** inside the trigger engine
(not an LLM reflection loop).

This runbook converts reflection from a vague prompt habit into a typed, schedulable
control loop with explicit latency and invocation rules. It should remain grounded in
the codebase's existing strategist/judge separation rather than introducing a new agent
type.

## Why This Exists

Current strategist/judge infrastructure already supports prompt updates, judge actions,
and outcome analysis. What is missing is a clean split between:

- policy-boundary coherence checks (must be lightweight), and
- periodic strategic learning/review (must use batched evidence and not run constantly).

Without this split, systems tend to do one of two bad things:

1. Run expensive "reflection" at every tick (latency blow-up, noise amplification)
2. Skip reflection entirely and rely on post-hoc judge interventions only

## Scope

1. **`schemas/reflection.py`** — new reflection request/response schemas
2. **`services/policy_level_reflection_service.py`** (or renamed `low_level_reflection_service.py`) — fast policy-boundary reflection logic
3. **`services/high_level_reflection_service.py`** — batch review + drift analysis
4. **`trading_core/trigger_engine.py`** / execution path — deterministic tick-level validation hooks
5. **`agents/strategies/plan_provider.py`** — invoke policy-level reflection before policy freeze
6. **`agents/judge_agent_client.py`** — consume high-level reflection outputs in evaluations/replans
7. **`prompts/policy_level_reflection.txt`** (or `prompts/low_level_reflection.txt`) — optional lightweight template for LLM-backed checks
8. **`prompts/high_level_reflection.txt`** — structured batch review template
9. **Temporal scheduling path** (`worker/*`, workflow timers, or service loop) for high-level cadence
10. **Telemetry/events** — persist reflection outputs, latency, skip reasons, and batch counts
11. **Tests** for cadence gating, schemas, and invocation rules

## Out of Scope

- Replacing the judge with a reflection loop (judge remains the decision gate for interventions)
- High-level reflection on every tick / bar (explicitly forbidden)
- Automated parameter mutation without judge/action contract validation
- New market data features (depends on Runbook 49 for standardized input)

## Reflection Contracts

## Tick-Level Validation (Deterministic, Every Tick/Bar)

This runs inside the trigger engine / execution layer and is **not** an LLM reflection path.

### Trigger

Every tick/bar as part of deterministic execution.

### Responsibilities

- stop breach detection
- target breach detection
- time stop / expiry checks
- activation trigger / micro-entry refinement checks for `THESIS_ARMED` setups
- risk-envelope breach checks
- deterministic state transitions and logging

### Hard Rule

No LLM calls. No memory retrieval. No policy/playbook mutation.

## Policy-Level Reflection (Fast Path, Event-Driven Policy Loop)

### Trigger

Run only during **policy loop events**, immediately after strategist proposal generation
and before policy freeze / judge validation.

Policy loop events include (configurable):

- regime state change detected
- position opened
- position closed
- significant volatility percentile band shift
- policy heartbeat elapsed (e.g., 15m for 1m systems, 1h for 5m systems)

Policy-level reflection is **not** invoked during:

- `THESIS_ARMED` activation-window tick checks
- `HOLD_LOCK` deterministic trade management ticks

except when a documented invalidation or safety override explicitly opens a policy boundary.

### Inputs

- `PolicySnapshot` (Runbook 49)
- strategist proposal (typed plan / playbook selection)
- policy state context (`THESIS_ARMED`, `POSITION_OPEN`, `HOLD_LOCK`, etc.)
- memory retrieval summary (Runbook 51)
- invariant context (risk caps, policy constraints, no-learn zones, kill switches)

### Required Checks

1. **Coherence check**
   - Does rationale match the selected playbook and triggers?
   - Are direction, stop/target, and invalidation internally consistent?
2. **Invariant check**
   - Any conflict with deterministic guardrails?
3. **Contrastive memory check**
   - Do retrieved failure patterns materially contradict the proposal?
4. **Expectation calibration**
   - Does stated conviction align with playbook hold-time / MAE/MFE expectations?

### Explicit Non-Scope for Policy-Level Reflection (Forbidden)

Policy-level reflection must not:

- re-cluster regimes across multi-episode history
- update playbook priors or historical statistics
- rewrite playbooks or eligibility rules
- tune memory distance metrics
- perform multi-episode statistical reanalysis

Those are high-level reflection or scheduled analytics tasks. This boundary is required to
avoid latency blowups and real-time cognitive thrashing.

### Output (typed)

```python
class PolicyLevelReflectionResult(BaseModel):
    status: Literal["pass", "revise", "block"]
    coherence_findings: list[str]
    invariant_findings: list[str]
    memory_findings: list[str]
    expectation_findings: list[str]
    requested_revisions: list[str] = []
    latency_ms: int
```

### Latency Requirement

Keep this path lightweight. Target p95 latency should be a small fraction of the
policy-event budget (for example `< 250-500 ms` depending on heartbeat/regime event path).

Deterministic checks should run first. LLM usage in the fast path must be optional,
bounded, and skippable.

If policy-level reflection exceeds budget, it must degrade safely (deterministic checks only)
rather than silently spilling heavy analysis into the trigger tick loop.

## High-Level Reflection (Structural Learning Loop, Slow Path)

### Trigger

Scheduled (daily/weekly) or event-based after a minimum batch of resolved episodes.
Never invoked inside the tick loop. In production, structural changes should be weekly+
unless a documented safety override applies.

### Inputs

- batch of episodes/signals with outcomes
- playbook-level and regime-level performance stats
- drift metrics (regime fingerprint changes, trigger fire quality changes)
- judge interventions/actions over the same window

### Required Outputs

1. Outcome clusters (what worked / what failed / where)
2. Regime drift assessment
3. Playbook update candidates (thresholds, eligibility, invalidation)
4. Confidence in recommendations + evidence references
5. Recommended next action (`hold`, `replan`, `policy_adjust`, `research_experiment`)

### Minimum Sample Size / Update Guardrails (Required)

High-level reflection may summarize any window, but **structural recommendations** must be
gated by sample size and evidence quality.

Baseline rules:

- No structural playbook change recommendation unless:
  - `>= 20` resolved trades in the relevant regime/playbook cluster, **or**
  - statistically significant expectancy deviation (per configured test/threshold)
- No regime eligibility refinement from fewer than the configured minimum regime samples
- If thresholds are not met, report `insufficient_sample` and restrict output to monitoring
  findings (no structural change recommendation)

### Output (typed)

```python
class HighLevelReflectionReport(BaseModel):
    window_start: datetime
    window_end: datetime
    n_episodes: int
    regime_cluster_summary: list[dict]
    playbook_findings: list[dict]
    drift_findings: list[str]
    recommendations: list[dict]
    evidence_refs: list[str]
```

## Implementation Steps

### Step 1: Add reflection schemas

Create typed request/response contracts in `schemas/reflection.py` for:

- `PolicyLevelReflectionRequest/Result`
- `HighLevelReflectionRequest/Report`
- `ReflectionInvocationMeta` (cadence, source, skip reason, latency)
- `TickValidationEvent` / `TickValidationResult` (optional telemetry-only schema)

### Step 2: Implement policy-level reflection service

`services/policy_level_reflection_service.py` (or existing low-level service renamed in
place) should run checks in this order:

1. deterministic invariants (cheap)
2. structural/coherence checks (cheap)
3. memory contradiction scan (bounded retrieval summary)
4. optional LLM consolidation (timeout-bounded)

If any deterministic invariant fails, return `block` immediately without expensive work.

Role-boundary constraint for optional LLM consolidation:

- it may **summarize, validate, or request revision** of the strategist proposal
- it may **not** introduce new triggers, new feature requirements, or new structural
  hypotheses/playbooks

If the reflection step needs a new trigger/feature/hypothesis, it must emit a
`revise`/`reject` finding for the strategist or defer to high-level reflection.

### Step 3: Add deterministic tick-level validation hooks

Implement or formalize deterministic tick-level validation in the trigger engine for:

- stop/target/time-stop/risk-envelope checks
- position-state transitions
- execution-safe state updates

This is intentionally separate from policy-level reflection.

### Step 4: Implement high-level reflection service

`services/high_level_reflection_service.py` should batch episodes and produce a report
that can be consumed by the judge and by human review.

High-level reflection should analyze:

- outcome clusters by playbook/regime/timeframe
- expectation miss patterns (actual hold vs expected P50/P90; MAE/MFE drift)
- failure-mode recurrence
- drift indicators warranting playbook or regime updates

### Step 5: Scheduling + gating rules

High-level reflection must be gated by both **time** and **sample size**:

- minimum elapsed interval (e.g., daily or weekly)
- minimum resolved episodes (e.g., `N >= 20` or playbook-specific threshold)

Structural updates (playbook eligibility/threshold changes) require an additional gate:

- minimum regime-cluster sample size or significance threshold
- otherwise emit monitor-only findings with explicit insufficiency reason

If either gate fails, emit a skip event with explicit reason.

### Step 6: Wire outputs into strategist/judge flow

- Trigger engine runs deterministic tick-level validation on every bar (no LLM)
- Strategist policy loop consumes `PolicyLevelReflectionResult` before policy freeze
- Judge path consumes `HighLevelReflectionReport` as evidence, not as direct command
- Judge policy-boundary validation consumes policy-level reflection evidence (Runbook 53)
- Reflection artifacts are persisted for later audits

Policy-level reflection remains suppressed during activation-window/HOLD_LOCK ticks unless
invalidation/safety paths explicitly reopen a policy boundary.

## Acceptance Criteria

- [ ] Tick-level validation runs on every tick/bar and is deterministic-only (no LLM, no memory retrieval)
- [ ] Policy-level reflection runs only during policy loop events (not every tick)
- [ ] Policy-level reflection is suppressed during `THESIS_ARMED` activation ticks and `HOLD_LOCK` management ticks unless invalidation/safety override opens a policy boundary
- [ ] Policy-level reflection can `pass`, `revise`, or `block` with typed findings
- [ ] Deterministic invariants run before any optional LLM reflection logic in the policy loop
- [ ] Policy-level reflection explicitly forbids slow-loop tasks (playbook rewrites, priors, regime reclustering, distance-metric tuning)
- [ ] Optional LLM consolidation in policy-level reflection cannot introduce new triggers/features/structural hypotheses (validate/reject only)
- [ ] High-level reflection runs only on scheduled cadence and/or after batch threshold
- [ ] High-level reflection is not triggered on every tick
- [ ] High-level reflection enforces minimum sample-size/significance gates before structural recommendations
- [ ] High-level reflection outputs include outcome clusters and drift findings
- [ ] Reflection invocations record latency, batch size, and skip reasons
- [ ] Strategist/judge telemetry persists reflection artifact references

## Test Plan

```bash
# Reflection schemas
uv run pytest tests/test_reflection_schema.py -vv

# Policy-level reflection invariants + status transitions
uv run pytest tests/test_low_level_reflection.py -vv

# Tick-level deterministic validation (no LLM path)
uv run pytest -k "tick_validation and trigger_engine" -vv

# High-level reflection cadence gating and batching
uv run pytest tests/test_high_level_reflection.py -vv

# Guardrails: no structural updates under insufficient sample
uv run pytest -k "high_level_reflection and insufficient_sample" -vv

# Integration: policy loop invokes policy-level reflection pre-freeze
uv run pytest -k "plan_provider and policy_reflection" -vv

# Integration: judge consumes high-level reflection evidence
uv run pytest -k "judge and high_level_reflection" -vv
```

## Human Verification Evidence

```text
TODO:
1. Trigger repeated bars/ticks and confirm only deterministic tick validation runs (no
   LLM calls, no memory retrieval).
2. Trigger one policy event with an intentionally inconsistent plan (e.g., invalidation
   contradicts stop direction) and confirm policy-level reflection returns status=revise/block.
3. Run repeated ticks within the same policy window and confirm policy-level reflection
   and judge policy validation do NOT re-run unless a policy event fires.
4. Force a high-level reflection window with fewer than the minimum samples and confirm
   output is monitor-only (or skipped) with `insufficient_sample`.
5. Force the daily/weekly schedule (or fixture) with sufficient samples and confirm a
   batch report is produced with n_episodes, drift findings, and gated recommendations.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-22 | Runbook created — dual-level reflection framework with cadence and latency constraints | Codex |

## Test Evidence (append results before commit)

```text
TODO
```

## Worktree Setup

```bash
git worktree add -b feat/dual-reflection-templates ../wt-dual-reflection main
cd ../wt-dual-reflection

# When finished
git worktree remove ../wt-dual-reflection
```

## Git Workflow

```bash
git checkout -b feat/dual-reflection-templates

git add schemas/reflection.py \
  services/low_level_reflection_service.py \
  services/high_level_reflection_service.py \
  trading_core/trigger_engine.py \
  agents/strategies/plan_provider.py \
  agents/judge_agent_client.py \
  prompts/low_level_reflection.txt \
  prompts/high_level_reflection.txt \
  worker/ \
  tests/test_reflection_schema.py \
  tests/test_low_level_reflection.py \
  tests/test_high_level_reflection.py \
  docs/branching/50-dual-reflection-templates.md \
  docs/branching/README.md

uv run pytest tests/test_reflection_schema.py tests/test_low_level_reflection.py tests/test_high_level_reflection.py -vv

# Commit only after Test Evidence + Human Verification Evidence are recorded
git commit -m "feat: add dual-level reflection templates and cadence gates (Runbook 50)"
```

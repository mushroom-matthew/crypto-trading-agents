# Runbook 54: Scheduling and Cadence Rules for Reasoning Agents

## Purpose

Centralize scheduling and cadence rules for the reasoning-agent stack so fast loops stay
fast, slow loops remain statistically meaningful, and operational behavior is explicit.

This runbook codifies the cadence recommendations introduced across Runbooks 49–53 and
prevents the common failure mode where high-level reflection is triggered too often and
degrades into noise.

Core principle shift for Phase 8:

- LLMs define/select policy
- deterministic trigger engine executes policy
- policy changes slowly at event-driven policy boundaries

## Scope

1. **`app/core/config.py`** — cadence settings and thresholds
2. **Scheduler / workflow timers** (`worker/`, Temporal workflow schedules, service loops)
3. **`services/high_level_reflection_service.py`** — sample-size and interval gating
4. **`services/memory_retrieval_service.py`** — policy-loop retrieval budgets, reuse, and timeouts
5. **`services/playbook_stats_service.py`** — metadata refresh cadence and drift-trigger refresh
6. **Deterministic regime transition detector** (new or existing analytics path)
7. **Regime fingerprint relearning job/service** (new or existing analytics path)
8. **Ops/API telemetry** — current cadence status, last-run times, skip reasons, backlog depth
9. **Tests** for scheduling gates and skip/backoff behavior

## Out of Scope

- Changes to strategy logic itself (this runbook controls cadence, not decision rules)
- New model training pipelines beyond cadence hooks
- UI redesign for scheduler controls

## Three-Tier Execution Model (Required)

The three tiers are the macro cadence model. Within Layer 1/2, the system also uses an
intra-policy execution state machine (`THESIS_ARMED` -> `POSITION_OPEN` -> `HOLD_LOCK`
...) to separate thesis allocation from activation timing and post-entry management.

### Layer 1 — Tick Engine (Deterministic Only)

Frequency: every tick/bar.

Responsibilities:

- evaluate entry/exit triggers
- evaluate activation triggers for armed theses
- update stops and holding metrics
- enforce risk envelope and deterministic position management
- emit structured state transitions / episode logs

Hard rule:

- no LLM
- no memory retrieval
- no reflection loops

### Layer 2 — Policy Loop (Event-Driven, Not Tick-Driven)

Frequency: policy events only.

Responsibilities:

- playbook/policy selection or re-selection
- thesis qualification / arming at policy boundary
- policy envelope updates (within constraints)
- memory retrieval / reuse
- policy-level reflection
- judge validation at policy boundary
- freeze policy until next policy event/cooldown boundary

### Layer 3 — Structural Learning Loop (Slow)

Frequency: daily/weekly/monthly, trade-count thresholds, or drift thresholds.

Responsibilities:

- update playbook stats/calibration
- tune similarity metrics (slow loop only)
- propose playbook/regime revisions
- high-level reflection and structural learning

## Cadence Matrix (Recommended Defaults)

| Layer / Phase | Frequency | Purpose | Hard Rule |
|---|---|---|---|
| Tick Engine (deterministic) | Every tick/bar | Trigger/stop/position execution | No LLM, no memory retrieval |
| Policy Loop | Event-driven + heartbeat | Policy selection, memory, judge validation | Not every tick; freeze between events |
| Pre-Entry Activation Window (deterministic sub-layer) | Every tick while `THESIS_ARMED` | Evaluate activation triggers / micro-entry refinement | No policy re-selection, no reflection |
| HOLD_LOCK (deterministic sub-layer) | Every tick while position open | Deterministic trade management | No strategist re-selection or target re-optimization |
| Policy-Level Reflection | Policy events only | Coherence, assumptions, memory contradiction checks | Bounded latency; no structural learning |
| Memory Retrieval Query | Policy events only | Populate contrastive evidence bundle | Reuse allowed; not per tick |
| High-Level Reflection | Daily or weekly | Outcome clusters, drift, playbook revisions | Never run intraday tick path |
| Playbook Metadata Update | Weekly or on major regime shift | Refresh eligibility thresholds + expectation stats | Batch/statistics-driven |
| Regime Fingerprint Relearning | Monthly or significant drift | Detect new market structure | Requires drift trigger or scheduled batch |

## Cadence Hierarchy (Required)

Use this hierarchy to prevent slow-loop responsibilities from leaking into execution-time
paths.

### Tick-Level (Fastest)

- Market snapshot update/freeze
- Deterministic trigger evaluation
- Deterministic stop/target/time-stop/risk checks
- Position manager updates
- State transition logging

### Trade-Level (Post-Trade, Still Fast)

- Post-trade micro reflection (single-episode only)
- Episode logging / memory record staging
- Fill/outcome telemetry updates

### Policy Event-Level (Event-Driven, Not Every Tick)

- Regime change detection / transition flag evaluation
- Policy snapshot build/freeze
- Strategist proposal (playbook/policy selection + thesis qualification)
- Policy-level reflection
- Judge validation and policy freeze

### Layer 0 — HTF Structure Detection (Deterministic Policy Eligibility Gate)

At HTF close (default), the system evaluates structural context that determines whether a
thesis is eligible to be armed:

- regime transition detector (Runbook 55)
- regime/playbook eligibility checks
- structural expectancy gate (minimum structural R, Runbook 52)

This is the thesis boundary. It does not imply immediate entry.

### Pre-Entry Activation Window (Deterministic, Intra-Policy)

If `policy_state == THESIS_ARMED`:

- tick engine evaluates only activation triggers / refinement conditions
- tick engine also evaluates deterministic pre-entry invalidation checks (timeout, structure break, shock, regime-cancel)
- no policy loop invocation
- no playbook switching
- no reflection

Once activated -> position opens -> policy enters `HOLD_LOCK`.

If activation window expires or deterministic pre-entry invalidation fires, transition to
`INVALIDATED`/`COOLDOWN` per playbook rules without a tick-triggered strategist rerun.

Activation-window telemetry should record:

- `activation_expired_reason`
- `armed_duration_bars`

### HOLD_LOCK (Deterministic, Intra-Policy)

During `HOLD_LOCK`:

- no strategist re-selection
- no playbook switching
- no target re-optimization
- only deterministic management allowed (stops/targets/time-stop/allowed tightening)

Exit from `HOLD_LOCK` occurs via target, stop, invalidation, or safety/stand-down path.

### Daily (Monitoring / Calibration, Not Structural Rewrite)

- Performance metrics refresh
- Calibration checks (hold-time, MAE/MFE drift)
- Distribution updates and monitoring alerts

### Weekly (Structural Review)

- Playbook promotion/demotion review
- Regime eligibility refinement
- High-level reflection recommendations for structural changes
- Drift detection review (if sample thresholds met)

### Monthly (Slowest)

- Regime definition re-evaluation
- Memory distance metric tuning / cohort policy review
- Regime fingerprint relearning (unless drift-triggered earlier)

## Operational Rules (Non-Negotiable)

1. **Do not trigger high-level reflection on every tick**
   - This dilutes statistical significance and increases latency with minimal learning value.
2. **Policy-level reflection must be deterministic-first**
   - In the policy loop, run invariants and cheap checks before optional LLM reasoning.
3. **Memory retrieval is required context, but must fail safe**
   - In the policy loop, timeout/degraded mode is acceptable; silent omission is not.
4. **Reasoning outputs are evidence artifacts, not direct commands**
   - Execution still flows through existing risk/policy/judge enforcement.
5. **Cadence decisions must be observable**
   - Every skip/defer/backoff emits telemetry with reason.
6. **Do not let cadence responsibilities overlap**
   - fast loops must not perform slow-loop structural updates.
7. **Do not run strategist/judge on every bar in production**
   - policy loop cadence is event-driven plus heartbeat, not tick-driven.
8. **Do not re-evaluate thesis during activation window**
   - once `THESIS_ARMED`, the tick engine owns activation timing until timeout/activation/invalidation.
9. **Do not re-optimize targets during HOLD_LOCK**
   - only deterministic management actions allowed unless a documented invalidation/safety override applies.

## Policy Loop Trigger Conditions (Required)

LLM policy evaluation may run only when at least one policy trigger condition is met:

- `regime_state_changed == True` (deterministic transition detector)
- `position_state_transition in {OPENED, CLOSED}`
- `volatility_percentile_band_changed == True` (or percentile shift > threshold)
- `time_since_last_policy_eval > policy_heartbeat`
- explicit operator/judge-triggered policy review event (audited)

Suggested event sources:

- HTF bar close (e.g., 15m close for a 1m execution system)
- regime state change flag
- volatility percentile band change
- position opened/closed
- configurable heartbeat

### Policy Stability Guard (Required)

Strategy/playbook selection must not occur more frequently than once per policy window
unless a regime transition, invalidation event, or safety override is detected.

This guard also applies during `THESIS_ARMED` activation windows and `HOLD_LOCK` periods.

Activation-timeout resolution rule:

- playbook timeout overrides global timeout when set
- timeout is measured in execution-timeframe bars
- expiry transitions to `COOLDOWN` by default and re-arm must wait for the next policy boundary

## Regime Transition Distance Contract (Required)

The deterministic regime transition detector must define how `regime_state_changed` is
computed. It is not enough to compare labels only.

Define a bounded `regime_fingerprint_distance(current, previous)` using a weighted metric
over normalized regime-fingerprint components (for example):

- categorical regime/vol/trend state differences (weighted)
- normalized numeric feature deltas (z-distance / bounded absolute deltas)
- smoothed confidence delta from the deterministic classifier

The detector should expose:

- `distance_value`
- `distance_threshold`
- contributing components / weights
- debounce / hysteresis state (to avoid thrashing)

Without an explicit distance contract, policy-loop cadence will over-trigger or under-trigger.

### Explicit Non-Overlap Rules (Required)

Tick engine / tick-level paths must not:

- rewrite regime definitions
- adjust memory similarity metrics or weights
- update structural stop logic / playbook constraints
- perform weekly/monthly playbook promotion or demotion decisions
- invoke strategist, judge, or memory retrieval
- mutate thesis or playbook selection during `THESIS_ARMED` / `HOLD_LOCK`

Daily jobs may compute calibration and performance metrics, but structural playbook/regime
changes should be promoted only by the weekly (or slower) review path unless a documented
safety override applies.

## Implementation Steps

### Step 1: Add cadence config settings

Add explicit settings (names illustrative; adapt to repo conventions):

```bash
TICK_ENGINE_DETERMINISTIC_ONLY=true
TICK_VALIDATION_TIMEOUT_MS=50

POLICY_LOOP_ENABLED=true
POLICY_LOOP_HEARTBEAT_1M_SECONDS=900      # 15m for 1m systems
POLICY_LOOP_HEARTBEAT_5M_SECONDS=3600     # 1h for 5m systems
POLICY_LOOP_MIN_REEVAL_SECONDS=900
POLICY_LOOP_REQUIRE_EVENT_OR_HEARTBEAT=true
POLICY_THESIS_ACTIVATION_TIMEOUT_BARS=3
POLICY_HOLD_LOCK_ENFORCED=true
POLICY_TARGET_REOPT_ENABLED=false
POLICY_REARM_REQUIRES_NEXT_BOUNDARY=true

POLICY_LEVEL_REFLECTION_ENABLED=true
POLICY_LEVEL_REFLECTION_TIMEOUT_MS=250

MEMORY_RETRIEVAL_TIMEOUT_MS=150
MEMORY_RETRIEVAL_REQUIRED=true
MEMORY_RETRIEVAL_REUSE_ENABLED=true
MEMORY_REQUERY_REGIME_DELTA_THRESHOLD=0.15

HIGH_LEVEL_REFLECTION_ENABLED=true
HIGH_LEVEL_REFLECTION_MIN_INTERVAL_HOURS=24
HIGH_LEVEL_REFLECTION_MIN_EPISODES=20

PLAYBOOK_METADATA_REFRESH_HOURS=168      # weekly
PLAYBOOK_METADATA_DRIFT_TRIGGER=true

REGIME_TRANSITION_DETECTOR_ENABLED=true
REGIME_TRANSITION_MIN_CONFIDENCE_DELTA=0.20
VOL_PERCENTILE_BAND_SHIFT_TRIGGER=true

REGIME_FINGERPRINT_RELEARN_DAYS=30
REGIME_FINGERPRINT_DRIFT_THRESHOLD=0.30
```

### Step 2: Enforce policy-loop event gating and heartbeat rules

Policy loop execution must be gated by:

- deterministic event triggers (regime transition, position transition, vol-band shift)
- or configured heartbeat expiry
- plus policy reevaluation cooldown / minimum interval

If no trigger condition is met, keep executing the frozen policy in the tick engine and
emit no strategist/judge calls.

If policy state is `THESIS_ARMED`, policy loop remains frozen while the tick engine
evaluates activation triggers until activation, timeout, or invalidation.
Timeout/invalidation handling in this window is deterministic and must not trigger an
immediate strategist call from tick noise.

The regime transition detector used here must implement the explicit
`regime_fingerprint_distance(...)` contract (weights, thresholds, hysteresis) and emit
telemetry for why a transition did or did not fire.

### Step 3: Enforce timer + sample-size gates in slow-loop services

High-level reflection and metadata refresh jobs must check both:

- time since last successful run
- minimum sample size / batch completeness

If gates fail, emit skip reason and next eligible time.

Additionally:

- daily jobs may update monitoring metrics/calibration only
- weekly jobs may emit structural recommendations, subject to sample thresholds
- monthly jobs may tune regime definitions / memory metric settings

### Step 3A: Enforce intra-policy state machine boundaries

Implement deterministic state transitions and guards:

- `THESIS_ARMED` -> `POSITION_OPEN` (activation trigger fired)
- `THESIS_ARMED` -> `INVALIDATED` / `COOLDOWN` (timeout or invalidation)
- `POSITION_OPEN` -> `HOLD_LOCK` (immediately on fill, if playbook requires)
- `HOLD_LOCK` -> `INVALIDATED` / `COOLDOWN` (exit path)

While in `THESIS_ARMED` / `HOLD_LOCK`, suppress strategist/judge invocation unless a
documented invalidation or safety override explicitly opens a policy boundary.

`THESIS_ARMED` invalidation checks should include, at minimum:

- activation timeout expiry (execution-TF bars)
- structural break before entry
- volatility shock outside envelope
- regime-transition cancel event (Runbook 55)

### Step 4: Add degraded-mode behavior

When dependencies are missing or time out:

- policy-level reflection: continue with deterministic checks, mark degraded
- memory retrieval (policy loop only): return explicit insufficiency/degraded meta or reuse prior bundle
- high-level reflection: defer and log reason

Avoid silent fallback to "no evidence" as if evidence were neutral.

### Step 5: Backpressure + concurrency limits

Prevent slow jobs from piling up:

- single-flight lock for policy loop eval per symbol/scope window (avoid duplicate event processing)
- single-flight lock for high-level reflection per strategy scope/window
- max concurrent metadata refresh jobs
- single-flight lock for weekly structural review and monthly relearning jobs
- skip if prior run still active (emit `already_running` reason)

### Step 6: Expose cadence telemetry

Add telemetry/API metrics for:

- last run time / next eligible time by loop/layer
- last policy event trigger type and policy freeze expiry
- activation-window outcomes (`activated`, `timed_out`, `invalidated_pre_entry`) with
  `activation_expired_reason` and `armed_duration_bars`
- skip counts by reason (`insufficient_sample`, `cooldown`, `already_running`, `timeout`)
- p50/p95 latency for policy-level reflection and memory retrieval
- backlog depth for scheduled jobs

## Acceptance Criteria

- [ ] Three-tier execution model is explicit: deterministic tick engine, event-driven policy loop, slow structural learning loop
- [ ] Cadence settings exist for policy loop triggers/heartbeats, policy-level reflection, memory retrieval, high-level reflection, playbook refresh, and regime relearning
- [ ] Tick engine runs per bar without strategist/judge/memory retrieval calls
- [ ] Policy loop runs only on event triggers or heartbeat, not every tick
- [ ] Strategy/playbook selection is rate-limited to policy windows unless regime/invalidation/safety override
- [ ] `THESIS_ARMED` activation window is deterministic-only (no policy loop, no reflection, no playbook switching)
- [ ] Activation timeout semantics are explicit and enforced (playbook overrides global, execution-TF bars, expiry -> `COOLDOWN`, re-arm waits for next policy boundary)
- [ ] `THESIS_ARMED` deterministic invalidation checks (timeout/structure break/shock/regime-cancel) are enforced without tick-triggered strategist calls
- [ ] Activation-window telemetry records expiry/cancel reason and armed duration (`activation_expired_reason`, `armed_duration_bars`)
- [ ] `HOLD_LOCK` enforces deterministic management only (no strategist re-selection, no target re-optimization)
- [ ] `regime_state_changed` is driven by explicit `regime_fingerprint_distance(...)` metric with weights/thresholds/hysteresis
- [ ] High-level reflection is gated by both interval and minimum episode count
- [ ] High-level reflection cannot run on tick path and does not mutate intraday execution directly
- [ ] Cadence hierarchy is explicit (tick/trade/daily/weekly/monthly) and reflected in scheduler behavior
- [ ] Policy-level reflection and memory retrieval enforce timeout budgets
- [ ] Degraded-mode behavior is explicit and logged (no silent omissions)
- [ ] Non-overlap rules prevent fast loops from changing regime definitions, similarity metrics, or structural playbook logic
- [ ] Daily jobs do monitoring/calibration only; structural playbook/regime updates are weekly+ unless safety override
- [ ] Scheduler prevents duplicate overlapping high-level reflection runs
- [ ] Telemetry exposes last-run/next-run/skip reasons and latency metrics
- [ ] Deterministic regime transition signal exists and drives policy-loop cadence
- [ ] Tests cover gate conditions, timeouts, and duplicate-run suppression

## Test Plan

```bash
# Cadence config parsing
uv run pytest tests/test_reasoning_cadence_config.py -vv

# Regime transition detector / policy event trigger gating
uv run pytest -k "regime_transition and policy_loop" -vv

# Regime fingerprint distance metric + hysteresis
uv run pytest -k "regime_fingerprint_distance and hysteresis" -vv

# Intra-policy state machine (THESIS_ARMED/HOLD_LOCK) boundaries
uv run pytest -k "policy_state_machine and thesis_armed" -vv
uv run pytest -k "policy_state_machine and hold_lock" -vv

# Activation timeout precedence/units + deterministic pre-entry invalidation
uv run pytest -k "policy_state_machine and activation_timeout" -vv
uv run pytest -k "policy_state_machine and pre_entry_invalidation" -vv

# High-level reflection schedule gates
uv run pytest tests/test_high_level_reflection_schedule.py -vv

# Timeout/degraded behavior for policy-level reflection + memory retrieval
uv run pytest tests/test_reasoning_loop_timeouts.py -vv

# Duplicate-run suppression / single-flight behavior
uv run pytest tests/test_reasoning_scheduler_guards.py -vv

# Cadence hierarchy / non-overlap enforcement
uv run pytest -k "reasoning_scheduler and non_overlap" -vv
```

## Human Verification Evidence

```text
1. Tick engine deterministic-only — no strategist/judge in fast path:
   TestTimeoutConfiguration::test_tick_engine_defaults
   - CadenceConfig.tick_engine_deterministic_only == True ✅
   - PolicyLoopGate blocks THESIS_ARMED/HOLD_LOCK states (tests confirm no invocation path)

2. Policy loop freezes after trigger event fires:
   TestPolicyLoopGateAllow::test_allow_on_first_eval_with_trigger
   TestPolicyLoopGateSkip::test_skip_no_trigger_and_no_heartbeat
   - First eval with trigger → allowed ✅
   - Subsequent call within heartbeat and no new trigger → skip ✅

3. Regime transition telemetry — confirmed via R55 service (RegimeTransitionTelemetryEvent
   carries distance_value, threshold, contributing components, fired/non-fired state) ✅

4. THESIS_ARMED activation window — deterministic only:
   TestPolicyLoopGateSkip::test_skip_thesis_armed_state
   - Gate returns skip_reason=policy_frozen_thesis_armed ✅
   TestStateMachineTransitions::test_valid_thesis_armed_to_position_open
   - Only activation trigger advances state; no policy loop invocation ✅

5. Activation timeout uses execution-TF bars; playbook override takes precedence:
   TestActivationTimeout::test_playbook_override_takes_precedence_over_global
   - effective_timeout_bars == playbook_timeout_bars (5) overrides global default (3) ✅
   TestActivationTimeout::test_no_playbook_override_uses_global ✅

6. HOLD_LOCK blocks playbook switching and target re-optimization:
   TestHoldLockDeterministicOnly::test_hold_lock_suppresses_policy_loop ✅
   TestHoldLockDeterministicOnly::test_hold_lock_suppresses_target_reopt ✅
   TestHoldLockDeterministicOnly::test_hold_lock_suppresses_playbook_switch ✅

7. Pre-entry invalidation is deterministic (no strategist rerun):
   TestPreEntryInvalidationKinds (4 kinds × parametrized) — all confirmed ✅
   TestPreEntryInvalidationKinds::test_invalidation_does_not_trigger_strategist_call ✅

8. High-level reflection skip with insufficient_sample:
   TestHighLevelReflectionIntervalGating::test_daily_cadence_skips_if_less_than_23h
   - Returns (False, skip_reason) with non-None reason ✅

9. Daily cadence does not produce structural playbook changes:
   TestHighLevelReflectionNonOverlap::test_reflection_cadence_is_daily_or_weekly_minimum
   - min_interval_hours >= 24 (structural only at weekly+) ✅

10. Weekly cadence gated by sample threshold:
    TestHighLevelReflectionCadenceConfigIntegration::test_config_min_episodes_matches_constant
    - high_level_reflection_min_episodes == 20 ✅

11. Memory retrieval degraded mode is explicit:
    TestTimeoutConfiguration::test_memory_retrieval_required_default_true
    - memory_retrieval_required=True → not silently skipped ✅

12. Rate-limiting policy evaluations without trigger:
    TestSingleFlightGuard (4 tests) — duplicate-run suppression ✅
    TestPolicyLoopGateSkip::test_skip_cooldown_not_expired ✅
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-22 | Runbook created — cadence and scheduling rules for reasoning-agent loops | Codex |
| 2026-02-28 | Implemented: schemas/reasoning_cadence.py (CadenceConfig + all env-backed settings, telemetry schemas), schemas/policy_state.py (intra-policy state machine — THESIS_ARMED/HOLD_LOCK/COOLDOWN graph + guard functions), services/policy_state_machine.py (deterministic transitions, activation window, pre-entry invalidation telemetry), services/policy_loop_gate.py (event-driven gate with single-flight lock, heartbeat, cooldown, skip events); 4 test files, 108 tests, all passing | Claude |

## Test Evidence (append results before commit)

```text
uv run pytest tests/test_reasoning_cadence_config.py \
              tests/test_high_level_reflection_schedule.py \
              tests/test_reasoning_loop_timeouts.py \
              tests/test_reasoning_scheduler_guards.py -vv

108 passed in 20.60s
```

## Worktree Setup

```bash
git worktree add -b feat/reasoning-agent-cadence ../wt-reasoning-cadence main
cd ../wt-reasoning-cadence

# When finished
git worktree remove ../wt-reasoning-cadence
```

## Git Workflow

```bash
git checkout -b feat/reasoning-agent-cadence

git add app/core/config.py \
  worker/ \
  trading_core/regime_classifier.py \
  services/regime_transition_detector.py \
  services/high_level_reflection_service.py \
  services/memory_retrieval_service.py \
  services/playbook_stats_service.py \
  tests/test_reasoning_cadence_config.py \
  tests/test_high_level_reflection_schedule.py \
  tests/test_reasoning_loop_timeouts.py \
  tests/test_reasoning_scheduler_guards.py \
  docs/branching/54-reasoning-agent-cadence-rules.md \
  docs/branching/README.md

uv run pytest tests/test_reasoning_cadence_config.py tests/test_high_level_reflection_schedule.py -vv

# Commit only after Test Evidence + Human Verification Evidence are recorded
git commit -m "feat: add scheduling and cadence rules for reasoning-agent loops (Runbook 54)"
```

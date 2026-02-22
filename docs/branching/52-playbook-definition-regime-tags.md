# Runbook 52: Playbook Definition with Regime Tags

## Purpose

Convert playbooks from loosely structured guidance into typed, parameterized strategy
definitions with explicit regime eligibility and expectation statistics. The strategist
must select a playbook first, then generate a concrete plan instance.

This runbook formalizes the "expert strategy + data-driven reasoning" blend by making
playbooks the structured interface between historical evidence and live plan generation.

## Why This Is a Leverage Point

Freeform proposal generation is hard to validate and hard to compare across regimes.
Typed playbooks give the system:

- a constrained proposal surface
- explicit invalidation conditions
- expectation distributions for reflection/judge calibration
- a stable object for evidence updates from Runbook 48 and memory stats from Runbook 51
- a policy-stability contract that prevents LLM-driven mid-trade strategy drift

## Scope

1. **`schemas/playbook_definition.py`** — new typed playbook contract
2. **`vector_store/playbooks/*.md`** — frontmatter/content normalization to typed fields
3. **`services/playbook_registry.py`** — load, validate, and serve playbook definitions
4. **`services/playbook_stats_service.py`** — attach historical stats by regime
5. **`agents/strategies/plan_provider.py`** — playbook-first plan generation flow
6. **`agents/strategies/llm_client.py`** — choose playbook + parameters, not freeform trigger invention
7. **`trading_core/trigger_engine.py` / compiler paths** — validate instantiated plan against playbook constraints
8. **`schemas/llm_strategist.py`** — add `playbook_id` / expectation references on plan output
9. **Tests** for playbook validation, eligibility filtering, and plan instantiation

## Out of Scope

- Removing all freeform strategies immediately (allow fallback while migrating)
- UI authoring tools for playbooks
- Auto-editing playbooks from judge suggestions (human review still required)
- New playbook research loops (Runbook 48 already covers evidence accumulation)

## Required Playbook Fields (Typed)

Each playbook definition must include:

1. **Identity**
   - `playbook_id`
   - `version`
   - `strategy_template` / `template_id` (if applicable)
   - `policy_class` (`trend_following`, `mean_reversion`, `breakout`, `volatility_expansion`, `volatility_compression`)

2. **Regime Eligibility**
   - eligible regimes (e.g., `bull`, `range`, `volatile`)
   - disallowed regimes / drift warnings
   - confidence thresholds
   - symbol/timeframe constraints (optional)

3. **Entry Rules (Two-Stage)**
   - **Thesis Qualification Layer (HTF / policy-boundary):**
     - structural setup conditions
     - regime eligibility confirmation
     - quality grades / optional filters
   - **Trigger Activation Layer (LTF / price-event, deterministic):**
     - activation triggers (e.g., break of HOD/range high, reclaim, sweep+reclaim)
     - activation timeout / expiry
     - activation refinement mode (`price_touch`, `close_confirmed`, `liquidity_sweep`, `next_bar_open`)

4. **Stop + Target Rules**
   - stop placement method(s)
   - target method(s)
   - fallback behavior if anchors unavailable
   - structural expectancy gate (`minimum_structural_r_multiple`, `require_structural_target`)

5. **Invalidation Conditions**
   - what invalidates the setup before entry
   - what invalidates the thesis after entry (risk-off / stand-down behavior)
   - deterministic pre-entry invalidation set for `THESIS_ARMED` (timeout, structure break, shock, regime-transition cancel)

6. **Time-Horizon Expectations**
   - expected hold bars / windows
   - setup maturation window
   - expiry / TTL guidance

7. **Historical Performance Statistics (by regime where possible)**
   - sample size (`n`)
   - win rate / expectancy / avg R
   - **P50 / P90 holding time**
   - hold-time mean/std (or robust equivalents) for calibration checks
   - **MAE / MFE distributions**
   - risk distribution (loss tail, stop hit frequency)
   - last-updated timestamp and evidence source

8. **Policy Stability Constraints**
   - minimum hold horizon before policy re-evaluation is allowed
   - policy mutation cooldown window
   - allowed mid-trade mutations (e.g., stop tightening only)
   - forbidden mid-trade mutations (e.g., switching playbook family)
   - same-policy-class mutation rules vs cross-policy-class mutation rules
   - exception conditions (structural invalidation, safety override)
   - state-transition rules for `THESIS_ARMED`, `POSITION_OPEN`, `HOLD_LOCK`, `INVALIDATED`, `COOLDOWN`

## Two-Stage Entry Contract (Required)

Playbooks must explicitly separate:

1. **Thesis qualification**
   - evaluated at policy boundary (typically HTF close)
   - determines whether a setup is worth arming
2. **Activation trigger**
   - evaluated by the deterministic tick engine during the activation window
   - determines whether/when entry actually occurs

This prevents "HTF breakout context valid" from being treated as "enter next bar blindly."

Illustrative schema shape:

```python
class EntryRuleSet(BaseModel):
    thesis_conditions: list[str]              # evaluated on HTF close / policy event
    activation_triggers: list[str]            # deterministic price-event triggers
    activation_timeout_bars: int | None = None   # measured in execution-timeframe bars
    activation_refinement_mode: Literal[
        "price_touch",
        "close_confirmed",
        "liquidity_sweep",
        "next_bar_open",
    ] = "price_touch"
```

### Activation Timeout Semantics (Required)

Activation timeouts must be unambiguous:

- **precedence**: playbook `activation_timeout_bars` overrides global default
  (`POLICY_THESIS_ACTIVATION_TIMEOUT_BARS`) when set
- **units**: timeout is measured in **execution timeframe bars** (LTF / trigger-engine TF),
  not policy/HTF bars
- **expiry behavior (default)**:
  - `THESIS_ARMED -> COOLDOWN` on activation timeout expiry
  - no immediate re-arm from tick noise; re-arm is only eligible on the next policy boundary
    (typically next HTF close / policy event), unless a documented safety override applies

Timeout expiry is a deterministic state transition, not a strategist re-evaluation event.

Required telemetry on activation expiry/cancel:

- `activation_expired_reason` (`timeout`, `structure_break`, `shock`, `regime_cancel`, `safety_cancel`)
- `armed_duration_bars` (execution-TF bars spent in `THESIS_ARMED`)

## Structural Expectancy Gate (Required Before Thesis Arm)

Playbooks must not force a nominal target like "always 2R" if structure does not support it.

Before arming a thesis:

1. compute structural stop candidate
2. compute structural target candidate
3. compute structural `R` multiple
4. arm thesis only if `R >= minimum_structural_r_multiple` (when configured)

Illustrative rule:

- `R = (target_price - entry_price) / (entry_price - stop_price)` (long example)

If the structural target does not provide sufficient expectancy, the correct outcome is
`no_trade`, not a fabricated target.

### Structural Target Candidate Selection Rule (Required)

The expectancy gate is only deterministic if the structural target candidate selection is
standardized. Each playbook must declare:

- allowed target-candidate sources (ordered or scored)
- target-candidate selection rule (priority or deterministic ranking)
- fallback behavior if no structural target candidate is valid

Examples of target-candidate sources:

- HTF structural level (`htf_daily_high`, `htf_5d_high`, etc.)
- measured move projection
- volatility expansion projection (deterministic formula)
- playbook-specific structural level

The selected target candidate must be recorded (e.g., `structural_target_source`) so the
expectancy gate decision is auditable and comparable across playbooks.

## Deterministic `THESIS_ARMED` Invalidation Set (Required)

Once a thesis is armed, the tick engine may cancel it deterministically without invoking
the strategist, using a playbook-defined pre-entry invalidation set.

Minimum supported invalidation categories:

- `activation_timeout_expired`
- `structure_break_before_entry` (e.g., failed reclaim / range structure invalidated)
- `volatility_shock_out_of_envelope`
- `regime_transition_cancel` (from Runbook 55 detector event)
- `safety_override_cancel`

These are **pre-entry invalidations** and should be modeled separately from post-entry
invalidation / exit behavior.

## Holding-Time Calibration Monitoring (Required)

Expected holding time is not just descriptive metadata; it must be calibrated against
realized outcomes.

At minimum, the playbook stats layer should compute and monitor:

- realized hold-time z-score (or robust z-score) relative to historical expectation
- quantile coverage (how often realized holds fall within expected P50/P90 bands)
- regime-specific calibration drift over time

Illustrative check:

- `z = (realized_hold - expected_mean_hold) / expected_hold_std`

Repeated large deviations (for example, sustained `|z| > 2` in a regime cluster) should
trigger high-level reflection review and may invalidate the playbook's time-horizon
assumptions.

If variance is unstable or heavy-tailed, use robust alternatives (median/MAD or quantile
coverage) and record the chosen method in playbook stats metadata.

## Strategist Flow Change (Mandatory)

Current behavior (too open-ended):

1. LLM reads context
2. LLM invents a plan
3. Compiler tries to salvage/validate it

Target behavior:

1. Filter eligible playbooks from `PolicySnapshot`
2. Select playbook (or `wait`)
3. Parameterize concrete plan from selected playbook
4. Validate instantiated plan against playbook constraints
5. Validate structural expectancy gate (minimum structural R) before arming thesis
6. Arm thesis (`THESIS_ARMED`) with activation window + deterministic activation triggers
7. Freeze policy envelope for the policy window / cooldown
8. Send to policy-level reflection / judge validation

This should reduce unsupported trigger combinations and make rejection reasons clearer.
It also prevents playbook switching churn while a trade is active unless a documented
invalidation/override condition is met.

## Implementation Steps

### Step 1: Define `PlaybookDefinition` schema

Create `schemas/playbook_definition.py` with typed submodels for:

- `RegimeEligibility`
- `EntryRuleSet`
- `RiskRuleSet` (stops/targets)
- `InvalidationRuleSet`
- `HorizonExpectations`
- `PlaybookPerformanceStats`
- `PolicyClass`
- `PolicyStabilityConstraints`

Prefer explicit fields over narrative prose. Narrative notes may remain as supplementary.

### Step 2: Normalize `vector_store/playbooks/*.md`

Add/standardize frontmatter fields and clearly delimited sections so docs remain human-
readable while being machine-validated.

At minimum, each playbook should expose machine-readable:

- `playbook_id`
- `policy_class`
- `regimes`
- `entry_identifiers`
- `thesis_conditions`
- `activation_triggers`
- `activation_refinement_mode`
- `invalidation_identifiers`
- `stop_methods`
- `target_methods`
- `minimum_structural_r_multiple`
- `horizon_expectations`
- `policy_stability_constraints`

### Step 3: Build registry + stats attachment

`services/playbook_registry.py` loads typed definitions.

`services/playbook_stats_service.py` attaches empirical stats from:

- Signal Ledger / outcome reconciler (Runbook 43)
- research validation evidence (Runbook 48)
- episode memory aggregates (Runbook 51)

This service must also compute expectation calibration metrics (hold-time calibration,
MAE/MFE drift) and expose them by regime for high-level reflection and judge use.

### Step 4: Enforce playbook-first plan generation

Update strategist flow so the LLM (or deterministic selector) chooses a `playbook_id`
first, then emits only the parameters/fields needed to instantiate the plan.

All plans should carry:

- `playbook_id`
- `playbook_version`
- expectation snapshot (hold P50/P90, MAE/MFE expectations used at decision time)
- expectation calibration snapshot (or reference) used for confidence calibration
- policy freeze window / mutation cooldown metadata used for judge enforcement
- activation window metadata (timeout, refinement mode, activation trigger set)
- pre-entry invalidation metadata (deterministic cancel conditions for `THESIS_ARMED`)
- lifecycle state metadata (e.g., initial state `THESIS_ARMED`)

### Step 5: Validate instantiated plans against playbook constraints

The compiler/validator must reject plans that violate selected playbook constraints:

- unsupported identifiers
- disallowed regime
- missing invalidation rules
- stop/target not aligned with playbook methods
- disallowed mid-trade policy mutation (unless invalidation/safety exception is present)
- thesis armed without passing structural expectancy gate when required
- activation trigger/refinement mode not allowed by playbook
- missing deterministic pre-entry invalidation set for `THESIS_ARMED` when playbook uses activation windows

### Step 6: Enforce policy stability constraints in orchestration

Orchestration / policy loop must honor playbook-defined stability rules:

- do not switch playbook families mid-trade inside cooldown window
- treat cross-`policy_class` mutations as stricter than same-class adjustments
- allow only explicitly permitted mutations (e.g., deterministic stop tightening)
- permit override only on structural invalidation or safety trigger with audit trail

Intra-policy execution state semantics (required):

- `THESIS_ARMED`: thesis approved, waiting for deterministic activation trigger
- deterministic pre-entry invalidation set active; timeout counted in execution-TF bars
- `POSITION_OPEN`: entry filled
- `HOLD_LOCK`: no strategist re-selection, no playbook switching, no target re-optimization
- `INVALIDATED`: thesis broken by invalidation/safety rule
- `COOLDOWN`: post-trade/post-transition suppression window

During `THESIS_ARMED`, only deterministic activation triggers may be evaluated. The
strategist must not re-evaluate thesis on every tick within the activation window.
Deterministic pre-entry invalidation checks (timeout/structure break/shock/regime cancel)
may transition the state out of `THESIS_ARMED` without a policy-loop call.

## Execution Refinement Contract (Runbooks 40/41/42 Interaction)

Micro-execution refinement is a trigger-engine responsibility, not a strategist function.

Playbooks/templates that rely on structural breakout timing (e.g., Runbooks 40/41/42
interactions) should use a deterministic activation refinement mode such as:

- `price_touch`
- `close_confirmed`
- `liquidity_sweep`
- `next_bar_open` (legacy fallback)

This keeps HTF thesis allocation separate from LTF execution timing.

### Refinement Mode -> Trigger Contract (Required)

Each `activation_refinement_mode` must map to:

- deterministic trigger identifier set (compiler-enforced)
- evaluation timeframe (execution TF or specified micro-TF)
- minimum confirmation rule

Examples:

- `price_touch` -> identifier: `break_level_touch`; timeframe: execution TF tick/bar; confirmation: touch/cross
- `close_confirmed` -> identifier: `break_level_close_confirmed`; timeframe: micro-TF close; confirmation: close beyond level
- `liquidity_sweep` -> identifiers: `sweep_low_reclaim` / `sweep_high_reject`; timeframe: micro-TF; confirmation: sweep + reclaim pattern
- `next_bar_open` -> identifier: `next_bar_open_entry`; timeframe: next execution-TF bar; confirmation: none

This prevents refinement modes from becoming ambiguous labels.

## Acceptance Criteria

- [ ] Typed `PlaybookDefinition` schema exists and validates required fields
- [ ] Playbooks include regime eligibility, entry rules, stop/target rules, invalidation rules
- [ ] Playbooks include time-horizon expectations and historical stats (including hold P50/P90, MAE/MFE)
- [ ] Playbook stats include holding-time calibration metrics (e.g., z-score/coverage) or explicit robust alternative
- [ ] Playbooks include machine-readable policy stability constraints (min hold / cooldown / allowed mutations / exceptions)
- [ ] Playbooks include `policy_class` and distinguish same-class vs cross-class mutation rules
- [ ] Playbooks define two-stage entry rules (`thesis_conditions` + `activation_triggers`) with activation timeout/refinement mode
- [ ] Activation timeout semantics are explicit (precedence, execution-TF bar units, expiry -> `COOLDOWN`, no immediate tick-based re-arm)
- [ ] Activation expiry/cancel telemetry fields are defined (`activation_expired_reason`, `armed_duration_bars`)
- [ ] Structural expectancy gate is enforced before `THESIS_ARMED` when `minimum_structural_r_multiple` is configured
- [ ] Structural target candidate selection rule is standardized and recorded (`structural_target_source`) for expectancy gating
- [ ] Strategist flow selects a playbook before plan instantiation (or outputs `wait`)
- [ ] Instantiated plan carries `playbook_id` + expectation metadata used at decision time
- [ ] Instantiated plan carries expectation calibration reference/snapshot for downstream reflection/judge checks
- [ ] Instantiated plan carries policy freeze/cooldown metadata for downstream enforcement
- [ ] Instantiated plan carries activation-window metadata and starts in `THESIS_ARMED` (when applicable)
- [ ] Instantiated plan carries deterministic pre-entry invalidation metadata for `THESIS_ARMED`
- [ ] Compiler/validator rejects plans that violate playbook constraints
- [ ] Orchestration rejects disallowed mid-trade playbook switching unless invalidation/safety exception is triggered
- [ ] During `HOLD_LOCK`, no playbook switching or target re-optimization is permitted (except invalidation/safety overrides)
- [ ] Each activation refinement mode maps to deterministic trigger identifiers, evaluation timeframe, and confirmation rule
- [ ] Stats are tagged by regime (or explicitly marked insufficient)
- [ ] Tests cover eligibility filtering and constraint enforcement

## Test Plan

```bash
# Playbook schema + registry
uv run pytest tests/test_playbook_definition_schema.py -vv
uv run pytest tests/test_playbook_registry.py -vv

# Stats attachment (holding distributions, MAE/MFE)
uv run pytest tests/test_playbook_stats_service.py -vv

# Calibration metrics / drift checks
uv run pytest -k "playbook_stats and calibration" -vv

# Strategist playbook-first generation flow
uv run pytest -k "playbook and plan_provider" -vv

# Constraint enforcement
uv run pytest -k "playbook_constraint" -vv

# Policy stability / mutation cooldown enforcement
uv run pytest -k "playbook and policy_stability" -vv

# Two-stage entry / activation window / structural expectancy gate
uv run pytest -k "playbook and activation_window" -vv
uv run pytest -k "playbook and structural_r_gate" -vv

# Activation timeout semantics + THESIS_ARMED invalidation set
uv run pytest -k "playbook and activation_timeout" -vv
uv run pytest -k "playbook and pre_entry_invalidation" -vv

# Structural target-candidate selection + refinement-mode trigger mapping
uv run pytest -k "playbook and structural_target_candidate" -vv
uv run pytest -k "playbook and refinement_mode_mapping" -vv
```

## Human Verification Evidence

```text
TODO:
1. Run a strategist cycle in a known compression setup and confirm the system selects
   `compression_breakout` (or other expected playbook) before plan instantiation.
2. Inspect the emitted plan and verify it includes playbook_id, version, and expectation
   metadata (hold P50/P90, MAE/MFE expectations), plus calibration reference/snapshot.
3. Inspect playbook stats for one regime and confirm hold-time calibration metric
   (z-score/coverage or robust equivalent) is present.
4. While a trade is open, attempt to switch from one playbook family to another inside the
   cooldown window and confirm it is blocked unless invalidation/safety exception applies.
5. Verify a thesis can be armed on HTF qualification but entry waits until the deterministic
   activation trigger (e.g., HOD/range-high break) fires.
6. Verify a setup with insufficient structural target distance (e.g., < configured minimum R)
   is rejected as `no_trade` rather than forcing a target.
7. Let an armed thesis expire without activation and confirm deterministic transition to
   `COOLDOWN` with no tick-triggered strategist re-evaluation.
8. Confirm activation expiry telemetry records `activation_expired_reason` and
   `armed_duration_bars`.
9. Confirm structural expectancy gate logs the selected `structural_target_source` used
   for the pass/fail decision.
10. Force a plan that violates the selected playbook's invalidation or identifier rules
   and confirm the validator rejects it with a specific reason.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-22 | Runbook created — typed playbook definition with regime tags and expectation stats | Codex |

## Test Evidence (append results before commit)

```text
TODO
```

## Worktree Setup

```bash
git worktree add -b feat/playbook-definition-regime-tags ../wt-playbook-definition main
cd ../wt-playbook-definition

# When finished
git worktree remove ../wt-playbook-definition
```

## Git Workflow

```bash
git checkout -b feat/playbook-definition-regime-tags

git add schemas/playbook_definition.py \
  services/playbook_registry.py \
  services/playbook_stats_service.py \
  vector_store/playbooks/ \
  agents/strategies/plan_provider.py \
  agents/strategies/llm_client.py \
  schemas/llm_strategist.py \
  trading_core/trigger_engine.py \
  tests/test_playbook_definition_schema.py \
  tests/test_playbook_registry.py \
  tests/test_playbook_stats_service.py \
  docs/branching/52-playbook-definition-regime-tags.md \
  docs/branching/README.md

uv run pytest tests/test_playbook_definition_schema.py tests/test_playbook_registry.py -vv

# Commit only after Test Evidence + Human Verification Evidence are recorded
git commit -m "feat: add typed playbook definitions with regime tags and expectation stats (Runbook 52)"
```

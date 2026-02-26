# Runbook 56: Structural Target and Activation Refinement Enforcement

## Purpose

Add a deterministic enforcement layer between the playbook schema (Runbook 52) and the
trigger engine / strategy primitives (Runbooks 40–42).

This runbook is not strategy logic. It is:

- compiler validation
- deterministic mapping enforcement
- telemetry contract hardening

It closes two remaining implementation risks:

1. structural target-candidate selection drifting across playbooks (breaking expectancy gating)
2. activation refinement modes being declared but not mapped to deterministic trigger primitives

## Position in Architecture

This runbook sits above strategy primitives and below policy/judge reasoning:

- **Inputs:** Runbook 52 playbook definitions, Runbooks 40–42 deterministic strategy/anchor primitives
- **Outputs:** compiler-validated plan instantiation artifacts consumed by trigger engine and judge

It should not be folded into Runbooks 40–42. Those runbooks define strategies and
anchors; this runbook defines the enforcement contract that binds them.

## Dependencies

- **Runbook 52** — Playbook schema (two-stage entry, structural expectancy gate, refinement modes)
- **Runbooks 40/41/42** — Deterministic strategy/HTF/anchor primitives
- **Runbook 54** — Activation window and HOLD_LOCK timing semantics
- **Runbook 53** — Judge boundary enforcement (consumes outputs / telemetry)
- **Runbook 55** — Regime transition detector may cancel `THESIS_ARMED` via deterministic invalidation

## Scope

1. **`schemas/playbook_definition.py`** — add/validate target candidate and refinement mapping fields
2. **`agents/strategies/trigger_compiler.py`** (or equivalent compiler path) — enforce contracts
3. **`trading_core/trigger_engine.py`** — consume canonical activation refinement primitives only
4. **`services/structural_target_selector.py`** (new) — deterministic candidate resolution and selection
5. **`schemas/llm_strategist.py`** / plan schema — carry canonical target/refinement telemetry fields
6. **Telemetry/events** — expectancy-gate and activation-refinement enforcement decisions
7. **Tests** — compiler rejects mismatch, target selection determinism, telemetry completeness

## Out of Scope

- Inventing new strategy templates
- LLM playbook selection or policy reasoning
- Tick-level microstructure heuristics beyond deterministic trigger definitions
- Regime detector design (Runbook 55)

## Part 1 — Structural Target Candidate Enforcement

## Required Playbook Contract

Each playbook that uses structural expectancy gating must define:

```python
class StructuralTargetPolicy(BaseModel):
    structural_target_candidates: list[str]
    target_selection_mode: Literal["priority", "ranked", "scored"]
    require_structural_target: bool = True
    minimum_structural_r_multiple: float | None = None
```

The candidate identifiers must map to deterministic primitives.

## Candidate Source -> Deterministic Origin Mapping (Examples)

| Candidate Source | Deterministic Origin |
|---|---|
| `htf_daily_high` | Runbook 41 (`htf_*` structure fields) |
| `htf_5d_high` | Runbook 41 |
| `measured_move` | Runbook 40 compression/range projection |
| `donchian_upper` | Runbook 40 Donchian breakout context |
| `fib_extension` | Runbook 38/indicator infrastructure |
| `range_projection` | Compression/range structure primitive |

This table must be implemented in code as a deterministic registry (not prompt text).

## Compiler Enforcement at Thesis Arming (Required)

At thesis arming time (policy boundary -> `THESIS_ARMED`):

1. Resolve structural stop candidate (Runbook 42 stop anchors)
2. Resolve **all declared** structural target candidates
3. Filter invalid candidates with explicit reason codes
4. Select target using declared `target_selection_mode`
5. Compute structural `R`
6. Arm thesis only if expectancy gate passes

Canonical long-form calculation (illustrative):

`R = (T - E) / (E - S)`

Where:

- `T` = selected structural target price
- `E` = intended entry price / activation reference
- `S` = structural stop price

If no valid target candidate remains:

- `no_trade` / do not arm thesis

No target fabrication allowed.

## Required Telemetry (Structural Expectancy Gate)

Plan instantiation / thesis-arming telemetry must record:

- `structural_target_source` (selected)
- `all_candidate_sources_evaluated`
- `selected_target_price`
- `structural_r`
- `candidate_rejections` (with reason codes)
- `target_selection_mode`
- `expectancy_gate_passed` (bool)

Without this, expectancy failures are not auditable.

## Part 2 — Activation Refinement Mode Compiler Enforcement

## Problem

Runbook 52 defines `activation_refinement_mode`, and Runbook 54 defines activation-window
behavior. This runbook makes that mapping enforceable so modes do not become vague labels.

## Required Refinement Mode -> Deterministic Trigger Mapping Table

The compiler must validate against a deterministic mapping table.

Example contract:

| `refinement_mode` | Required Trigger Identifiers | Eval TF | Confirmation Rule |
|---|---|---|---|
| `price_touch` | `level_break_intrabar` | execution TF | immediate touch/cross |
| `close_confirmed` | `level_break_close` | execution TF (or declared micro-TF) | bar close beyond level |
| `liquidity_sweep` | `sweep_then_reclaim` | execution TF / micro-TF | wick through + close reclaim/reject |
| `next_bar_open` | `next_bar_open_entry` | execution TF | next-bar open |

Notes:

- exact identifier names should match the trigger-engine namespace used in this repo
- if a mode requires a micro-TF, that timeframe must be explicit and compiler-validated

## Compiler Validation Rules (Required)

The compiler must reject plans when:

- refinement mode is declared but trigger identifiers are incompatible
- required confirmation primitive does not exist in trigger engine namespace
- evaluation timeframe is missing or incompatible with mode requirements
- rogue trigger identifiers are used outside allowed mapping for the mode

Mismatch is a **hard reject before execution**.

## Required Telemetry (Activation Refinement Enforcement)

For each armed thesis / activation configuration, log:

- `activation_refinement_mode`
- `activation_trigger_identifiers`
- `activation_eval_timeframe`
- `activation_confirmation_rule`
- `refinement_mapping_validated` (bool)
- `refinement_mapping_errors` (if any)

## Part 3 — Separation Guarantees (Restated)

This runbook reinforces the architectural separation already established:

- refinement modes are purely deterministic
- no LLM calls inside the activation window
- no strategy reinterpretation during activation refinement
- no target re-optimization during `HOLD_LOCK` (except explicitly allowed boundary exceptions)

If implementation of Runbook 56 introduces LLM dependency in micro timing, the layer
boundary has failed.

## Implementation Steps

### Step 1: Extend playbook schema with enforceable target/refinement policy fields

Add typed fields (or typed submodels) for:

- structural target candidates
- target selection mode
- refinement mode mapping requirements
- activation evaluation timeframe / confirmation metadata (where applicable)

### Step 2: Implement deterministic target-candidate registry

Create a code registry that maps candidate source identifiers to deterministic resolvers
backed by Runbooks 40–42 primitives.

The registry must be versioned and testable.

### Step 3: Add structural target selector service

Implement `services/structural_target_selector.py` (or equivalent) to:

- resolve candidates
- filter invalids with reasons
- select target deterministically
- compute structural R
- emit telemetry payload

### Step 4: Add compiler enforcement for refinement mode mapping

Compiler validates refinement mode against:

- trigger identifiers
- evaluation timeframe
- confirmation primitive availability

Reject invalid combinations before `THESIS_ARMED`.

### Step 5: Thread canonical outputs into plan instantiation / telemetry

Ensure instantiated plans (or arming artifacts) carry canonical fields:

- structural target selection outputs
- activation refinement mapping outputs

### Step 6: Trigger-engine integration sanity checks

Ensure trigger engine executes only canonical refinement primitives emitted by the compiler.
No free-form refinement interpretation at runtime.

## Acceptance Criteria

- [ ] Playbook schema supports enforceable structural target-candidate declarations and target selection mode
- [ ] Candidate source identifiers map to deterministic strategy/anchor primitives (Runbooks 40–42) via code registry
- [ ] At thesis arming, compiler resolves all declared structural target candidates and logs rejections with reason codes
- [ ] Structural target is selected deterministically and `structural_r` is computed before expectancy gate decision
- [ ] If no valid structural target candidate exists, thesis is not armed (`no_trade`)
- [ ] Expectancy-gate telemetry includes `structural_target_source`, evaluated candidates, selected target, `structural_r`, and rejection reasons
- [ ] Refinement mode -> trigger identifier / timeframe / confirmation mapping is compiler-enforced
- [ ] Compiler hard-rejects incompatible refinement mode / trigger identifier combinations
- [ ] Trigger engine consumes only canonical deterministic refinement primitives from compiler output
- [ ] No LLM dependency is introduced in activation-window timing or structural target selection

## Test Plan

```bash
# Playbook schema extensions (target/refinement policy)
uv run pytest tests/test_playbook_definition_schema.py -k "structural_target or refinement_mode" -vv

# Structural target candidate registry + deterministic selection
uv run pytest tests/test_structural_target_selector.py -vv

# Expectancy gate telemetry completeness
uv run pytest -k "structural_target and telemetry" -vv

# Refinement mode compiler enforcement (identifier/timeframe/confirmation mapping)
uv run pytest tests/test_trigger_compiler.py -k "refinement_mode" -vv

# Compiler rejects rogue identifiers / invalid mappings
uv run pytest -k "trigger_compiler and refinement_mapping" -vv

# Trigger-engine consumes canonical refinement primitives only
uv run pytest -k "trigger_engine and canonical_refinement" -vv
```

## Human Verification Evidence

```text
1. CANDIDATE_SOURCE_REGISTRY inspected: 14 entries covering Runbook 40 (donchian/measured_move),
   Runbook 41 (htf_daily_high/low, htf_prev, htf_5d), Runbook 38 (fib_extension), and
   Runbook 42 (r_multiple_2/3). Each entry has indicator_field, origin, and direction keys.

2. evaluate_expectancy_gate() verified:
   - test_gate_passes_long_simple: entry=100, stop=95, target(htf_daily_high)=115 → R=3.0, gate passed
   - test_gate_rejects_insufficient_r_multiple: R=0.2 < minimum 1.5 → rejected with typed reason code
   - test_gate_zero_stop_distance_rejects_all: stop==entry → all candidates rejected (degenerate check)

3. close_confirmed refinement mode:
   - test_enforce_refinement_mode_mapping_close_confirmed_passes: entry_rule contains
     "break_level_close_confirmed" → no violations, plan.refinement_mapping_validated=True
   - test_enforce_refinement_mode_mapping_close_confirmed_violation: plain "close > sma_medium"
     entry_rule → violation with missing_identifiers=["break_level_close_confirmed"]

4. Mismatched refinement mode confirmed:
   - test_enforce_refinement_mode_mapping_price_touch_violation: "price_touch" mode with
     "close > sma_short" trigger → RefinementModeViolation emitted, plan.refinement_mapping_validated=False

5. No LLM calls confirmed by inspection: services/structural_target_selector.py and the new
   enforce_refinement_mode_mapping() function are pure, deterministic, no I/O.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-22 | Runbook created — deterministic enforcement layer for structural targets and activation refinement mapping | Codex |
| 2026-02-26 | Implemented: CANDIDATE_SOURCE_REGISTRY (14 entries), evaluate_expectancy_gate, ExpectancyGateTelemetry, StructuralCandidateRejection in services/structural_target_selector.py; REFINEMENT_MODE_COMPILER_TABLE, _REFINEMENT_ACTIVATION_PRIMITIVES, RefinementModeViolation, enforce_refinement_mode_mapping, target_selection_mode on RiskRuleSet in trading_core/trigger_compiler.py; 5 R56 telemetry fields on StrategyPlan in schemas/llm_strategist.py; 30 tests in test_structural_target_selector.py; 14 new tests in test_trigger_compiler.py | Claude |

## Test Evidence (append results before commit)

```text
$ uv run pytest tests/test_structural_target_selector.py -vv
30 passed in 15.10s

$ uv run pytest tests/test_trigger_compiler.py -vv
106 passed in 12.99s

$ uv run pytest (full suite, excluding pre-existing DB_DSN import errors):
2 failed (pre-existing test_factor_loader), 1503 passed, 2 skipped

R56 specific test plan:
  uv run pytest tests/test_structural_target_selector.py -vv                    → 30 passed
  uv run pytest tests/test_trigger_compiler.py -k "refinement_mode" -vv        → 11 passed
  uv run pytest tests/test_trigger_compiler.py -k "refinement_mapping" -vv     → 3 passed
```

## Worktree Setup

```bash
git worktree add -b feat/structural-target-refinement-enforcement ../wt-structural-target-refinement main
cd ../wt-structural-target-refinement

# When finished
git worktree remove ../wt-structural-target-refinement
```

## Git Workflow

```bash
git checkout -b feat/structural-target-refinement-enforcement

git add schemas/playbook_definition.py \
  agents/strategies/trigger_compiler.py \
  trading_core/trigger_engine.py \
  services/structural_target_selector.py \
  schemas/llm_strategist.py \
  tests/test_structural_target_selector.py \
  tests/test_trigger_compiler.py \
  docs/branching/56-structural-target-activation-refinement-enforcement.md \
  docs/branching/52-playbook-definition-regime-tags.md \
  docs/branching/54-reasoning-agent-cadence-rules.md \
  docs/branching/README.md

uv run pytest tests/test_structural_target_selector.py tests/test_trigger_compiler.py -vv

# Commit only after Test Evidence + Human Verification Evidence are recorded
git commit -m "feat: enforce structural target selection and activation refinement mappings (Runbook 56)"
```

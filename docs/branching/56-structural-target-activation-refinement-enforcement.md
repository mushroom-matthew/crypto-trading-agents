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
TODO:
1. Inspect a breakout playbook with structural expectancy gating and confirm target
   candidates are resolved deterministically and `structural_target_source` is logged.
2. Verify a setup with no valid structural target candidate is rejected before `THESIS_ARMED`.
3. Configure `close_confirmed` refinement mode and confirm compiler requires the correct
   trigger identifier + timeframe + confirmation rule.
4. Attempt a mismatched refinement mode/identifier combo (e.g., `liquidity_sweep` with
   plain `price_touch` trigger) and confirm hard reject before execution.
5. Confirm no LLM calls occur during activation refinement in runtime traces.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-22 | Runbook created — deterministic enforcement layer for structural targets and activation refinement mapping | Codex |

## Test Evidence (append results before commit)

```text
TODO
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

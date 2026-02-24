# Runbook 47: Hard Template Binding

## Purpose

Runbook 46 routes the LLM to the right prompt template based on the indicator snapshot,
but the LLM can still ignore the template and generate arbitrary triggers. This runbook
closes that gap: the LLM is required to **declare** which named template it is using,
and the trigger compiler **enforces** that all trigger identifiers are members of that
template's allowed identifier set. Triggers using identifiers outside the template's
set are blocked at compile time, not silently accepted.

This shifts the instrument-level role from "author of trigger rules" to "selector and
parameterizer of a known template." The user's strategic control moves to: (a) which
templates exist in the vector store, and (b) how the sniffer is tuned to surface
instruments that match specific template contexts (see Runbook 39 amendment).

### Dependency

Depends on Runbook 46 (template-matched retrieval) being validated via paper trading
first. Hard binding is only useful once the retrieval routing table is known to be
accurate — if retrieval routes to the wrong template, binding will produce silently
wrong plans rather than gracefully degraded ones.

## Scope

1. **`schemas/llm_strategist.py`** — add `template_id: str | None` field to
   `StrategyPlan` and `template_parameters: dict | None` for threshold overrides
2. **`vector_store/retriever.py`** — expose `allowed_identifiers(template_id)` helper
   that returns the identifier set from the template's vector store doc frontmatter
3. **`trading_core/trigger_compiler.py`** — add template enforcement pass: strip
   triggers whose identifiers are not in the declared template's allowed set;
   add `block_reason="template_identifier_violation"` to blocked trigger log
4. **`agents/strategies/llm_client.py`** — update `_build_system_prompt()` to inject
   the named template list in the instruction; update `_extract_plan_json()` to
   tolerate `template_id` field; add `_sanitize_plan_dict()` pass for template_id
5. **`prompts/strategy_plan_schema.txt`** — add `template_id` field documentation and
   the list of valid template IDs
6. **`tests/test_hard_template_binding.py`** — new test file

## Out of Scope

- Forcing the LLM to use ONLY threshold parameters (fully constrained output) — the LLM
  may still write entry_rule / exit_rule strings; the compiler enforces identifiers
- Backtest runner changes (the StrategyPlan schema is backwards compatible because the
  field is Optional — old plans without template_id still validate)
- UI changes (template_id visible in event payload; no dedicated UI needed)
- Per-instrument workflow (see `_per-instrument-workflow.md`)

## Key Files

- `schemas/llm_strategist.py` (modify: add `template_id`, `template_parameters`)
- `vector_store/retriever.py` (modify: expose `allowed_identifiers()`)
- `trading_core/trigger_compiler.py` (modify: template enforcement pass)
- `agents/strategies/llm_client.py` (modify: prompt + sanitize)
- `prompts/strategy_plan_schema.txt` (modify: document template_id)
- `tests/test_hard_template_binding.py` (new)

## Implementation Steps

### Step 1: Schema changes in `schemas/llm_strategist.py`

In `StrategyPlan`, add after `regime`:

```python
template_id: str | None = Field(
    default=None,
    description=(
        "Named strategy template selected by the LLM from the provided candidate list. "
        "Must match a template ID from the vector store (e.g., 'compression_breakout', "
        "'bull_trending'). Null if no template applies."
    ),
)
template_parameters: dict | None = Field(
    default=None,
    description=(
        "Optional threshold overrides for the selected template. "
        "Key: parameter name (e.g., 'entry_vol_multiple_min'). "
        "Value: float. Only parameter names documented in the template are valid."
    ),
)
```

### Step 2: `allowed_identifiers(template_id)` in `vector_store/retriever.py`

```python
def allowed_identifiers(template_id: str) -> set[str]:
    """Return the allowed indicator identifiers for a named template."""
    store = get_strategy_vector_store()
    for doc in store.documents:
        if doc.doc_id == template_id or doc.template_file == template_id:
            return set(doc.identifiers)
    return set()
```

### Step 3: Template enforcement in `trading_core/trigger_compiler.py`

Add after existing enforcement passes (exit_binding, hold rule rejection):

```python
def enforce_template_identifiers(
    triggers: list[CompiledTrigger],
    template_id: str | None,
) -> tuple[list[CompiledTrigger], list[dict]]:
    """
    Block triggers that use identifiers not in the declared template's allowed set.
    Emergency exits are always exempt (same rule as other enforcement passes).
    """
    if not template_id:
        return triggers, []

    allowed = vector_store_allowed_identifiers(template_id)
    if not allowed:
        # Template not found — log warning, don't block (fail open)
        logger.warning("Template '%s' not found in vector store; skipping enforcement", template_id)
        return triggers, []

    kept: list[CompiledTrigger] = []
    blocked: list[dict] = []
    for trigger in triggers:
        if trigger.category == "emergency_exit":
            kept.append(trigger)
            continue
        violations = _find_identifier_violations(trigger, allowed)
        if violations:
            blocked.append({
                "trigger_id": trigger.trigger_id,
                "block_reason": "template_identifier_violation",
                "violations": violations,
                "template_id": template_id,
            })
        else:
            kept.append(trigger)
    return kept, blocked
```

`_find_identifier_violations()` parses `entry_rule`, `exit_rule`, `hold_rule` strings
using the existing identifier extraction logic (already present for autocorrect) and
returns identifiers not in `allowed`.

### Step 4: Update prompt in `agents/strategies/llm_client.py`

In `_build_system_prompt()`, when a `strategy_context` is present that includes a
`template_id` candidate, inject the instruction:

```
TEMPLATE SELECTION:
Select one of the following templates and declare it as `template_id` in your JSON output:
  - compression_breakout  (use when compression_flag=1 and bb_bandwidth_pct_rank < 0.25)
  - bull_trending         (use when trend_state=uptrend and macd_hist > 0)
  - range_mean_revert     (use when trend_state=sideways and vol_state=low/normal)
  - volatile_breakout     (use when vol_state=high or extreme)
  - uncertain_wait        (use when regime is unclear or conflicting signals)
  - null                  (if no template fits, omit template_id)

Your triggers must use only identifiers listed in the selected template. The trigger
compiler will reject any triggers using identifiers outside the template's allowed set.
```

The template list in the instruction is auto-generated from the vector store's loaded
documents (not hardcoded), so adding a new vector store doc automatically extends the list.

### Step 5: Update `prompts/strategy_plan_schema.txt`

Add to the `StrategyPlan` fields table:

```
template_id          Optional[str]  Named template from the vector store candidate list.
                                    Null if no template applies to current conditions.
template_parameters  Optional[dict] Threshold overrides for the selected template.
                                    Example: {"entry_vol_multiple_min": 1.5}
```

### Step 6: Tests in `tests/test_hard_template_binding.py`

```python
def test_template_enforcement_blocks_invalid_identifiers():
    """Triggers using identifiers outside compression_breakout's allowed set are blocked."""
    ...

def test_emergency_exit_exempt_from_template_enforcement():
    """emergency_exit category triggers pass enforcement regardless of identifiers."""
    ...

def test_null_template_id_skips_enforcement():
    """When template_id is None, enforcement pass is skipped entirely."""
    ...

def test_unknown_template_id_fails_open():
    """If template_id points to unknown template, enforcement logs warning and passes."""
    ...

def test_strategy_plan_validates_with_template_id():
    """StrategyPlan with template_id='compression_breakout' validates without error."""
    ...

def test_strategy_plan_validates_without_template_id():
    """Existing plans without template_id still validate (backwards compatible)."""
    ...

def test_allowed_identifiers_returns_template_set():
    """allowed_identifiers('compression_breakout') returns compression indicator names."""
    ...
```

## Schema Backwards Compatibility

`template_id` and `template_parameters` are both `Optional` with `default=None`. All
existing cached plans and test fixtures continue to validate — no fixture updates needed.
The trigger compiler enforcement is skipped when `template_id is None`.

## Environment Variables

```
# None new. Template enforcement is always active when template_id is present in the plan.
# To disable enforcement without schema changes, set TEMPLATE_ENFORCEMENT_ENABLED=false.
TEMPLATE_ENFORCEMENT_ENABLED=true  # Default true (new)
```

## Test Plan

```bash
# Unit: template enforcement
uv run pytest tests/test_hard_template_binding.py -vv

# Regression: existing trigger compiler tests
uv run pytest tests/test_trigger_compiler.py -vv

# Schema: backwards compatibility
uv run pytest -k "strategy_plan" -vv

# Full suite
uv run pytest -x -q
```

## Test Evidence

```
uv run pytest tests/test_hard_template_binding.py -vv
============================= test session starts ==============================
collected 9 items

tests/test_hard_template_binding.py::test_template_enforcement_blocks_invalid_identifiers PASSED
tests/test_hard_template_binding.py::test_emergency_exit_exempt_from_template_enforcement PASSED
tests/test_hard_template_binding.py::test_null_template_id_skips_enforcement PASSED
tests/test_hard_template_binding.py::test_unknown_template_id_fails_open PASSED
tests/test_hard_template_binding.py::test_strategy_plan_validates_with_template_id PASSED
tests/test_hard_template_binding.py::test_strategy_plan_validates_without_template_id PASSED
tests/test_hard_template_binding.py::test_allowed_identifiers_returns_template_set PASSED
tests/test_hard_template_binding.py::test_enforce_plan_quality_includes_template_violations PASSED
tests/test_hard_template_binding.py::test_enforcement_disabled_observes_but_does_not_remove PASSED

============================== 9 passed in 3.96s ===============================

Full suite (excluding DB_DSN collection errors which are pre-existing):
  2 failed (test_factor_loader.py — pre-existing pandas freq="H" deprecation, unrelated)
  926 passed, 2 skipped
```

## Acceptance Criteria

- [x] `StrategyPlan.template_id: str | None` added; all existing tests still pass
- [x] `enforce_template_identifiers()` in trigger_compiler blocks non-template triggers
- [x] Emergency exits exempt from template enforcement
- [x] `template_id=None` → enforcement skipped (no regression)
- [x] Unknown template_id → warning logged, enforcement skipped (fail open)
- [x] LLM prompt includes named template list with selection instructions
- [x] `allowed_identifiers(template_id)` returns correct set from vector store doc
- [x] All 9 unit tests pass (2 extra: enforce_plan_quality integration + enforcement-disabled mode)
- [x] Full test suite clean (926 passed, 2 pre-existing pandas freq="H" failures unrelated)

## Human Verification Evidence

```
Implementation complete. Manual paper trading validation deferred — gate from R46 still
applies (R46 routing must reach ≥80% accuracy before R47 hard enforcement is trusted).

To monitor gate progress during ongoing paper trading, use:
  GET http://localhost:8081/analytics/template-routing

This endpoint reports:
  - retrieved_template_pct: how often retrieval surfaces a template (R46 gate)
  - declared_template_pct: how often LLM declares template_id in plan (R47 compliance)
  - gate_r46_note: shows GATE MET ✓ once ≥80% threshold is reached
  - gate_r47_note: shows GATE MET ✓ once LLM compliance reaches ≥80%

When R46 gate is met, consider enabling TEMPLATE_ENFORCEMENT_ENABLED=true in production.
Until then, enforcement runs in "observe" mode (violations logged but triggers not stripped)
by setting TEMPLATE_ENFORCEMENT_ENABLED=false in .env.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-21 | Runbook created — hard template binding via trigger compiler enforcement | Claude |
| 2026-02-24 | Implementation: `schemas/llm_strategist.py` (template_id, template_parameters fields), `vector_store/retriever.py` (allowed_identifiers_for_template), `trading_core/trigger_compiler.py` (TemplateViolation, enforce_template_identifiers, wired into enforce_plan_quality), `agents/strategies/llm_client.py` (_build_template_selection_block prompt injection), `prompts/strategy_plan_schema.txt` (template_id docs), `ops_api/routers/analytics.py` (new: /analytics/template-routing gate endpoint), `ops_api/app.py` (analytics router registration), `tests/test_hard_template_binding.py` (9 tests), `vector_store/strategies/compression_breakout.md` (single-line identifiers fix) | Claude |

## Worktree Setup

```bash
git worktree add -b feat/hard-template-binding ../wt-hard-template main

# Depends on: feat/template-matched-retrieval (Runbook 46) validated in paper trading first
```

## Git Workflow

```bash
git checkout -b feat/hard-template-binding

git add schemas/llm_strategist.py \
  vector_store/retriever.py \
  trading_core/trigger_compiler.py \
  agents/strategies/llm_client.py \
  prompts/strategy_plan_schema.txt \
  tests/test_hard_template_binding.py

uv run pytest tests/test_hard_template_binding.py tests/test_trigger_compiler.py -vv
git commit -m "feat: hard template binding via trigger compiler enforcement (Runbook 47)"
```

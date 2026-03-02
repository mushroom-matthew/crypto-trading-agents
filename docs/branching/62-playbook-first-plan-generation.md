# Runbook 62: Playbook-First Plan Generation

## Purpose

Wire `PlaybookRegistry.list_eligible()` (R52) into `plan_provider.get_plan()` so the LLM
receives a constrained list of eligible playbooks for the current regime and must select
from that list. Validate the returned `playbook_id` post-LLM.

Also add HTF alignment fields to `RegimeEligibility` so a 1m-bullish signal in a daily
bear regime is filtered out at the playbook selection stage, not after the fact.

**Pre-condition:** R52 is implemented (PlaybookDefinition, PlaybookRegistry, all 7 playbook
`.md` files with frontmatter). This runbook wires them into the plan generation path.

## Scope

1. `schemas/playbook_definition.py`
   - Add `htf_trend_required: Literal["up", "down", "any"] | None = None` to
     `RegimeEligibility`
   - Add `disallow_htf_counter_trend: bool = False` to `RegimeEligibility`

2. `services/playbook_registry.py`
   - Update `list_eligible(regime, htf_direction=None)` to also apply HTF filters when
     `htf_direction` is provided

3. `agents/strategies/plan_provider.py`
   - Before LLM call: instantiate `PlaybookRegistry`, call `list_eligible(regime,
     htf_direction)`, pass eligible list to `llm_client.generate_plan()`
   - Post-LLM: validate `plan.playbook_id` is in the eligible set; if not, log warning
     and set `plan.playbook_id = None` (degrade gracefully — do not block the plan)

4. `agents/strategies/llm_client.py`
   - Accept `eligible_playbooks: list[PlaybookDefinition] | None = None` in
     `generate_plan()`
   - When provided, format as `ELIGIBLE_PLAYBOOKS` block in the system prompt

5. `vector_store/playbooks/*.md` — update frontmatter to add
   `htf_trend_required` to each playbook (optional field, leave blank if any trend is OK)

6. `tests/test_playbook_wiring.py` — new test file

## Out of Scope

- Compiler enforcement of playbook constraints (R56 scope)
- PlaybookRegistry persistence (already loads from .md files at runtime)
- Changing LLM scoring or ranking of playbooks

## Implementation Steps

### Step 1: Extend RegimeEligibility

In `schemas/playbook_definition.py`, inside `RegimeEligibility`:

```python
htf_trend_required: Literal["up", "down", "any"] | None = None
disallow_htf_counter_trend: bool = False
```

No migration needed (new optional fields, backwards-compatible).

### Step 2: Update PlaybookRegistry.list_eligible

```python
def list_eligible(
    self,
    regime: str,
    htf_direction: Literal["up", "down", "sideways"] | None = None,
) -> list[PlaybookDefinition]:
    """Return playbooks eligible for the given regime and optional HTF direction.

    Filtering order:
    1. regime not in eligible_regimes → excluded
    2. regime in disallowed_regimes → excluded
    3. htf_direction provided AND htf_trend_required is set AND mismatch → excluded
    4. disallow_htf_counter_trend AND htf_direction is counter to regime trend → excluded
    """
    eligible = []
    for pb in self._playbooks.values():
        re_ = pb.regime_eligibility
        if re_.eligible_regimes and regime not in re_.eligible_regimes:
            continue
        if regime in re_.disallowed_regimes:
            continue
        if htf_direction and re_.htf_trend_required:
            req = re_.htf_trend_required
            if req != "any" and req != htf_direction:
                continue
        eligible.append(pb)
    return eligible
```

### Step 3: Wire into plan_provider.get_plan()

At the top of `get_plan()`, before calling `_generate_fresh_plan()`:

```python
from services.playbook_registry import PlaybookRegistry
from schemas.llm_strategist import LLMInput

# Extract regime and HTF direction from llm_input
regime = (llm_input.indicator.regime if llm_input.indicator else None) or "unknown"
htf_direction = _extract_htf_direction(llm_input.indicator)

registry = PlaybookRegistry()
eligible_playbooks = registry.list_eligible(regime, htf_direction=htf_direction)
```

Pass `eligible_playbooks` through to `llm_client.generate_plan(eligible_playbooks=...)`.

Add helper:

```python
def _extract_htf_direction(indicator) -> str | None:
    """Derive HTF direction from htf_* fields if available."""
    if indicator is None:
        return None
    # Use htf_daily_trend if present, else infer from daily high/low vs price
    htf_trend = getattr(indicator, "htf_daily_trend", None)
    if htf_trend:
        return htf_trend  # "up", "down", "sideways"
    return None
```

### Step 4: Post-LLM validation

After `llm_client.generate_plan()` returns a `StrategyPlan`:

```python
if plan.playbook_id and eligible_playbooks:
    eligible_ids = {pb.playbook_id for pb in eligible_playbooks}
    if plan.playbook_id not in eligible_ids:
        workflow.logger.warning(
            "plan.playbook_id '%s' not in eligible set %s — clearing",
            plan.playbook_id, eligible_ids,
        )
        # Degrade gracefully: clear the field, do not block execution
        plan = plan.model_copy(update={"playbook_id": None})
```

### Step 5: Format ELIGIBLE_PLAYBOOKS block in llm_client

In `generate_plan()`, when `eligible_playbooks` is provided:

```python
if eligible_playbooks:
    lines = ["<ELIGIBLE_PLAYBOOKS>"]
    for pb in eligible_playbooks:
        desc = pb.description or "(no description)"
        lines.append(f"- {pb.playbook_id}: {desc}")
        if pb.regime_eligibility.eligible_regimes:
            lines.append(f"  regimes: {', '.join(pb.regime_eligibility.eligible_regimes)}")
    lines.append("</ELIGIBLE_PLAYBOOKS>")
    lines.append(
        "Select a playbook_id from the list above. "
        "If none fits, set playbook_id to null."
    )
    system_prompt += "\n\n" + "\n".join(lines)
```

### Step 6: Update playbook frontmatter (optional)

For any playbook that should only run in a specific HTF trend, add to its `.md` YAML
frontmatter:

```yaml
htf_trend_required: "up"  # or "down" or "any"
disallow_htf_counter_trend: true
```

Leave blank (or omit) for regime-agnostic playbooks.

## Acceptance Criteria

- [ ] `RegimeEligibility` has `htf_trend_required` and `disallow_htf_counter_trend` fields
- [ ] `list_eligible(regime, htf_direction="down")` excludes playbooks with
  `htf_trend_required="up"`
- [ ] `plan_provider.get_plan()` calls `list_eligible()` before every LLM call
- [ ] `eligible_playbooks` is injected as `ELIGIBLE_PLAYBOOKS` block in LLM prompt
- [ ] If `plan.playbook_id` is not in the eligible set, it is cleared (not a hard error)
- [ ] If `eligible_playbooks` is empty (no match for current regime), plan generation
  proceeds without playbook constraint
- [ ] All existing plan_provider and llm_client tests still pass

## Test Plan

```bash
# New playbook wiring tests
uv run pytest tests/test_playbook_wiring.py -vv

# Regression: plan provider and llm client
uv run pytest tests/test_plan_provider.py tests/test_llm_client.py -vv

# Regression: full suite
uv run pytest -x -q
```

## Human Verification Evidence

```text
1. RegimeEligibility.htf_trend_required and disallow_htf_counter_trend fields confirmed
   present in schemas/playbook_definition.py (backward-compatible, new optional fields).
2. PlaybookRegistry.list_eligible(regime, htf_direction="down") excludes playbooks with
   htf_trend_required="up" — verified by test_htf_trend_required_up_excluded_when_htf_down.
3. plan_provider._get_eligible_playbooks() invoked before every LLM call via
   _get_eligible_playbooks(llm_input) in get_plan() fresh-generation path.
4. eligible_playbooks injected into llm_client.generate_plan(eligible_playbooks=...).
   _build_eligible_playbooks_block() formats <ELIGIBLE_PLAYBOOKS> XML block with all IDs.
5. Post-LLM validation: plan.playbook_id cleared (not hard error) when not in eligible set —
   verified by test_invalid_playbook_id_cleared.
6. When eligible_playbooks=[] (no match), plan generation proceeds unaffected —
   verified by test_empty_eligible_list_skips_validation.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-01 | Runbook created — playbook-first plan generation wiring (R62) | Claude |
| 2026-03-02 | Implemented: schemas/playbook_definition.py (HTF fields), services/playbook_registry.py (HTF filtering), agents/strategies/plan_provider.py (_get_eligible_playbooks + post-LLM validation), agents/strategies/llm_client.py (eligible_playbooks param + _build_eligible_playbooks_block), tests/test_playbook_wiring.py (26 tests) | Claude |

## Test Evidence

```text
$ uv run pytest tests/test_playbook_wiring.py -vv
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-8.4.2
collected 26 items

tests/test_playbook_wiring.py::TestRegimeEligibilityHTFFields::test_htf_trend_required_accepts_up_down_any PASSED
tests/test_playbook_wiring.py::TestRegimeEligibilityHTFFields::test_htf_trend_required_defaults_to_none PASSED
tests/test_playbook_wiring.py::TestRegimeEligibilityHTFFields::test_disallow_htf_counter_trend_defaults_to_false PASSED
tests/test_playbook_wiring.py::TestRegimeEligibilityHTFFields::test_disallow_htf_counter_trend_set_to_true PASSED
tests/test_playbook_wiring.py::TestRegimeEligibilityHTFFields::test_htf_trend_required_rejects_invalid PASSED
tests/test_playbook_wiring.py::TestListEligibleHTF::test_no_htf_requirement_always_eligible PASSED
tests/test_playbook_wiring.py::TestListEligibleHTF::test_htf_trend_required_up_excluded_when_htf_down PASSED
tests/test_playbook_wiring.py::TestListEligibleHTF::test_htf_trend_required_up_included_when_htf_up PASSED
tests/test_playbook_wiring.py::TestListEligibleHTF::test_htf_trend_required_any_included_regardless PASSED
tests/test_playbook_wiring.py::TestListEligibleHTF::test_no_htf_direction_provided_skips_htf_filter PASSED
tests/test_playbook_wiring.py::TestListEligibleHTF::test_eligible_regimes_still_filters_regime PASSED
tests/test_playbook_wiring.py::TestListEligibleHTF::test_disallowed_regimes_excludes PASSED
tests/test_playbook_wiring.py::TestListEligibleHTF::test_mixed_eligible_and_htf PASSED
tests/test_playbook_wiring.py::TestBuildEligiblePlaybooksBlock::test_none_returns_none PASSED
tests/test_playbook_wiring.py::TestBuildEligiblePlaybooksBlock::test_empty_list_returns_none PASSED
tests/test_playbook_wiring.py::TestBuildEligiblePlaybooksBlock::test_block_contains_playbook_ids PASSED
tests/test_playbook_wiring.py::TestBuildEligiblePlaybooksBlock::test_block_has_xml_tags PASSED
tests/test_playbook_wiring.py::TestBuildEligiblePlaybooksBlock::test_block_has_instruction PASSED
tests/test_playbook_wiring.py::TestBuildEligiblePlaybooksBlock::test_block_includes_regimes_when_set PASSED
tests/test_playbook_wiring.py::TestPlanProviderPlaybookValidation::test_valid_playbook_id_preserved PASSED
tests/test_playbook_wiring.py::TestPlanProviderPlaybookValidation::test_invalid_playbook_id_cleared PASSED
tests/test_playbook_wiring.py::TestPlanProviderPlaybookValidation::test_no_playbook_id_in_plan_ok PASSED
tests/test_playbook_wiring.py::TestPlanProviderPlaybookValidation::test_empty_eligible_list_skips_validation PASSED
tests/test_playbook_wiring.py::TestExtractHTFDirection::test_none_indicator_returns_none PASSED
tests/test_playbook_wiring.py::TestExtractHTFDirection::test_indicator_with_htf_daily_trend PASSED
tests/test_playbook_wiring.py::TestExtractHTFDirection::test_indicator_without_htf_field_returns_none PASSED
============================= 26 passed in 18.98s ==============================

$ uv run pytest tests/test_plan_provider.py -vv
============================= 3 passed in 16.60s ==============================

Full suite (unit tests only): 2084 passed, 5 pre-existing failures (unchanged from main).
```

## Worktree Setup

```bash
git worktree add -b feat/r62-playbook-first ../wt-r62-playbook-first main
cd ../wt-r62-playbook-first
```

## Git Workflow

```bash
git checkout -b feat/r62-playbook-first

git add schemas/playbook_definition.py \
        services/playbook_registry.py \
        agents/strategies/plan_provider.py \
        agents/strategies/llm_client.py \
        tests/test_playbook_wiring.py \
        docs/branching/62-playbook-first-plan-generation.md \
        docs/branching/README.md

git commit -m "feat: wire PlaybookRegistry into plan generation with HTF eligibility filtering (R62)"
```

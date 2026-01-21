# Backtest 6d140887 Fixes: Trigger Churn

Context: `backtest-6d140887-acea-4a0c-93ad-3769573022ae` ran with the live LLM strategist (no shim). The run generated 5 plans in one day, including judge-triggered replans with `trigger_diff.unchanged = 6`, and one intra-day trigger swap.

## What Needs Fixing

- Replans are triggered even when the plan did not materially change.
- One intra-day replan swapped two triggers during the same plan window, making plan continuity fragile.
- `plan_log.generated_at` always reflects the plan window start, not the actual replan timestamp.

## Fix Options

### 1) Add a "No-Change" Replan Guard

Skip judge-triggered replans when:
- `trigger_diff.unchanged == len(triggers)` and
- No active constraint changes (risk limits, vetoes, regime shift).

Suggested hooks:
- `agents/strategies/plan_provider.py` (decision point before LLM call)
- `backtesting/llm_strategist_runner.py` (adaptive replan gate)

Success criteria:
- `num_llm_calls` drops without reducing trade count.
- `trigger_diff.added/removed` only changes when constraints shift.

### 2) Enforce Trigger Continuity via Input Contract

Hard-require `previous_triggers` in `LLMInput` and assert:
- LLM must keep ≥ N% of triggers when constraints unchanged.
- LLM must only replace triggers when reasoning is tied to a regime/constraint shift.

Suggested hooks:
- `schemas/llm_strategist.py` (already added)
- `agents/strategies/plan_provider.py` (pass previous triggers)
- `agents/strategies/llm_client.py` (validate response)
- `prompts/llm_strategist_prompt.txt` (make continuity mandatory)

Success criteria:
- `trigger_diff.unchanged > 0` in most replans.
- Trigger IDs remain stable across judge-triggered replans.

### 3) Add Deterministic Trigger ID Matching

Stabilize IDs across replans when logic is equivalent:
- Hash `(symbol, timeframe, category, direction, entry_rule, exit_rule)` into IDs.
- When a new plan arrives, attempt to match triggers and reuse IDs.

Suggested hooks:
- `agents/strategies/trigger_vector_store.py` or a new matcher in plan provider.

Success criteria:
- Trigger history is continuous across replans.
- Judge analytics can attribute performance to consistent triggers.

### 4) Record Actual Replan Timestamps

Add `generated_at_actual` and `replan_reason` to plan logs so audits show real replan timing.

Suggested hooks:
- `backtesting/llm_strategist_runner.py` (plan_log payload)
- `services/strategist_plan_service.py` (event payload)

Success criteria:
- `plan_log.generated_at_actual` matches plan generation time.
- Churn metrics can be computed without log scraping.

## Recommended Order

1) Add no-change replan guard (cheap, immediate win).
2) Enforce continuity via `previous_triggers`.
3) Add deterministic trigger ID matching.
4) Add `generated_at_actual` and `replan_reason` fields.


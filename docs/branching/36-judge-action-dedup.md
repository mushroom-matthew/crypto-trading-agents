# Branch: judge-action-dedup

## Purpose
Multiple judge actions can fire at the same timestamp within an evaluation window, causing later actions to silently overwrite earlier ones. In backtest `58cb897f`, 4 actions hit at the same timestamp on Jan 1, with earlier ETH multiplier adjustments (0.85) overwritten by later actions. Only the last action's multipliers persisted.

## Source Evidence
- Backtest `58cb897f`: 15 `judge_action_applied` events, all intraday
- Jan 1: 4 actions at same timestamp, ETH 0.85 multiplier appears in early actions but NOT in final risk_adjustments
- The overwrite is silent -- no event indicates that a prior action was superseded
- This undermines judge effectiveness: the judge makes corrections that don't persist

## Root Cause
The judge action application code in `judge_feedback_service.py` applies each action independently. When multiple actions arrive in the same evaluation window, each one overwrites the prior one's adjustments without merging or conflict resolution.

## Scope
1. **De-duplicate judge actions** within the same evaluation window: keep only the last action per scope (per-symbol or global)
2. **Emit superseded events** when an action overwrites a prior one
3. **Merge compatible adjustments** when possible (e.g., two actions adjusting different symbols should both apply)

## Out of Scope
- Changing the judge evaluation frequency or timing
- Modifying how the judge generates actions

## Key Files
- `services/judge_feedback_service.py` — Action application logic
- `agents/event_emitter.py` — Emit superseded events
- `schemas/judge_feedback.py` — JudgeAction schema (check for scope/TTL fields)

## Implementation Steps

### Step 1: Group actions by evaluation window
Before applying actions, group by timestamp (within a 60-second window) and scope (global vs per-symbol).

### Step 2: Merge non-conflicting actions
If two actions in the same window adjust DIFFERENT symbols, merge both:
```python
# Action 1: ETH multiplier = 0.85
# Action 2: BTC multiplier = 0.90
# Merged: both apply (no conflict)
```

### Step 3: Resolve conflicting actions
If two actions in the same window adjust the SAME scope:
- Keep the last one (by timestamp)
- Emit `judge_action_superseded` event with the dropped action's details

### Step 4: Add guard in application code
In `judge_feedback_service.py`, add a `_last_applied_action_id` tracker per scope. Skip application if a newer action for the same scope was already applied in this window.

## Test Plan
```bash
# Unit: non-conflicting actions merged
uv run pytest tests/test_judge_feedback_service.py -k action_merge -vv

# Unit: conflicting actions de-duped (last wins)
uv run pytest tests/test_judge_feedback_service.py -k action_dedup -vv

# Unit: superseded event emitted
uv run pytest tests/test_judge_feedback_service.py -k action_superseded -vv
```

## Test Evidence
```
(to be filled after implementation)
```

## Acceptance Criteria
- [ ] Non-conflicting actions in same window are merged
- [ ] Conflicting actions are de-duped (last wins)
- [ ] `judge_action_superseded` event emitted for dropped actions
- [ ] All multiplier adjustments persist correctly after de-dup
- [ ] Existing tests still pass

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-12 | Runbook created from backtest 58cb897f validation analysis | Claude |

## Git Workflow
```bash
git checkout -b fix/judge-action-dedup
# ... implement changes ...
git add services/judge_feedback_service.py agents/event_emitter.py tests/test_judge_feedback_service.py
git commit -m "De-duplicate judge actions within evaluation windows, merge non-conflicting adjustments"
```

# Branch: judge-action-contract

## Purpose
Convert judge outputs from advisory text into explicit, persisted, and auditable actions that are enforced deterministically across backtest and live paths.

## Source Evidence
- Backtest `7c860ae1`: judge suggestions visible in the event timeline but not consistently enforced.
- Constraints overwritten between evals caused disabled triggers to reappear.
- Judge attribution schema exists but `recommended_action` is not wired into decision routing.

## Root Cause
Judge output is not treated as an action contract. There is no structured, persisted “judge action” record, no TTL/override semantics, and no gating from `recommended_action` into actual replan/policy behavior.

## Scope
- Define a structured Judge Action contract with TTL and precedence rules.
- Persist judge actions in the run registry and emit explicit action events.
- Wire `JudgeAttribution.recommended_action` into evaluation routing (replan vs hold vs policy_adjust).
- Ensure judge constraints persist across evaluations until TTL expiry or explicit reset.

## Out of Scope / Deferred
- Policy engine math changes (Phase 1 policy integration).
- Model integration or p_hat sourcing.
- UI redesign beyond surfacing action events.

## Key Files
- `schemas/judge_feedback.py`
- `schemas/strategy_run.py`
- `services/judge_feedback_service.py`
- `agents/judge_agent_client.py`
- `backtesting/llm_strategist_runner.py`
- `ops_api/event_store.py`
- `ops_api/materializer.py`
- `ops_api/routers/agents.py`
- `ui/src/components/EventTimeline.tsx`
- `ui/src/components/AgentInspector.tsx`

## Frontend Impact
- New event types (`judge_action_applied`, `judge_action_skipped`) should be surfaced in the Event Timeline.
- Agent Inspector may need to display action status, TTL, and applied constraints.

## Implementation Steps

### Step 1: Define Judge Action schema
Add a `JudgeAction` model capturing:
- `action_id`, `source_eval_id`, `recommended_action`, `constraints`, `policy_adjustments`, `risk_adjustments`, `stance_override`, `ttl_evals`, `created_at`, `applied_at`, `status`.

### Step 2: Persist and version judge actions
Store the latest Judge Action on `StrategyRun` and persist with timestamps. Enforce TTL and explicit reset semantics so constraints are not overwritten blindly each evaluation.

### Step 3: Wire recommended_action routing
Use `JudgeAttribution.recommended_action` to decide whether to replan, hold, or policy_adjust. This replaces score-only triggers.

### Step 4: Emit action events
Emit `judge_action_applied` and `judge_action_skipped` events with correlation IDs and reason codes. Surface these in Event Timeline for auditability.

### Step 5: Backfill deterministic behavior
Ensure backtest and live paths use the same action contract logic. The output should be consistent across shim and LLM modes.

## Test Plan
```bash
uv run pytest tests/test_judge_action_contract.py -vv
uv run pytest tests/test_judge_recommended_action_routing.py -vv
uv run pytest -k judge -vv
```

## Test Evidence
```
TODO
```

## Acceptance Criteria
- Judge actions are persisted with TTL and do not reset unless explicitly cleared.
- `recommended_action` drives replan vs policy_adjust vs hold.
- Event timeline shows `judge_action_applied` or `judge_action_skipped` for each evaluation.
- Backtest and live paths apply the same action contract.

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-11 | Runbook created from backtest 7c860ae1 analysis | Claude |

## Worktree Setup (recommended for parallel agents)
```bash
git fetch
git worktree add -b judge-action-contract ../wt-judge-action-contract main
cd ../wt-judge-action-contract

# When finished (after merge)
git worktree remove ../wt-judge-action-contract
```

## Git Workflow
```bash
git checkout main
git pull
git checkout -b judge-action-contract

# ... implement changes ...

git add schemas/judge_feedback.py \
  schemas/strategy_run.py \
  services/judge_feedback_service.py \
  agents/judge_agent_client.py \
  backtesting/llm_strategist_runner.py \
  ops_api/event_store.py \
  ops_api/materializer.py \
  ops_api/routers/agents.py \
  ui/src/components/EventTimeline.tsx \
  ui/src/components/AgentInspector.tsx

uv run pytest -k judge -vv
git commit -m \"Judge: add structured action contract and routing\"
```

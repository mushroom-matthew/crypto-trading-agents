# Branch: strategist-tool-loop (later)

## Purpose
Add a read-only tool-call loop for the strategist across backtesting, paper trading, and live planning.

## Source Plans
- docs/STRATEGIST_TOOL_LOOP_PLAN.md

## Scope
- Implement tool request schema parsing and allowlist enforcement.
- Execute read-only tools in activities/services only (never inside workflows).
- Inject tool results into LLMInput.global_context.tool_results.
- Add telemetry events for tool request/result and LLM call metadata.
- Update prompt templates to reference tool results when present.

## Out of Scope / Deferred
- Any state-changing tool calls.
- Policy engine or judge unification.

## Key Files
- backtesting/activities.py
- backtesting/llm_strategist_runner.py
- tools/paper_trading.py
- services/strategist_plan_service.py
- agents/strategies/plan_provider.py
- prompts/llm_strategist_prompt.txt
- mcp_server/app.py (tool inventory and allowlist)

## Dependencies / Coordination
- Coordinate with policy-pivot-phase0 on replan guard behavior.
- Avoid conflicts with prompt changes in comp-audit-indicators-prompts.

## Acceptance Criteria
- Tool request schema validated and allowlist enforced.
- Read-only tools executed only in activities/services.
- Tool usage visible in event store.
- Planning succeeds even if tools fail.

## Test Plan (required before commit)
- uv run pytest -k strategist -vv
- uv run pytest -k backtesting -vv
- uv run pytest -k paper_trading -vv

If tests cannot be run, obtain user-run output and paste it below before committing.

## Human Verification (required)
- Run a backtest or paper/live plan with tool loop enabled and confirm tool requests/results are logged.
- Attempt a disallowed tool request and confirm allowlist enforcement blocks it.
- Paste run id/log observations in the Human Verification Evidence section.

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b strategist-tool-loop

# Work, then review changes
git status
git diff

# Stage changes
git add backtesting/activities.py \
  backtesting/llm_strategist_runner.py \
  tools/paper_trading.py \
  services/strategist_plan_service.py \
  agents/strategies/plan_provider.py \
  prompts/llm_strategist_prompt.txt \
  mcp_server/app.py

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest -k strategist -vv
uv run pytest -k backtesting -vv
uv run pytest -k paper_trading -vv

# Commit ONLY after test evidence is captured below
git commit -m "Strategist: read-only tool loop"
```

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)


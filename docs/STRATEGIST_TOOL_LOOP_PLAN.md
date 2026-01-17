# Strategist Tool-Loop Plan

This document proposes a detailed, non-implementation plan for adding a read-only tool-call loop to the LLM strategist across backtesting, paper trading, and live operation. It focuses on safety, determinism, observability, and integration boundaries.

## Summary

We will add a two-pass strategist flow:
1) LLM emits a structured request for read-only tool calls.
2) The system executes those tools, injects the results into the strategist context, then asks the LLM to return a StrategyPlan.

This enables adaptation to shifting market conditions while preserving safety and deterministic workflow execution.

## Goals

- Allow the strategist to request additional read-only data when needed.
- Keep all state-changing operations disallowed in the tool loop.
- Ensure the loop runs in activities/services only (never inside Temporal workflows).
- Provide rich telemetry: which tool was used, when, why, and with what result.
- Make behavior consistent across backtesting, paper trading, and live modes.

## Non-Goals

- Allowing the LLM to place orders or modify state directly.
- Implementing a multi-step agent conversation or long tool chains.
- Replacing existing indicator computations or removing precomputed inputs.
- Implementing a vector store or retrieval system (that is a separate effort).

## Current Behavior (Baseline)

- The strategist only sees precomputed indicators and context injected into LLMInput.
- No tool calls are possible from the LLM side.
- Strategy selection in UI maps to a prompt template loaded from `prompts/` or `prompts/strategies/`.
- LLM calls are wrapped with Langfuse spans and EventStore events for telemetry.

### Current State by Mode

Backtesting (LLM strategist mode):
- Strategy prompts are passed from the backtest config and routed into the backtester as `strategy_prompt`.
- LLMInput includes portfolio state, per-asset indicator snapshots, and a `global_context` that can include market_structure, factor_exposures, rpr_comparison, judge feedback, and strategy memory when available.
- The strategist runs inside activities, not inside workflows, and produces StrategyPlan outputs without any tool calls.

Paper trading:
- Strategy selection occurs in the API layer; the selected template is loaded server-side and passed to the strategist as `strategy_prompt`.
- LLMInput is built from a lighter market_context snapshot and portfolio state; it is not identical to backtesting inputs.
- Strategy planning occurs in an activity, uses LLMClient directly, and does not invoke tools.

Live planning:
- StrategyPlanProvider/StrategistPlanService invoke LLMClient with LLMInput built from live pipeline state.
- The default prompt is loaded from `prompts/llm_strategist_prompt.txt` unless overridden via an explicit template or env var.
- No tool calls are possible; all inputs are precomputed and injected in-process.

## Proposed Tool-Loop Architecture

### Two-Pass Flow (Single Iteration)

1) Tool Request Pass
   - LLM receives the same input as today and returns a JSON object describing tool calls.
   - Output is validated against a strict schema and capped by limits.

2) Tool Execution
   - Only read-only tools are executed.
   - Results are collected, normalized, and added into LLMInput.global_context.

3) Plan Pass
   - LLM receives the original input plus tool results.
   - LLM returns a StrategyPlan JSON payload.

If the tool request is empty or invalid, the system skips tool execution and goes directly to plan generation using the existing inputs.

### Determinism and Temporal Safety

- Tool execution is non-deterministic and must never run inside Temporal workflows.
- All tool calls must happen inside activities or service layers before workflow logic resumes.
- Backtesting and paper trading already use activities, so the loop is safe there.
- Live planning should run through the same service layer and should never call tools inside workflow code.

## Read-Only Tool Policy

### Allowlist

Only read-only tools are permitted. A tool is read-only if it:
- Does not mutate portfolios, ledgers, orders, preferences, or workflows.
- Does not initiate streaming or subscribe to data.
- Does not write to disk except for local caches.

### Disallowed

Any state-changing or side-effecting tool, including:
- Placing orders or signals.
- Modifying user preferences.
- Starting market streams.
- Writing to persistent stores.

### Initial Candidate Tool Categories

The exact allowlist is to be finalized by inventorying MCP and Ops API endpoints, but likely candidates include:
- Market data snapshots (ticks, candles, spreads).
- Technical metrics (precomputed indicator bundles).
- Portfolio status reads (positions, exposure, PnL).
- Risk and performance metrics (Sharpe, drawdown, volatility).

### Tool Inventory Task (Required Before Implementing)

1) Enumerate MCP tools in `mcp_server/app.py`.
2) Tag each tool as read-only or state-changing.
3) Select the final allowlist and assign stable tool names.

## Tool Request Schema (LLM Output)

The LLM must output a JSON object with strict structure. Example schema:

{
  "tool_calls": [
    {
      "tool_name": "get_market_snapshot",
      "params": {"symbols": ["BTC-USD", "ETH-USD"], "timeframe": "1h"},
      "reason": "Need current volatility regime to set breakout thresholds."
    }
  ]
}

Schema rules:
- tool_calls is optional; default empty list if missing.
- Maximum tool_calls per plan: 2.
- Only tool_name values in the allowlist are accepted.
- params must be JSON-serializable.
- reason is a short string for auditing.

## Tool Results Injection

Results are stored in `LLMInput.global_context.tool_results`:

{
  "tool_results": [
    {
      "tool_name": "get_market_snapshot",
      "params": {...},
      "duration_ms": 120,
      "result": {...}
    }
  ]
}

This keeps all tool outputs visible to the strategist while avoiding ad-hoc fields.

## System Limits and Fallbacks

- max_tool_calls_per_plan: 2
- max_tool_iterations: 1
- max_total_tool_time_ms: 5000
- tool_timeout_ms per call: 2000

Fallback behavior:
- If a tool fails, record the failure in tool_results and continue.
- If all tools fail, proceed with the plan pass using existing inputs.
- If the tool request is invalid, ignore it and proceed without tools.

## Integration Points (All Three Modes)

### Backtesting

Location:
- `backtesting/activities.py` -> `run_llm_backtest_activity`
- `backtesting/llm_strategist_runner.py` (strategy prompt path is injected here)

Plan:
- Introduce a tool-loop helper in the backtester path before `LLMClient.generate_plan`.
- Keep the loop outside any workflow code and inside the activity.

### Paper Trading

Location:
- `tools/paper_trading.py` -> `generate_strategy_plan_activity`

Plan:
- Add the tool-loop helper in this activity before `LLMClient.generate_plan`.

### Live Strategist

Location:
- `services/strategist_plan_service.py` and/or `agents/strategies/plan_provider.py`

Plan:
- Centralize the tool loop in the plan-provider or LLM client layer so all live plan requests share the same behavior.
- Ensure this is only run in service code and not inside workflows.

## Prompt Updates

The strategist prompt must:
- Mention optional tool results if present in `global_context.tool_results`.
- Never reference repository file paths or assume external documents.
- Use optional telemetry only when present (market_structure, factor_exposures, rpr_comparison, quality signals).
- Avoid labeling any sections as DRAFT.

## Telemetry and Auditing

Emit EventStore events for:

1) strategist_tool_request
   - plan_id, run_id, tool_calls, source, prompt_template_id

2) strategist_tool_result
   - plan_id, run_id, tool_name, duration_ms, status, result_summary

3) llm_call
   - add fields: prompt_template_id, prompt_template_hash, context_flags, tool_call_count

These events allow UI/ops to audit tool usage per plan and ensure tools are used only when appropriate.

## Safety Checklist

- Tool allowlist enforced at runtime.
- No tool calls inside workflows.
- Strict JSON parsing and schema validation for tool requests.
- Maximum call counts and timeouts enforced.
- Clear fallback to metrics-only planning.

## Acceptance Criteria

- Strategist can request read-only tool calls in all three modes.
- Tool calls are never state-changing; violations are blocked.
- Tool usage is visible in `/agents/events` with timestamps and plan_id correlation.
- Planning proceeds even when tools fail.
- Prompt templates contain no DRAFT sections and no file-path references.

## Open Questions

- Which exact MCP tools are approved for the allowlist?
- Should tool results be normalized to a strict schema or stored raw?
- Do we want a UI indicator for tool usage per plan?

## Implementation Phases (When We Build It)

Phase 1: Tool Inventory and Allowlist
- Classify MCP tools and Ops API endpoints.
- Define and document the read-only allowlist.

Phase 2: Tool Loop Helper
- Implement the tool request schema parser.
- Implement tool execution wrapper with timeouts and telemetry.

Phase 3: Integration
- Wire into backtesting, paper trading, and live plan provider.
- Add prompt updates and global_context injection.

Phase 4: Observability
- Emit tool events and update UI to display tool usage.

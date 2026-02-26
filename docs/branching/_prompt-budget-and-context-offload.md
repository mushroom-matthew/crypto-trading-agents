# Backlog: Prompt Budget + Context Offloading (ADR Follow-on)

## Status: Backlog - can begin before per-instrument workflow implementation

## Purpose

Build on `ADR-portfolio-judge-routing-and-control-plane.md` by defining a concrete plan
to reduce prompt fat, enforce token budgets, and offload static/reference context to
vector store or file search instead of inlining it into strategist/judge prompts.

This runbook is focused on prompt efficiency and prompt quality:

- make prompts sharper (less generic prose, more decision-critical context)
- reduce token cost and latency
- prevent multi-instrument prompt ballooning
- preserve or improve plan/judge quality via better retrieval and deterministic summaries

## Why This Exists

The current architecture is transitioning toward symbol-local strategist/judge workflows
plus a deterministic portfolio control plane (ADR, draft dated 2026-02-24), but prompt
construction is still largely optimized for convenience rather than token discipline.

Observed issues from the prompt audits (2026-02-26):

- **Large strategist fixed cost**: `simple + schema` is ~6.8k tokens before runtime
  payloads; `full + schema` is ~10.8k.
- **Multi-instrument payload growth is linear and steep**: strategist `LLMInput` grows
  roughly with `symbols × timeframes × populated_indicator_fields`.
- **Judge prompts include uncapped JSON dumps** (`market_structure`, `factor_exposures`,
  `risk_state`, and some `strategy_context`) that can grow with instrument count.
- **Static prompt duplication risk**: `LLM_STRATEGIST_PROMPT` may be injected twice
  (as `prompt_template` and again via `build_prompt_context()`).
- **Retrieval misalignment**: strategy retrieval query mostly keys off the first asset,
  while the serialized strategist payload may include many assets.

Without budget controls, per-call tokens can balloon faster than model quality improves.

## ADR Alignment (What This Runbook Adds)

This runbook extends the portfolio ADR with a prompt-boundary policy:

1. **Symbol-local strategist/judge prompts should be symbol-local by default**
   - one symbol's indicators + local position context + compact portfolio envelope
   - no full portfolio-wide raw indicator dumps

2. **Portfolio-level reasoning prompts (monitor layer) should use aggregated summaries**
   - cohort/correlation/drawdown summaries and evidence refs
   - no raw per-symbol indicator snapshots unless explicitly escalated

3. **Portfolio control plane remains deterministic**
   - no LLM prompt expansion in the control plane
   - portfolio constraint envelopes are typed and compact

This keeps the ADR's authority boundaries intact while reducing prompt bloat.

## Baseline Measurements (Prompt Audit Snapshot - 2026-02-26)

These numbers should be used as the initial benchmark for improvements.

### Strategist Prompt Files (`o200k_base`)

- `prompts/llm_strategist_simple.txt`: ~872 tokens
- `prompts/llm_strategist_prompt.txt`: ~4,858 tokens
- `prompts/strategy_plan_schema.txt`: ~5,966 tokens
- `simple + schema` runtime system prompt baseline: ~6,838 tokens
- `full + schema` runtime system prompt baseline: ~10,824 tokens
- `simple + schema + strategy template` (varies by template): ~7,748 to ~8,501 tokens

### Judge Prompt File (`o200k_base`)

- `prompts/llm_judge_prompt.txt`: ~1,163 tokens (base template only)
- Actual judge prompt size varies substantially with injected runtime sections

### Strategist Payload Growth (Synthetic `LLMInput.to_json()`, `o200k_base`)

Measured trend from schema-conformant synthetic payloads:

- `10 symbols x 1 timeframe`: ~6.9k to ~8.6k tokens (user payload only)
- `10 symbols x 2 timeframes`: ~13.1k to ~16.6k tokens
- `10 symbols x 4 timeframes`: ~25.7k to ~32.6k tokens

Implication: multi-symbol strategist calls can exceed 40k input tokens once the system
prompt and retrieval context are included.

## Problem Statement

The current prompt construction paths inline too much information:

- strategist system prompt inlines large schema + guidance blocks
- strategist user payload often includes all active symbols and multi-timeframe snapshots
- judge prompt inlines raw JSON objects that grow with portfolio complexity
- static instructions and reference material are sometimes included inline instead of
  retrieved or summarized on demand

This reduces prompt precision and makes token usage scale poorly as autonomous instrument
selection expands.

## Scope

1. Token-budget policy for strategist/judge/portfolio-monitor prompts
2. Prompt decomposition into core instructions vs retrievable references
3. Context slimming for multi-instrument strategist payloads
4. Judge context summarization/capping for multi-symbol portfolios
5. Retrieval offloading plan (vector store + file search)
6. Telemetry and regression checks for token usage and prompt quality
7. ADR-aligned prompt boundaries for symbol-local vs portfolio-level reasoning layers

## Out of Scope

- Implementing per-instrument workflows themselves (`_per-instrument-workflow.md`)
- Implementing portfolio control plane (`_portfolio-control-plane.md`)
- Changing deterministic trigger engine semantics
- Replacing the current model stack

## Success Criteria (What “Sharper Prompts” Means)

1. Lower token usage with no reduction in schema validity or safety behavior
2. Higher relevance: prompts carry decision-critical context, not generic reference text
3. Clear boundary between:
   - static reference knowledge (retrieval/file search)
   - dynamic decision state (prompt)
   - deterministic enforcement (code)
4. Scalable behavior as symbol count increases

## Proposed Budget Policy (Initial Targets)

These are starting targets for paper-trading tuning, not hard production limits yet.

### Symbol-local Strategist (target future shape)

- **System prompt target**: <= 4,500 tokens
- **User payload target**: <= 6,000 tokens per symbol invocation
- **Stretch target**: <= 3,000 + <= 4,000

### Portfolio Monitor / Portfolio Judge (future ADR layer)

- **System prompt target**: <= 2,500 tokens
- **User payload target**: <= 5,000 tokens (aggregated summaries only)

### Judge (current backtest/service path)

- **Base template + dynamic prompt target**: <= 6,000 input tokens for common cases
- Hard cap strategy should degrade gracefully to deterministic summaries when exceeded

## Key Findings Driving Actions (from the audits)

1. **`strategy_plan_schema.txt` dominates strategist fixed cost**
   - The schema block (~5,966 tokens) is the single largest fixed component.

2. **Strategist payloads balloon with instrument count**
   - `assets[]` includes rich multi-timeframe indicator snapshots.
   - Backtesting commonly passes all symbols in one `LLMInput`.

3. **Judge prompt growth is partly uncapped**
   - Several sections dump raw JSON without token-aware truncation/summarization.

4. **Retrieval is underused for prompt slimming**
   - Static instructions/examples remain inline instead of retrieved as needed.

5. **Per-instrument architecture is planned but not implemented**
   - We should slim current prompts now and design the budget policy to fit the future
     symbol-local ADR flow.

## Implementation Plan

### Step 1: Add prompt-budget telemetry and guardrails (no behavior change)

Add budget instrumentation before modifying prompt content.

Actions:

- Add a shared `estimate_tokens_o200k(text)` helper (tiktoken-backed) for prompt assembly
  sites.
- Record section-level token counts for strategist and judge prompts:
  - base prompt
  - schema block
  - template block
  - prompt_context block
  - strategy retrieval block
  - trigger example block
  - user payload (`LLMInput.to_json()`)
- Emit token breakdowns to existing LLM telemetry/events (alongside `tokens_in/out`).
- Add warning logs when prompt sections exceed budget thresholds.

Acceptance:

- We can answer “where the tokens went” from telemetry for every strategist/judge call.

### Step 2: Fix prompt duplication and dead-weight injection

Remove obvious token waste before deeper refactors.

Actions:

- Eliminate duplicate `LLM_STRATEGIST_PROMPT` injection path:
  - today it can be appended as `prompt_template` and also via `build_prompt_context()`
  - choose one source of truth for free-form strategy guidance
- Audit prompt assembly for repeated headings/sections introduced by both retrieval and
  static prompt text.
- Add a small normalization layer to dedupe identical blocks before final concatenation.

Acceptance:

- Identical guidance text cannot appear twice in the same strategist system prompt.

### Step 3: Split “core prompt” vs “reference material” for strategist

The strategist system prompt should include only what is needed for valid outputs.
Everything else should be retrievable.

Actions:

- Create a **compact schema contract** (new file) with only:
  - required output shape
  - hard constraints that cannot be recovered from validators
  - minimal examples (or no examples)
- Move extended examples/explanations from `strategy_plan_schema.txt` into retrievable
  docs (vector store docs and/or local reference files).
- Keep deterministic enforcement rules in code/docs, not prompt prose, when possible.
- Make strategist prompt assembly choose between:
  - `schema_core` (default)
  - `schema_verbose` (debug/fallback only)

Notes:

- This should preserve the hard-template-binding and compile-time enforcement model
  rather than teaching everything in prompt prose.

Acceptance:

- Strategist fixed prompt cost drops materially (target: reduce `simple+schema` by >=30%).

### Step 4: Slim multi-instrument strategist payloads before per-instrument split lands

We should reduce payload size even before `_per-instrument-workflow.md` is implemented.

Actions:

- Introduce a **payload shaping mode** for strategist calls:
  - `symbol_local_preferred` (default when screener recommendation exists)
  - `multi_symbol_compact`
  - `multi_symbol_full` (debug only)
- When `instrument_recommendation.selected_symbol` / `preferred_symbol` exists:
  - include full indicator snapshots only for the selected symbol
  - include compact summaries for other symbols (price/trend/vol/regime/exposure only)
- Cap `previous_triggers` by relevance (already capped at 50 in backtest runner; add
  symbol-aware ranking before truncation).
- Cap timeframes included in `assets[].indicators` by template/timeframe need.
  Example: if selected template is `compression_breakout`, include only relevant
  timeframes for entry/anchor logic rather than all cached timeframes.

Acceptance:

- Multi-symbol strategist user payload growth flattens significantly for non-selected
  symbols (compact summaries instead of full snapshots).

### Step 5: Make judge prompt construction token-aware and symbol-scalable

Judge prompts should consume compact evidence summaries, not raw JSON dumps.

Actions:

- Replace raw JSON dumps (`market_structure`, `factor_exposures`, `risk_state`,
  `strategy_context`) with deterministic summarizers and top-k selectors.
- Add per-section caps and truncation strategies for judge context:
  - max symbols included in each section
  - max fields per symbol
  - max bytes/chars per serialized object
- Split judge inputs into:
  - **symbol-local judge** payload format (for future per-instrument workflows)
  - **portfolio-monitor** payload format (aggregated portfolio diagnostics only)
- Add explicit overflow behavior:
  - emit summary + “omitted N items” markers
  - never silently exceed token budget

Acceptance:

- Judge prompt size remains bounded as instrument count rises.

### Step 6: Offload static/reference content to vector store and file search

This is the core focus of this runbook.

Actions:

- Classify prompt content into three buckets:
  - **inline core instructions** (must remain in prompt)
  - **retrievable strategy/playbook references** (vector store)
  - **repo docs/contracts/examples** (file search / read-only tool loop)
- Expand vector store coverage for strategy/playbook docs that are currently represented
  as prompt prose or repeated examples.
- Prefer retrieval of:
  - template-specific examples
  - playbook definitions
  - historical validation evidence snippets
  - longer rationale text / edge-case examples
- Keep inline only:
  - output contract
  - hard response rules
  - immediate decision objective
  - current state summaries
- Introduce a read-only **file-search preprocessor** (deterministic) or strategist
  tool-loop path (when the backlog `later/_strategist-tool-loop.md` is pursued) to fetch
  relevant local docs instead of embedding them wholesale in prompts.

Acceptance:

- New prompt guidance additions default to retrieval/file search unless explicitly marked
  `inline-required`.

### Step 7: ADR-following prompt boundaries for future per-instrument architecture

Translate ADR boundaries into prompt contracts now so the migration is clean later.

Actions:

- Define prompt contracts for three future layers:
  1. `InstrumentStrategistPromptInput` (symbol-local)
  2. `InstrumentJudgePromptInput` (symbol-local performance/evidence)
  3. `PortfolioMonitorPromptInput` (portfolio aggregates only)
- Explicitly forbid portfolio-control-plane prompt usage (deterministic only).
- Specify allowed shared fields across symbol prompts:
  - portfolio constraint envelope
  - portfolio mode
  - symbol allocation approval/clipping reason
- Forbid symbol prompts from receiving full raw multi-symbol indicator snapshots by
  default (opt-in debug mode only).

Acceptance:

- Prompt contracts reflect ADR authority boundaries before implementation starts.

### Step 8: Retrieval quality + token efficiency validation

Prompt slimming can reduce quality if retrieval is weak. Validate both.

Actions:

- Add tests/benchmarks for:
  - strategist parse validity
  - template routing correctness (R46/R47 analytics)
  - trigger compile rejection rates
  - judge JSON validity
  - token usage reduction
- Create a prompt regression suite with representative cases:
  - 1 symbol / 1 timeframe
  - 3 symbols / mixed templates
  - 10 symbols / high-noise screener output
- Track “tokens saved vs outcome quality” in paper trading telemetry.

Acceptance:

- Token reductions do not materially degrade routing accuracy or schema validity.

## Specific Code/Doc Touchpoints (Initial Candidates)

### Strategist

- `agents/strategies/llm_client.py`
  - section-level token accounting
  - dedupe of prompt blocks
  - budget-aware prompt assembly
- `agents/strategies/prompt_builder.py`
  - avoid duplicate `STRATEGY_GUIDANCE`
  - compact/verbose modes for prompt context
- `agents/strategies/plan_provider.py`
  - token-budget metadata in generation telemetry
- `schemas/llm_strategist.py`
  - introduce compact payload schemas or shaping helpers (symbol-local / compact modes)
- `services/strategist_plan_service.py`
  - apply payload shaping when screener recommendation selects a symbol

### Judge

- `services/judge_feedback_service.py`
  - replace raw JSON dumps with summarizers
  - per-section caps + overflow markers
  - symbol-local vs portfolio-monitor payload builders
- `agents/judge_agent_client.py`
  - align live judge path with the same summarization/budget policy

### Retrieval / reference offloading

- `vector_store/retriever.py`
  - budget-aware retrieval block sizing
  - optional symbol-aware querying (avoid first-asset bias for multi-symbol prompts)
- `vector_store/strategies/*.md`, `vector_store/playbooks/*.md`
  - move long prompt prose/examples into retrievable docs
- `docs/branching/*` and contract docs
  - mark content as `retrieval/reference` vs `inline-required`

## Deterministic Rules (Prompt Budget Discipline)

1. **No uncapped JSON dumps** in LLM prompts.
2. **Inline only what affects the immediate decision**.
3. **Static reference text belongs in retrieval/file search by default**.
4. **Budget overflow must degrade gracefully** (summaries + omission markers), not fail
   open into giant prompts.
5. **Per-instrument migration must reduce per-call payload size**, not just move the same
   monolithic context into N separate prompts.

## Dependencies and Ordering

Can start now (recommended):

- prompt telemetry
- duplication cleanup
- strategist/judge summarization and caps
- schema compaction design
- retrieval/file-search offloading plan

Should align with (but not block on) implementation:

- `_per-instrument-workflow.md`
- `ADR-portfolio-judge-routing-and-control-plane.md`
- `_portfolio-control-plane.md`
- `_portfolio-monitor-and-reflection.md`

## Open Questions

1. What minimum inline schema text preserves parse validity without increasing repair loops?
2. Should symbol-local strategist payload shaping happen in `StrategistPlanService` or in
   a dedicated prompt-input builder?
3. How much strategy/playbook detail should be retrievable vs summarized into the prompt?
4. Should the future file-search offload be a deterministic preprocessor only, or a
   read-only strategist tool loop (see backlog `later/_strategist-tool-loop.md`)?
5. What token budgets should become hard production limits vs warning thresholds?

## Acceptance Criteria (for this runbook's implementation phase)

- Strategist prompt assembly emits section-level token telemetry
- Duplicate strategist guidance injection is removed
- A compact schema path exists and is the default
- Judge prompt builder no longer emits uncapped raw JSON sections
- Multi-symbol strategist payload shaping supports a selected-symbol focused mode
- Retrieval/file-search offloading policy is documented and enforced in review
- Prompt budgets are codified for symbol-local strategist/judge and portfolio monitor
- Paper-trading telemetry shows meaningful token reduction with no material regression in:
  - plan JSON validity
  - template routing accuracy
  - compile-time trigger rejection quality
  - judge JSON validity

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-26 | Backlog runbook created - prompt budget, context offloading, and ADR-aligned prompt boundaries | Codex |

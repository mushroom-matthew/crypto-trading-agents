# Prompt Audit - 2026-04-23

## Executive Summary

The prompt surface is larger than the live system actually uses, and several prompt paths are overlapping or stale.

The main issues:

1. The live strategist default is `prompts/llm_strategist_simple.txt`, but the prompt management API edits `prompts/llm_strategist_prompt.txt`. That means prompt edits can miss the real production path.
2. `prompts/strategy_plan_schema.txt` and `prompts/strategy_plan_schema_core.txt` are currently identical, so the `STRATEGIST_SCHEMA_MODE` split has no practical effect.
3. Strategy guidance can be injected through multiple paths:
   - explicit `prompt_template`
   - `LLM_STRATEGIST_PROMPT`
   - `global_context.strategy_guidance` / `strategy_profile`
   This increases duplication risk and makes prompt behavior harder to reason about.
4. Several prompt files exist but are not wired into a live LLM path:
   - `prompts/low_level_reflection.txt`
   - `prompts/high_level_reflection.txt`
   - `prompts/judge_prompt_research.txt`
5. Several strategy templates overlap heavily, and some contain stale examples or identifiers that do not match the current schema/allowed-rule contract.

The immediate direction should be:

- shrink to one primary strategist base prompt
- shrink to one canonical schema prompt
- keep only a small allowlisted set of strategy templates visible by default
- put the rest behind feature flags
- add prompt metadata/registry so the UI and runtime use the same source of truth

This prompt hygiene work is also a prerequisite for the companion lifecycle and
replay work described in
`docs/analysis/TRADE_LIFECYCLE_REPLAY_ALIGNMENT_PLAN.md`: replay artifacts need
stable prompt, schema, and template provenance in order to be trustworthy.

## Live Prompt Topology

### Strategist

- Base prompt default: [prompts/llm_strategist_simple.txt](/home/getzinmw/crypto-trading-agents/prompts/llm_strategist_simple.txt)
  Source: [agents/strategies/llm_client.py](/home/getzinmw/crypto-trading-agents/agents/strategies/llm_client.py:774)
- Legacy alternate base prompt: [prompts/llm_strategist_prompt.txt](/home/getzinmw/crypto-trading-agents/prompts/llm_strategist_prompt.txt)
  Selected only when `STRATEGIST_PROMPT=full`
- Shared schema block: [prompts/strategy_plan_schema_core.txt](/home/getzinmw/crypto-trading-agents/prompts/strategy_plan_schema_core.txt) or [prompts/strategy_plan_schema.txt](/home/getzinmw/crypto-trading-agents/prompts/strategy_plan_schema.txt)
  Source: [agents/strategies/llm_client.py](/home/getzinmw/crypto-trading-agents/agents/strategies/llm_client.py:795)
- Strategy template addenda can enter through:
  - explicit `prompt_template`
  - retrieved vector-store template id
  - paper-trading `strategy_id`
  - backtest prompt-template path
  - `LLM_STRATEGIST_PROMPT`

### Judge

- Active judge prompt: [prompts/llm_judge_prompt.txt](/home/getzinmw/crypto-trading-agents/prompts/llm_judge_prompt.txt)
  Source: [agents/judge_agent_client.py](/home/getzinmw/crypto-trading-agents/agents/judge_agent_client.py:369)

### Other Active Prompted Flows

- Instrument selection: [prompts/instrument_recommendation.txt](/home/getzinmw/crypto-trading-agents/prompts/instrument_recommendation.txt)
  Source: [services/universe_screener_service.py](/home/getzinmw/crypto-trading-agents/services/universe_screener_service.py:938)
- Hypothesis mode paper trading: [prompts/hypothesis_plan.txt](/home/getzinmw/crypto-trading-agents/prompts/hypothesis_plan.txt)
  Source: [tools/paper_trading.py](/home/getzinmw/crypto-trading-agents/tools/paper_trading.py:591)

### Present But Not Live

- `prompts/low_level_reflection.txt`
  `services/low_level_reflection_service.py` still has a TODO to wire the LLM step.
- `prompts/high_level_reflection.txt`
  `services/high_level_reflection_service.py` is pure computation and does not load the prompt.
- `prompts/judge_prompt_research.txt`
  No runtime loader found.

## Findings

### 1. Prompt editor and runtime default are misaligned

The prompt API exposes `strategist` as [prompts/llm_strategist_prompt.txt](/home/getzinmw/crypto-trading-agents/prompts/llm_strategist_prompt.txt) via [ops_api/routers/prompts.py](/home/getzinmw/crypto-trading-agents/ops_api/routers/prompts.py:16), but the strategist runtime defaults to [prompts/llm_strategist_simple.txt](/home/getzinmw/crypto-trading-agents/prompts/llm_strategist_simple.txt) via [agents/strategies/llm_client.py](/home/getzinmw/crypto-trading-agents/agents/strategies/llm_client.py:774).

Impact:

- editing the prompt in the UI/API may not affect live plan generation
- debugging prompt behavior becomes ambiguous
- production prompt provenance is harder to trust

### 2. Schema mode split currently has no value

`cmp` shows [prompts/strategy_plan_schema.txt](/home/getzinmw/crypto-trading-agents/prompts/strategy_plan_schema.txt) and [prompts/strategy_plan_schema_core.txt](/home/getzinmw/crypto-trading-agents/prompts/strategy_plan_schema_core.txt) are identical.

Impact:

- `STRATEGIST_SCHEMA_MODE=core|verbose` adds complexity without changing behavior
- maintaining two copies guarantees future drift risk

### 3. Prompt injection paths overlap

The system prompt is assembled in [agents/strategies/llm_client.py](/home/getzinmw/crypto-trading-agents/agents/strategies/llm_client.py:440), while [agents/strategies/prompt_builder.py](/home/getzinmw/crypto-trading-agents/agents/strategies/prompt_builder.py:29) can also append `STRATEGY_GUIDANCE` from runtime context or `LLM_STRATEGIST_PROMPT`.

Impact:

- duplicate strategy guidance can be appended twice
- hard to tell whether a behavior came from the base prompt, template, env var, or request context
- prompt evaluation becomes noisy

### 4. Strategy template quality is uneven

There are 15 strategy templates under [prompts/strategies](/home/getzinmw/crypto-trading-agents/prompts/strategies), but several are either overlapping or stale relative to the current schema.

Concrete stale examples:

- `scalper_fast.txt` still shows `position == 'flat'` examples instead of `is_flat`
- `aggressive_active.txt` still references `position != 'long'` / `position == 'flat'`
- `momentum_trend_following.txt` references identifiers like `donchian_high`, `volume_ma`, `highest_close_20` that are not in the current allowed list
- `range_long.txt`, `range_short.txt`, `compression_breakout_*`, `volatile_breakout_*` describe `price_position_in_range`, which is not in the current allowed list

Impact:

- some templates are safe only as conceptual guidance, not as precise operational prompts
- LLM outputs from these prompts are more likely to require validator repair or get blocked

### 5. UI and strategy taxonomy are drifting

The screener/UI mapping in [ui/src/components/PaperTradingControl.tsx](/home/getzinmw/crypto-trading-agents/ui/src/components/PaperTradingControl.tsx:2794) still maps screener hypotheses to older template families, and scalper presets in [ui/src/lib/presets.ts](/home/getzinmw/crypto-trading-agents/ui/src/lib/presets.ts:90) still default to `aggressive_active` rather than `scalper_fast`.

Impact:

- the UI nudges users toward broad older templates
- specialized templates exist but are not first-class in the operating workflow

## Prompt Inventory

### Top-Level Prompts

| Prompt | Role | Runtime Status | Recommendation |
| --- | --- | --- | --- |
| `llm_strategist_simple.txt` | Primary strategist base prompt | Active default | Keep and strengthen |
| `llm_strategist_prompt.txt` | Legacy verbose strategist base prompt | Active only via env or explicit default strategy file use | Feature-flag as legacy fallback |
| `strategy_plan_schema.txt` | Shared strategist schema | Active | Merge into one canonical file |
| `strategy_plan_schema_core.txt` | Shared strategist schema | Active | Delete after merge or convert to generated alias |
| `llm_judge_prompt.txt` | Judge prompt | Active | Keep and strengthen |
| `instrument_recommendation.txt` | Screener recommendation prompt | Active | Keep |
| `hypothesis_plan.txt` | Hypothesis-mode planning prompt | Conditionally active | Keep behind explicit feature flag |
| `low_level_reflection.txt` | Policy reflection prompt | Not wired | Hide behind flag until implemented |
| `high_level_reflection.txt` | Batch reflection prompt | Not wired | Hide behind flag until implemented |
| `judge_prompt_research.txt` | Judge addendum | Not wired | Archive or wire explicitly, but do not leave half-live |

### Strategy Templates

| Template | Current Assessment | Recommendation |
| --- | --- | --- |
| `scalper_fast` | High-value specialized prompt, but stale examples | Keep and upgrade |
| `range_long` | High-value specialized prompt aligned with recent runtime fixes | Keep and upgrade |
| `range_short` | High-value specialized prompt aligned with recent runtime fixes | Keep and upgrade |
| `compression_breakout_long` | Useful directional specialization, but stale identifiers | Keep behind flag until refreshed |
| `compression_breakout_short` | Useful directional specialization, but stale identifiers | Keep behind flag until refreshed |
| `volatile_breakout_long` | Useful directional specialization, but stale identifiers and naming drift | Keep behind flag until refreshed/renamed |
| `volatile_breakout_short` | Useful directional specialization, but stale identifiers and naming drift | Keep behind flag until refreshed/renamed |
| `bear_defensive` | Potentially useful specialized short-only template | Keep behind flag |
| `compression_breakout` | Overlaps directional variants | Merge or archive |
| `volatility_breakout` | Broad archetype overlaps directional breakout templates | Flag as legacy |
| `mean_reversion` | Broad archetype overlaps range long/short and scalper MR logic | Flag as legacy |
| `momentum_trend_following` | Broad archetype with stale examples | Flag as legacy |
| `aggressive_active` | Broad catch-all template, currently overused by presets | Flag immediately and remove from default presets |
| `balanced_hybrid` | Broad catch-all, weakly differentiated | Flag as legacy |
| `conservative_defensive` | Broad catch-all, potentially useful only as fallback | Keep hidden fallback or merge into judge/risk stance behavior |

## Proposed Core Prompt Set

The system should converge toward this smaller first-class set:

1. `llm_strategist_simple.txt`
   The single primary strategist base prompt.
2. `llm_judge_prompt.txt`
   The single judge prompt.
3. `instrument_recommendation.txt`
   Separate instrument-selection task.
4. `hypothesis_plan.txt`
   Research-only planning mode.
5. Specialized strategy templates:
   - `scalper_fast`
   - `range_long`
   - `range_short`
   - one breakout family after refresh
   - optional `bear_defensive`

Everything else should be either:

- hidden fallback
- feature-flagged experiment
- archived

## Recommended Feature-Flag Model

Do not hardcode enablement checks across the codebase. Add one prompt registry or manifest and let runtime/UI consult it.

Suggested metadata per prompt/template:

- `id`
- `kind`: `strategist_base` | `schema` | `judge` | `template` | `research`
- `status`: `active` | `flagged` | `legacy` | `hidden`
- `enabled_by_default`
- `visible_in_ui`
- `runtime_allowlist`
- `owner`
- `notes`

Suggested first flags:

- `PROMPT_ENABLE_LEGACY_STRATEGIST=false`
- `PROMPT_ENABLE_BROAD_ARCHETYPE_TEMPLATES=false`
- `PROMPT_ENABLE_DIRECTIONAL_BREAKOUT_TEMPLATES=false`
- `PROMPT_ENABLE_BEAR_DEFENSIVE=false`
- `PROMPT_ENABLE_HYPOTHESIS_MODE=true`

Longer term, prefer a checked-in manifest file over env-only sprawl.

## Immediate Execution Order

### Phase 1 - Clean Up the Live Path

1. Make the prompt API expose the actual runtime default strategist prompt.
2. Collapse the duplicated schema files into one canonical source.
3. Remove duplicate `STRATEGY_GUIDANCE` injection paths so strategy guidance has one source of truth per request.

### Phase 2 - Prune the Template Surface

1. Hide `aggressive_active`, `balanced_hybrid`, `mean_reversion`, `momentum_trend_following`, and `volatility_breakout` from the default UI list.
2. Keep only a small allowlisted set exposed in paper trading and backtests.
3. Update presets so fast-timeframe flows use `scalper_fast` rather than `aggressive_active`.

### Phase 3 - Strengthen the Surviving Prompts

1. Rewrite `scalper_fast` around:
   - fast-timeframe realism
   - tighter target semantics
   - explicit VWAP / mid-band / SMA mean targets
   - less generic breakout language
2. Rewrite `range_long` and `range_short` around:
   - valid allowed identifiers only
   - explicit invalidation behavior
   - target realism toward `bollinger_middle` and `sma_medium`
3. Refresh one breakout family and remove stale identifiers before re-enabling.

### Phase 4 - Add Evaluation

Track prompt cohorts by prompt id/hash and compare:

- schema pass rate
- validation pass rate
- blocked-trigger rate
- unresolvable stop/target incidence
- empty-plan rate
- trade frequency
- stand-down rate
- realized and unrealized trade quality

This evaluation surface should also emit prompt provenance fields that can be
joined onto lifecycle replay artifacts later:

- `prompt_id`
- `prompt_hash`
- `schema_id`
- `template_id`
- `strategy_template_version`

## First Implementation Slice

The first code change set should do only these:

1. introduce a prompt registry/manifest
2. re-point the prompt API to the actual live strategist prompt
3. hide legacy templates from UI/runtime defaults
4. update fast presets to `scalper_fast`

That gives us an immediately cleaner operating surface without rewriting every prompt at once.

## Cross-Reference

Once the prompt registry and provenance cleanup above are complete, the next
complementary documentation and implementation track is:

- `docs/analysis/TRADE_LIFECYCLE_REPLAY_ALIGNMENT_PLAN.md`

That document depends on this audit's cleanup so replay packs and plan-audit
surfaces can tie executed trades back to the correct prompt cohort without
ambiguous runtime prompt selection.

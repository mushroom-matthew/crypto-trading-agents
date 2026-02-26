# Runbook 59: Directional Bias Templates

## Purpose

Introduce explicit long/short directional bias across all strategy hypotheses — in the screener,
the vector store, the prompt templates, and the trigger compiler.

### Problem statement

The current template architecture is directionality-ambiguous by design, which causes three
compounding failures observed in paper trading:

1. **LLM guesses direction from context** — When the screener surfaces `compression_breakout`
   without indicating whether price is at the top or bottom of the compressed range, the LLM
   reaches for whichever direction its context predicts (often the prevailing trend), producing
   `trend_continuation` triggers inside a session that was supposed to be compression-themed.

2. **Targets go unset** — Without a directional mandate, the LLM has no structural anchor for
   a target price. The trigger compiler only rejects entries when a target anchor is declared but
   *fails to resolve*. If no target is declared at all, the entry goes through and the position
   is managed purely by exit triggers — no R-multiple framework, no partial-exit automation.

3. **Screener groupings are not actionable** — "Compression Breakout (5m)" describes a setup
   type and a hold timeframe but not a direction. An operator cannot form a position thesis from
   it. Adding direction makes each group a complete, actionable signal.

### Design principle

Direction is a first-class dimension alongside `hypothesis` and `timeframe`. Every screener
group, every recommendation item, every paper trading session, and every template is explicitly
long, short, or neutral.

---

## Scope

1. **`schemas/screener.py`** — `direction_bias` field on `SymbolAnomalyScore`,
   `InstrumentRecommendationItem`, and `InstrumentRecommendationGroup`
2. **`services/universe_screener_service.py`** — `price_position_in_range` metric,
   directional hypothesis routing, direction-aware grouping (adds direction to group key)
3. **`vector_store/strategies/`** — 6 new directional docs (split from ambiguous parents)
4. **`prompts/strategies/`** — 5 new directional prompt templates
5. **`vector_store/retriever.py`** — `allowed_identifiers_for_template()` updated for each
   directional template name
6. **`trading_core/trigger_compiler.py`** — direction-aware identifier enforcement +
   mandatory target requirement for directional templates
7. **`tools/paper_trading.py`** + **`ops_api/routers/paper_trading.py`** — thread
   `direction_bias` from screener candidate through session config into LLM prompt
8. **`ui/src/lib/api.ts`** + **`ui/src/components/PaperTradingControl.tsx`** — surface
   direction badge in screener group headers and candidate cards
9. **Tests** — screener direction routing, compiler enforcement, target requirement

## Out of Scope

- ML-trained direction prediction (price_position_in_range is purely deterministic)
- Reversal-specific entry refinement (covered by R56 activation modes)
- Short-selling infrastructure for live capital (paper trading only in this runbook)
- New screener scoring metrics beyond `price_position_in_range` (kept minimal)

---

## Hard Constraints (Non-Negotiable)

### 1. Direction is deterministic

`direction_bias` must be computed from observable OHLCV-derived metrics only — no LLM
involvement. The screener service assigns it; the LLM receives it as an instruction.

### 2. Backward compatibility

Existing sessions and the unqualified `compression_breakout` template_id must continue to work.
Directional template names are additive. Existing `template_file: compression_breakout` in the
vector store stays valid (treated as `compression_breakout_long` at the compiler layer).

### 3. Mandatory target for directional templates

An entry trigger inside a directional (long or short) template **must** declare either
`target_anchor_type` or a numeric `target_price_abs`. Absence is a compile-time rejection with
reason code `missing_target_for_directional_template`. This closes the "Not Set" gap
structurally, not at runtime.

### 4. No cross-direction identifiers

Long-biased templates must not contain identifiers that imply a short thesis and vice versa.
The trigger compiler enforces this via per-template allowed-identifier sets (R47 extension).

---

## `price_position_in_range` Computation

Add to `_score_symbol()` in `universe_screener_service.py`:

```python
# Donchian-based range position (0 = at range low, 1 = at range high)
lookback = min(20, len(df))
range_high = df["high"].rolling(lookback, min_periods=5).max().iloc[-1]
range_low  = df["low"].rolling(lookback, min_periods=5).min().iloc[-1]
span = range_high - range_low
price_position_in_range = (
    _clamp((close - range_low) / span, 0.0, 1.0) if span > 1e-9 else 0.5
)
```

Store in `score_components["price_position_in_range"]` and expose as a field on
`SymbolAnomalyScore`.

---

## Direction Routing Rules (Deterministic)

```python
def _candidate_direction(candidate, hypothesis) -> Literal["long", "short", "neutral"]:
    pos = candidate.price_position_in_range  # 0.0–1.0

    if hypothesis == "compression_breakout":
        # Compression at range top → expect downside break; at range bottom → expect upside
        if pos >= 0.65:   return "short"
        if pos <= 0.35:   return "long"
        return "neutral"  # middle of range — ambiguous direction, surface as neutral

    if hypothesis == "volatile_breakout":
        # Momentum direction follows where price has moved to in the range
        if pos >= 0.65:   return "long"   # price already near top, riding upside momentum
        if pos <= 0.35:   return "short"  # price near bottom, riding downside momentum
        return "neutral"

    if hypothesis == "range_mean_revert":
        # Only actionable at extremes
        if pos >= 0.70:   return "short"  # sell the rally at range top
        if pos <= 0.30:   return "long"   # buy the dip at range bottom
        return "neutral"  # middle — no edge, screener should route to uncertain_wait

    if hypothesis == "bull_trending":
        return "long"   # always

    if hypothesis == "bear_defensive":
        return "short"  # always

    return "neutral"  # uncertain_wait
```

---

## Directional Template Map

| Screener Hypothesis | Direction | template_id (new) | Prompt file (new) |
|---|---|---|---|
| `compression_breakout` | long | `compression_breakout_long` | `compression_breakout_long.txt` |
| `compression_breakout` | short | `compression_breakout_short` | `compression_breakout_short.txt` |
| `compression_breakout` | neutral | `compression_breakout` (existing) | `compression_breakout.txt` (existing) |
| `volatile_breakout` | long | `volatile_breakout_long` | `volatile_breakout_long.txt` |
| `volatile_breakout` | short | `volatile_breakout_short` | `volatile_breakout_short.txt` |
| `volatile_breakout` | neutral | `volatile_breakout` (existing) | (existing, no change) |
| `range_mean_revert` | long | `range_long` | `range_long.txt` |
| `range_mean_revert` | short | `range_short` | `range_short.txt` |
| `range_mean_revert` | neutral | → reroute to `uncertain_wait` | — |
| `bull_trending` | long | `bull_trending` (existing) | (existing, no change) |
| `bear_defensive` | short | `bear_defensive` | `bear_defensive.txt` (new) |

`compression_breakout_long` is functionally identical to `compression_breakout` — the existing
template covers it. The short variant is the new addition with direction-reversed identifiers.

---

## New Vector Store Documents

### `vector_store/strategies/compression_breakout_long.md`

```yaml
---
title: Compression Breakout Long
type: strategy
direction: long
regimes: [range, volatile]
tags: [breakout, compression, long]
identifiers: [compression_flag, bb_bandwidth_pct_rank, expansion_flag, breakout_confirmed,
              is_impulse_candle, is_inside_bar, vol_burst, donchian_upper_short,
              volume_multiple, atr_14, candle_strength, close]
template_file: compression_breakout_long
---
```

Context: Compression at or below range midpoint; expect upside expansion.
Entry: `compression_flag > 0.5` and `breakout_confirmed > 0.5` and `close > donchian_upper_short`.
Stop: below `donchian_lower_short` at entry bar (max 1.5× ATR).
Target: measured move (range width projected from breakout point) or `htf_daily_high`.
False-break exit: close returns inside range within 2 bars.

### `vector_store/strategies/compression_breakout_short.md`

```yaml
---
title: Compression Breakout Short
type: strategy
direction: short
regimes: [range, volatile]
tags: [breakout, compression, short, breakdown]
identifiers: [compression_flag, bb_bandwidth_pct_rank, expansion_flag, breakout_confirmed,
              is_impulse_candle, is_inside_bar, vol_burst, donchian_lower_short,
              volume_multiple, atr_14, candle_strength, close]
template_file: compression_breakout_short
---
```

Context: Compression at or above range midpoint; expect downside expansion (breakdown).
Entry: `compression_flag > 0.5` and `breakout_confirmed > 0.5` and `close < donchian_lower_short`.
Stop: above `donchian_upper_short` at entry bar (max 1.5× ATR).
Target: measured move downward or `htf_daily_low`.
False-break exit: close returns inside range within 2 bars.

### `vector_store/strategies/volatile_breakout_long.md`

```yaml
---
title: Volatile Breakout Long
type: strategy
direction: long
regimes: [volatile]
tags: [breakout, volatility, long, momentum]
identifiers: [close, atr_14, realized_vol_short, realized_vol_medium,
              bollinger_upper, vol_state, trend_state, candle_strength, volume_multiple]
template_file: volatile_breakout_long
---
```

Context: High-vol regime, price near/above range top — ride upside momentum.
Entry: `close > bollinger_upper` and `realized_vol_short` elevated and `trend_state` not "downtrend".
Stop: 1.5–2× ATR below entry (wide stops required for vol regime).
Target: 2× ATR measured move from entry, or `htf_daily_high`.

### `vector_store/strategies/volatile_breakout_short.md`

```yaml
---
title: Volatile Breakout Short
type: strategy
direction: short
regimes: [volatile]
tags: [breakout, volatility, short, breakdown]
identifiers: [close, atr_14, realized_vol_short, realized_vol_medium,
              bollinger_lower, vol_state, trend_state, candle_strength, volume_multiple]
template_file: volatile_breakout_short
---
```

Context: High-vol regime, price near/below range bottom — ride downside momentum.
Entry: `close < bollinger_lower` and `realized_vol_short` elevated and `trend_state` not "uptrend".
Stop: 1.5–2× ATR above entry.
Target: 2× ATR measured move downward, or `htf_daily_low`.

### `vector_store/strategies/range_long.md`

```yaml
---
title: Range Long (Buy the Dip)
type: strategy
direction: long
regimes: [range]
tags: [mean_reversion, range, long, support]
identifiers: [close, bollinger_lower, bollinger_middle, rsi_14, sma_medium,
              atr_14, vol_state, trend_state, candle_strength]
template_file: range_long
---
```

Context: Sideways regime, price at lower band — buy into support with mean-reversion target.
Entry: `close < bollinger_lower` and `rsi_14 < 38` and `vol_state` not "extreme".
Stop: 0.5× ATR below entry (tight — support level defines invalidation).
Target: `bollinger_middle` or `sma_medium` (mean reversion, not a trend trade).
Invalidation: close below stop, or `trend_state` flips to "downtrend".

### `vector_store/strategies/range_short.md`

```yaml
---
title: Range Short (Sell the Rally)
type: strategy
direction: short
regimes: [range]
tags: [mean_reversion, range, short, resistance]
identifiers: [close, bollinger_upper, bollinger_middle, rsi_14, sma_medium,
              atr_14, vol_state, trend_state, candle_strength]
template_file: range_short
---
```

Context: Sideways regime, price at upper band — sell into resistance with mean-reversion target.
Entry: `close > bollinger_upper` and `rsi_14 > 62` and `vol_state` not "extreme".
Stop: 0.5× ATR above entry.
Target: `bollinger_middle` or `sma_medium`.
Invalidation: close above stop, or `trend_state` flips to "uptrend".

---

## `schemas/screener.py` Changes

### `SymbolAnomalyScore`

Add:
```python
price_position_in_range: float = Field(default=0.5, ge=0.0, le=1.0)
    # 0 = at 20-bar Donchian low, 1 = at high
direction_bias: Literal["long", "short", "neutral"] = "neutral"
```

### `InstrumentRecommendationItem`

Add:
```python
direction_bias: Literal["long", "short", "neutral"] = "neutral"
```

### `InstrumentRecommendationGroup`

Add:
```python
direction_bias: Literal["long", "short", "neutral"] = "neutral"
```

Update `label` to include direction, e.g. `"Compression Breakout Short (5m)"`.

---

## `services/universe_screener_service.py` Changes

### `_score_symbol()`

Compute `price_position_in_range` and `direction_bias` via `_candidate_direction()`.
Store in `score_components["price_position_in_range"]` and `score_components["direction_bias"]`.
Populate on returned `SymbolAnomalyScore`.

### `_candidate_template_id()`

Return the directional template name when direction is long or short:

```python
def _candidate_template_id(self, candidate) -> str | None:
    base_hint = candidate.score_components.get("template_id_suggestion")
    direction = candidate.direction_bias or "neutral"
    if base_hint == "compression_breakout":
        if direction == "short": return "compression_breakout_short"
        if direction == "long":  return "compression_breakout_long"
    if base_hint in {"volatile_breakout"}:
        if direction == "long":  return "volatile_breakout_long"
        if direction == "short": return "volatile_breakout_short"
    if base_hint == "range_mean_revert":  # resolved via hypothesis routing
        if direction == "long":  return "range_long"
        if direction == "short": return "range_short"
    return str(base_hint) if base_hint else None
```

### `build_recommendation_batch()`

Group key becomes `(hypothesis, direction, timeframe)` instead of `(hypothesis, timeframe)`:

```python
direction = self._candidate_direction(candidate, hypothesis)
key = (hypothesis, direction, timeframe)
```

Propagate `direction_bias` to `InstrumentRecommendationItem` and `InstrumentRecommendationGroup`.

### `_group_label()`

```python
direction_label = {"long": " ↑ Long", "short": " ↓ Short", "neutral": ""}
return f"{label_map.get(hypothesis, hypothesis)}{direction_label[direction]} ({timeframe})"
```

---

## `vector_store/retriever.py` Changes

Extend `allowed_identifiers_for_template()` with the 6 new directional template IDs, each
mapping to the appropriate direction-specific identifier set (subset/superset of the neutral
parent's identifiers):

```python
"compression_breakout_long":  COMPRESSION_LONG_IDENTIFIERS,
"compression_breakout_short": COMPRESSION_SHORT_IDENTIFIERS,
"volatile_breakout_long":     VOLATILE_LONG_IDENTIFIERS,
"volatile_breakout_short":    VOLATILE_SHORT_IDENTIFIERS,
"range_long":                 RANGE_LONG_IDENTIFIERS,
"range_short":                RANGE_SHORT_IDENTIFIERS,
```

Long/short variants share most neutral identifiers except the direction-discriminating ones:
- Long templates include `donchian_upper_short`, `bollinger_upper`, direction-up candlesticks
- Short templates include `donchian_lower_short`, `bollinger_lower`, direction-down candlesticks

---

## `trading_core/trigger_compiler.py` Changes

### Direction-aware identifier enforcement (new rule)

Add `_get_template_direction(template_id) -> str` that returns `"long" | "short" | "neutral"`.

For directional templates, `enforce_template_identifiers()` additionally checks that no
identifier is semantically cross-direction (e.g., `close < donchian_lower_short` in a long
template, or `close > donchian_upper_short` in a short template).

Emit new `ViolationCode`: `cross_direction_identifier`.

### Mandatory target for directional templates (new rule)

Add `enforce_directional_target_requirement(plan, template_id) -> list[PlanViolation]`:

```python
direction = _get_template_direction(template_id)
if direction not in {"long", "short"}:
    return []
for trigger in plan.triggers:
    if trigger.get("intent") not in {"entry", "entry_add"}:
        continue
    has_target = (
        trigger.get("target_anchor_type")
        or trigger.get("target_price_abs") is not None
    )
    if not has_target:
        violations.append(PlanViolation(
            trigger_id=trigger.get("trigger_id", "?"),
            violation_type="missing_target_for_directional_template",
            detail=f"Entry trigger in {template_id} template must declare target_anchor_type "
                   f"or target_price_abs. Received neither.",
        ))
return violations
```

Wire into `enforce_plan_quality()` (called after `enforce_template_identifiers()`).

---

## Paper Trading Threading

### `ops_api/routers/paper_trading.py`

Add `direction_bias: str = "neutral"` to `PaperTradingSessionConfig`.

### `tools/paper_trading.py`

Add `direction_bias: str = "neutral"` to `PaperTradingConfig` and `SessionState`.

In `generate_strategy_plan_activity()`, inject the direction hint into the LLM prompt after
the timeframe hint:

```python
if direction_bias and direction_bias != "neutral":
    direction_str = "LONG (upside breakout / buy)" if direction_bias == "long" else "SHORT (downside break / sell)"
    prompt_extra += f"\nDIRECTION: {direction_str}. All entry triggers must align with this direction."
    prompt_extra += f"\nREQUIRED: Every entry trigger must specify a target_anchor_type or numeric target_price_abs."
```

### `ui/src/lib/api.ts`

Add `direction_bias?: string` to `PaperTradingSessionConfig` interface.

### `ui/src/components/PaperTradingControl.tsx`

In `applyScreenerCandidateToForm()`, thread `item.direction_bias` into `directionBias` state
alongside `indicatorTimeframe`.

In screener group headers, add direction badge:
- `↑ Long` (green), `↓ Short` (red), or nothing for neutral.

---

## Implementation Steps

### Step 1: Add `price_position_in_range` and `direction_bias` to screener schemas

Update `schemas/screener.py` — 3 model changes, all backward compatible (Optional/default fields).

### Step 2: Compute direction in screener service

- Add `_compute_price_position_in_range(df)` helper
- Add `_candidate_direction(candidate, hypothesis)` routing method
- Update `_score_symbol()` to compute and store both
- Update `_candidate_template_id()` for directional resolution
- Update `build_recommendation_batch()` grouping key to include direction
- Update `_group_label()` to include direction
- Update `InstrumentRecommendationGroup` population to carry `direction_bias`

### Step 3: Write 6 directional vector store docs

Create `vector_store/strategies/`:
- `compression_breakout_long.md`
- `compression_breakout_short.md`
- `volatile_breakout_long.md`
- `volatile_breakout_short.md`
- `range_long.md`
- `range_short.md`

### Step 4: Write 5 directional prompt templates

Create `prompts/strategies/`:
- `compression_breakout_long.txt` (can be copy/refine of `compression_breakout.txt` with long-explicit language)
- `compression_breakout_short.txt` (mirror, short-explicit)
- `volatile_breakout_long.txt`
- `volatile_breakout_short.txt`
- `range_long.txt`
- `range_short.txt`
- `bear_defensive.txt` (was missing; `bear_defensive` hypothesis needs a prompt)

Each template must explicitly state the direction and mandate a target in its instructions.

### Step 5: Update `vector_store/retriever.py` allowed identifier sets

Extend `allowed_identifiers_for_template()` with the 6 new template IDs.

### Step 6: Extend trigger compiler

- `_get_template_direction(template_id)` function
- `enforce_directional_target_requirement()` function
- Wire into `enforce_plan_quality()`
- Add `cross_direction_identifier` and `missing_target_for_directional_template` violation types

### Step 7: Thread direction through paper trading

- Schema: `direction_bias` on `PaperTradingSessionConfig`, `PaperTradingConfig`, `SessionState`
- Activity: direction hint + target mandate injected into LLM prompt
- Ops API router: `direction_bias` forwarded from request to workflow config
- UI: `direction_bias` extracted from screener item, sent to session start, direction badge shown

### Step 8: Tests

```bash
# Screener direction routing (price_position_in_range → direction_bias)
uv run pytest tests/test_universe_screener.py -vv -k direction

# Trigger compiler: mandatory target for directional templates
uv run pytest tests/test_trigger_compiler.py -vv -k directional_target

# Trigger compiler: cross-direction identifier rejection
uv run pytest tests/test_trigger_compiler.py -vv -k cross_direction

# Paper trading: direction_bias threaded to session config
uv run pytest tests/test_screener_timeframe_threading.py -vv
```

---

## Acceptance Criteria

- [ ] `price_position_in_range` (0–1) computed deterministically from 20-bar Donchian channel in screener scoring
- [ ] `direction_bias: "long" | "short" | "neutral"` on `SymbolAnomalyScore`, `InstrumentRecommendationItem`, and `InstrumentRecommendationGroup`
- [ ] Screener groups include direction in group key and label (e.g. "Compression Breakout ↓ Short (5m)")
- [ ] Directional `template_id` (e.g. `compression_breakout_short`) returned from screener recommendation items
- [ ] 6 new vector store strategy docs covering directional variants of 3 hypotheses
- [ ] 6+ new directional prompt template files; each explicitly states direction and mandates target
- [ ] `allowed_identifiers_for_template()` updated for all 6 new directional template IDs
- [ ] Trigger compiler rejects entry triggers with no target in directional templates (`missing_target_for_directional_template`)
- [ ] Trigger compiler rejects cross-direction identifiers in directional templates (`cross_direction_identifier`)
- [ ] `direction_bias` threaded from screener candidate → session config → LLM prompt injected hint
- [ ] Direction badge visible in screener group headers in UI
- [ ] `bear_defensive.txt` prompt template created (was missing)
- [ ] All existing tests pass (backward-compatible: `compression_breakout` still works as neutral/long alias)
- [ ] `range_mean_revert` candidates with `neutral` direction are rerouted to `uncertain_wait` group (no trade)

---

## Test Plan

```bash
# Schema + screener direction computation
uv run pytest tests/test_universe_screener.py -vv

# Direction routing: price_position_in_range → direction_bias per hypothesis
uv run pytest tests/test_universe_screener.py -vv -k "direction"

# Trigger compiler: mandatory target + cross-direction rejection
uv run pytest tests/test_trigger_compiler.py -vv -k "directional"

# Paper trading config threading
uv run pytest tests/test_screener_timeframe_threading.py -vv

# Full suite regression
uv run pytest --ignore=tests/test_agent_workflows.py --ignore=tests/test_metrics_tools.py -q
```

---

## Human Verification Evidence

```text
(To be filled after implementation)
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-25 | Runbook created — directional bias templates for long/short across all hypotheses | Claude |

## Worktree Setup

```bash
git worktree add -b feat/directional-bias-templates ../wt-directional-templates main
cd ../wt-directional-templates
```

## Git Workflow

```bash
git add \
  schemas/screener.py \
  services/universe_screener_service.py \
  vector_store/strategies/compression_breakout_long.md \
  vector_store/strategies/compression_breakout_short.md \
  vector_store/strategies/volatile_breakout_long.md \
  vector_store/strategies/volatile_breakout_short.md \
  vector_store/strategies/range_long.md \
  vector_store/strategies/range_short.md \
  prompts/strategies/compression_breakout_long.txt \
  prompts/strategies/compression_breakout_short.txt \
  prompts/strategies/volatile_breakout_long.txt \
  prompts/strategies/volatile_breakout_short.txt \
  prompts/strategies/range_long.txt \
  prompts/strategies/range_short.txt \
  prompts/strategies/bear_defensive.txt \
  vector_store/retriever.py \
  trading_core/trigger_compiler.py \
  tools/paper_trading.py \
  ops_api/routers/paper_trading.py \
  ui/src/lib/api.ts \
  ui/src/components/PaperTradingControl.tsx \
  tests/test_universe_screener.py \
  tests/test_trigger_compiler.py \
  docs/branching/59-directional-bias-templates.md \
  docs/branching/README.md

git commit -m "feat: add directional bias templates — long/short across all screener hypotheses (Runbook 59)"
```

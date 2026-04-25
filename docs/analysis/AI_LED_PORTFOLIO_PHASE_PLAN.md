# AI-Led Portfolio — Full Phase Plan

_Last updated: 2026-04-24. Supersedes runbook order in `AI_LED_PORTFOLIO_SESSION_PLAN_V0.md`._

---

## Status Summary

| Phase | Runbooks | Status |
|-------|----------|--------|
| 0 — Instrumentation & invariants | R69–R73 | ✅ Complete |
| 1 — Opportunity scanner | R74, R75 | ✅ Complete (2026-03-13) |
| 2 — AI portfolio planner | R76 | ✅ Complete — SessionIntent + use_ai_planner wired |
| 3a — Round-trip hypothesis model | R78 | ✅ Complete (2026-03-13) |
| 3b — Hypothesis synthesis + cadence | R77 | ✅ Complete — CadenceGovernor wired |
| 4 — Hypothesis feedback loop | R79 | ✅ Complete — attribution + playbook evidence |
| 4+ — Trailing stops + risk management | R80–R85 | ✅ Complete (5 trailing modes, CaN size fix) |
| **5 — Prompt hygiene** | Audit phases 1–4 | ✅ Complete (2026-04-24) — dedup fix, cohort telemetry, breakout templates refreshed |
| **6 — SOTA foundation** | R88, R90 | **Next up** |
| **7 — Memory closure** | R91, R93, R97 | Queued — depends on Phase 6 |
| **8 — Generation quality** | R89, R92 | Queued — expensive; validate Phase 6/7 first |
| **9 — Operational improvements** | R94, R95 | Queued — can run parallel with Phase 8 |
| **10 — Trigger architecture** | R96 | Queued — highest architectural lift; last |

**Dependency order (full, revised):**
```
[Phases 0–4+: ✅ Complete]
       ↓
  Phase 5 (prompt hygiene — prerequisite for all prompt-touching work)
       ↓
  Phase 6: R88 (scratchpad) + R90 (hallucination scorer)
       ↓
  Phase 7: R91 (reflexion) + R93 (uncertainty) + R97 (logprob)
       ↓
  Phase 8: R89 (best-of-N) ─── R92 (debate)          Phase 9: R94 ─── R95 (parallel with Phase 8)
       ↓
  Phase 10: R96 (trigger catalog — depends on all prior prompt/plan stability)
```

**Companion workstream:**
`docs/analysis/TRADE_LIFECYCLE_REPLAY_ALIGNMENT_PLAN.md` is a cross-cutting
support track for lifecycle identity, replay, and pattern retrieval. It does
not change the phase order above. It starts after Phase 5 prompt hygiene, then
supports Phase 7 memory closure, Phase 9 operator transparency, and Phase 10
trigger architecture.

**Legacy dependency order (Phases 0–4, for reference):**
```
[Phase 0 re-run gate]
       ↓
  R74 ─┬─ R75 (parallel)
       ↓
  R78  (TradeHypothesis schema + executor)
       ↓
  R76  (SessionIntent wired to hypotheses)
       ↓
  R77  (Cadence governor + hypothesis synthesis)
       ↓
  R79  (Feedback loop)
```

---

## Phase 0 — Controlled Re-run Protocol

R69–R73 are committed. Before Phase 1 can begin:

1. Start 4 parallel sessions (same configs as 2026-03-13 audit).
2. Let run for 2h minimum.
3. For each session: `GET /paper-trading/sessions/{id}/block-summary`
4. **Pass criteria (all must be true):**
   - `block_class_counts.quality_gate == 0` (no `no_target_rr_undefined` or `target_price_unresolvable`)
   - `block_class_counts.infra == 0` (no price feed timeouts killing sessions)
   - Zero workflows terminated from `ScheduleToClose`
   - Zero zero-trigger stand-downs without a `plan_construction_failed` structured event
5. Phase 1 gate opens only when all 4 criteria pass.

---

## Phase 1 — Opportunity Scanner

### R74 — `OpportunityCard` Scorer and Scanner Service

**Goal:** Score every symbol in the universe every 5–15 min. Expose a stable ranked top-10.

**Schema** (`schemas/opportunity.py`, new file):

```python
class OpportunityCard(BaseModel):
    symbol: str
    opportunity_score: float        # [-0.10, 0.90] raw weighted score
    opportunity_score_norm: float   # [0, 1] clamped for consumers
    vol_edge: float                 # ATR expansion vs 20-bar mean
    structure_edge: float           # level count × clarity (structure engine)
    trend_edge: float               # EMA alignment + ADX
    liquidity_score: float          # volume_ratio from indicator snapshot
    spread_penalty: float           # bid/ask spread pct
    instability_penalty: float      # consecutive_price_failures / 3 clamped to [0,1]
    expected_hold_horizon: Literal["scalp", "intraday", "swing"]
    scored_at: datetime
    indicator_as_of: datetime       # freshness check
    component_explanation: Dict[str, str]  # human-readable why each score is what it is

class OpportunityRanking(BaseModel):
    ranked_at: datetime
    cards: List[OpportunityCard]    # sorted by opportunity_score_norm desc
    universe_size: int
    scan_duration_ms: int
```

**Score formula (from V0 plan):**
```
opportunity_score = 0.28*vol_edge + 0.24*structure_edge + 0.18*trend_edge
                  + 0.20*liquidity_score - 0.07*spread_penalty - 0.03*instability_penalty
opportunity_score_norm = clamp((opportunity_score + 0.10) / 1.00, 0, 1)
```

**Service** (`services/opportunity_scanner.py`, new file):

- `score_symbol(symbol, indicator, structure_snapshot, ticker) → OpportunityCard`
  - `vol_edge`: `atr / atr_20` ratio clamped to [0,1]; >1.5 = high expansion
  - `structure_edge`: `len(snapshot.levels) / 10` clamped to [0,1]; bonus for levels near price
  - `trend_edge`: EMA alignment score from `htf_price_vs_daily_mid` + `adx_14` / 50
  - `liquidity_score`: `volume_ratio` from indicator snapshot (volume / 20-bar mean volume)
  - `spread_penalty`: `(ask - bid) / mid` from ticker; 0 if unavailable
  - `instability_penalty`: `consecutive_price_failures / 3` from SessionState
  - `expected_hold_horizon`: scalp if ATR contraction; swing if trend + structure; else intraday
- `rank_universe(symbols, indicator_snapshots, structure_snapshots, tickers) → OpportunityRanking`
  - Calls `score_symbol` per symbol, sorts by `opportunity_score_norm` desc

**Integration into `PaperTradingWorkflow`:**
- Add `_last_opportunity_ranking: OpportunityRanking | None` to `__init__`
- After `fetch_indicator_snapshots_activity` in `_generate_plan()`, call scanner (non-fatal)
- Emit `opportunity_ranking` event with top-10 cards
- Store ranking for `_generate_plan()` to inject into LLM context as `OPPORTUNITY_CONTEXT` block

**New activity** `score_opportunities_activity(symbols, indicator_snapshots, structure_snapshots)`:
- Calls `rank_universe`; returns `OpportunityRanking.model_dump()`

**Ops API endpoint** (`ops_api/routers/scanner.py`, new router):
- `GET /scanner/opportunities` → current `OpportunityRanking`
- `GET /scanner/opportunities/{symbol}` → single `OpportunityCard` with full breakdown
- `GET /scanner/history?limit=20` → recent rankings (last N from event store)

**Exit criteria:**
- Top-10 updates every 5–15 min without CPU overload
- All 6 component scores visible per card in API
- `opportunity_score_norm` correctly ranges [0, 1] across all symbols

---

### R75 — Scanner UI Panel

**Goal:** Live opportunity ranking visible in React UI alongside session panel.

**Changes:**
- `ops_api/routers/scanner.py` — router registered in `ops_api/app.py`
- `ui/src/components/ScannerPanel.tsx` — new component:
  - Table: rank, symbol, score bar, vol_edge, structure_edge, trend_edge, hold horizon
  - Color coding: green > 0.6, amber 0.4–0.6, grey < 0.4
  - Click → expands `component_explanation` breakdown
  - Refresh every 60s (not 30s — scanner is slower than tick)
- `ui/src/components/PaperTradingControl.tsx` — add scanner panel section below session config

**Exit criteria:** Scanner panel renders, updates live, shows component breakdown on expand.

---

## Phase 2 — AI Portfolio Planner

### R76 — `SessionIntent` + Symbol/Risk Allocation

**Goal:** Session starts with AI-selected symbols and explicit risk map. Remove user-selected
`strategy_prompt` as required input (keep as optional override).

**Schema** (`schemas/session_intent.py`, new file):

```python
class SymbolIntent(BaseModel):
    symbol: str
    opportunity_score_norm: float       # from scanner
    risk_budget_fraction: float         # fraction of session risk (sum across symbols ≤ 1.0)
    playbook_id: str | None             # derived from regime + structure, not user-selected
    thesis_summary: str                 # 1-2 sentence AI-generated basis
    expected_hold_horizon: str          # from OpportunityCard
    direction_bias: Literal["long", "short", "neutral"]

class SessionIntent(BaseModel):
    intent_id: str
    created_at: datetime
    selected_symbols: List[str]         # 5–10, ranked by opportunity_score
    planned_trade_cadence_min: int = 5  # target round trips per 24h
    planned_trade_cadence_max: int = 10
    symbol_intents: List[SymbolIntent]
    total_risk_budget_fraction: float = 1.0
    regime_summary: str                 # overall market regime at intent creation
    planner_rationale: str              # why these symbols were chosen
    use_ai_planner: bool = True
```

**New service** `services/session_planner.py`:
- `generate_session_intent(opportunity_ranking, portfolio_state, session_config) → SessionIntent`
- LLM call: input = top-10 `OpportunityCard`s + portfolio state + risk constraints
- Output: `SessionIntent` with symbol selection and risk allocation
- Playbook assignment: derived from `PlaybookRegistry.list_eligible(regime)` per symbol
- Falls back to user-specified symbols + neutral allocation if LLM fails

**Integration into `PaperTradingWorkflow`:**
- Add `use_ai_planner: bool = False` to `PaperTradingConfig` (feature flag for safe rollout)
- Add `session_intent: Optional[Dict] = None` to `SessionState`
- At session start (first `_generate_plan` call), if `use_ai_planner`:
  - Run scanner → generate `SessionIntent`
  - Override `self.symbols` with `intent.selected_symbols`
  - Store `session_intent` on `SessionState`
  - Emit `session_intent_generated` event
- `SessionIntent` injected into LLM context for plan generation

**Changes to `generate_strategy_plan_activity`:**
- Accept optional `session_intent_dict` parameter
- When present: inject `SESSION_INTENT` block into prompt
  (selected symbols, per-symbol thesis, playbook hints)
- `strategy_prompt` becomes optional when `session_intent` is present

**Exit criteria:**
- Session with `use_ai_planner=True` starts with AI-selected symbols
- `session_intent_generated` event emitted with full symbol/risk breakdown
- `strategy_prompt=None` sessions still work when intent is present

---

## Phase 3a — Round-Trip Hypothesis Model (R78)

**This is the foundational architectural change. Must precede R77.**

### R78 — `TradeHypothesis` Schema + `HypothesisExecutor`

**Goal:** Replace "trigger soup" with one coherent round-trip unit per symbol.
The LLM generates hypotheses, not trigger bags. Entry, stop, and target
are properties of the same object. Exit triggers as a separate class disappear.

**Problem with current model:**
- LLM generates 4–8 disconnected triggers per symbol
- Entry trigger has no guaranteed link to its exit trigger
- `emergency_exit`, `risk_off`, `risk_reduce` triggers proliferate and mostly evaluate against no position
- `no_target_rr_undefined` and `target_price_unresolvable` exist because stops/targets are
  late-resolved at fire time, not required at plan creation time
- Zero trades despite many triggers because none of the exit-class triggers know which
  entry opened the position they're supposed to close

**New model:**

#### `schemas/hypothesis.py` (new file)

```python
class StructureBasis(BaseModel):
    """The structural context that motivates the hypothesis."""
    key_support: float | None           # price level acting as support anchor
    key_resistance: float | None        # price level acting as resistance anchor
    stop_anchor_level: float            # absolute price for stop (required)
    target_anchor_level: float          # absolute price for target (required)
    stop_anchor_source: str             # e.g. "swing_low_03-12", "htf_daily_low"
    target_anchor_source: str           # e.g. "swing_high_03-10", "r_multiple_2"
    structure_notes: str | None         # plain-text description of the structure

class TradeHypothesis(BaseModel):
    """A complete round-trip trade specification. The atomic unit of the new model.

    Replaces the 3–5 disconnected TriggerConditions that previously encoded
    one trade idea across entry + emergency_exit + risk_off + risk_reduce triggers.
    """
    id: str = Field(default_factory=lambda: f"hyp_{uuid4().hex[:8]}")
    symbol: str
    timeframe: str
    direction: Literal["long", "short"]
    confidence_grade: Literal["A", "B", "C"] = "B"
    playbook_id: str | None = None

    # ── Thesis ────────────────────────────────────────────────────────────
    thesis: str                         # plain text: "POL forming compression below $0.098..."
    regime_context: str | None          # market regime at hypothesis creation
    indicator_basis: str                # which indicators motivated this: "RSI=42, ATR contracting"

    # ── Structure ─────────────────────────────────────────────────────────
    structure: StructureBasis

    # ── Entry ─────────────────────────────────────────────────────────────
    entry_rule: str                     # DSL expression, same evaluator as TriggerCondition
    entry_notes: str | None             # optional rationale for entry timing

    # ── Exit (inline — no separate exit trigger needed) ───────────────────
    # Stop and target are REQUIRED at plan time, validated before acceptance.
    # The executor watches them price-tick by price-tick (same as existing
    # _sweep_stop_target), not via exit_rule evaluation.
    stop_price: float                   # absolute stop level (from structure.stop_anchor_level)
    target_price: float                 # absolute target level (from structure.target_anchor_level)
    stop_loss_pct: float | None         # fallback if structural stop not resolvable at runtime
    trailing_stop_activation_r: float | None  # activate trailing stop at N×R (None = no trail)

    # ── Invalidation ──────────────────────────────────────────────────────
    # Thesis invalidation: conditions that make the setup stale even if stop not hit.
    # Separate from stop — stop is execution, invalidation is intellectual.
    invalidation_rule: str | None       # DSL expression; fires → close position + record WHY
    invalidation_horizon_bars: int = 12 # max bars before hypothesis expires regardless

    # ── Derived (computed by compiler, not LLM-authored) ──────────────────
    rr_ratio: float | None = None       # (target - entry) / (entry - stop), filled at compile
    estimated_bars_to_resolution: int | None = None  # LLM forecast for attribution

    class Config:
        extra = "forbid"
```

#### Updates to `schemas/llm_strategist.py`

Add `hypotheses` list to `StrategyPlan` alongside existing `triggers` (backward-compat):

```python
class StrategyPlan(SerializableModel):
    ...
    triggers: List[TriggerCondition] = Field(default_factory=list)  # deprecated path
    hypotheses: List[TradeHypothesis] = Field(default_factory=list) # new path
    ...
```

When `hypotheses` is non-empty, the executor uses hypotheses. When empty, falls back to triggers.
This allows gradual migration without breaking existing sessions.

#### `services/hypothesis_compiler.py` (new file)

```python
class HypothesisCompileResult:
    valid: List[TradeHypothesis]
    invalid: List[dict]  # {hypothesis_id, reason, detail}

def compile_hypotheses(
    hypotheses: List[TradeHypothesis],
    indicator: IndicatorSnapshot,
    current_price: float,
) -> HypothesisCompileResult:
    """Validate and enrich hypotheses at plan-compile time.

    For each hypothesis:
    1. Verify stop_price and target_price are set and directionally correct.
    2. Compute rr_ratio = abs(target - current) / abs(current - stop).
    3. Verify rr_ratio >= min_rr_ratio (default 1.2).
    4. Verify stop is on the correct side (long: stop < price, short: stop > price).
    5. Verify target is on the correct side (long: target > price, short: target < price).
    6. If stop_price is missing: attempt to resolve from stop_anchor_source via indicator.
    7. If target_price is missing: attempt to resolve from target_anchor_source via indicator.

    Returns only valid hypotheses. Invalid ones emit compile_time_hypothesis_violation events.
    """
```

#### `services/hypothesis_executor.py` (new file)

```python
class OpenHypothesisState(BaseModel):
    """Runtime state for a hypothesis that has entered (position is open)."""
    hypothesis_id: str
    symbol: str
    direction: Literal["long", "short"]
    entry_price: float
    entry_bar_index: int
    stop_price: float
    target_price: float
    trailing_stop_price: float | None
    bars_held: int = 0
    invalidated: bool = False
    invalidation_reason: str | None = None
    hypothesis_snapshot: Dict[str, Any]  # full hypothesis dict for attribution

class HypothesisExecutor:
    """Manages open hypotheses for one symbol.

    One hypothesis open at a time per symbol (no concurrent same-symbol positions).
    Replaces the exit trigger evaluation path for hypothesis-mode plans.

    Per-tick: check if stop or target has been crossed (price-based, fast path).
    Per-bar: evaluate entry_rule for pending hypotheses; check invalidation_rule for open ones.
    """

    def on_tick(self, symbol, current_price) -> Order | None:
        """Fast path: stop/target sweep. Returns exit order if crossed."""

    def on_bar(self, bar, indicator, portfolio) -> tuple[Order | None, List[dict]]:
        """Entry evaluation for pending hypotheses.
        Invalidation evaluation for open hypotheses."""

    def on_fill(self, symbol, fill_price, direction):
        """Record entry fill → move hypothesis from pending to open."""

    def on_exit(self, symbol, exit_price, exit_reason):
        """Record exit → move hypothesis to completed, emit attribution record."""

    def get_open_state(self, symbol) -> OpenHypothesisState | None:
        ...
```

#### Integration into `PaperTradingWorkflow`

- Add `self._hypothesis_executor: HypothesisExecutor` to `__init__`
- In `_evaluate_and_execute`:
  - **Tier 1 (tick):** if plan has hypotheses, call `executor.on_tick()` alongside `_sweep_stop_target`
  - **Tier 2 (bar):** if plan has hypotheses, call `executor.on_bar()` alongside `TriggerEngine.on_bar()`
  - When `len(plan.hypotheses) > 0`, suppress exit trigger categories from TriggerEngine
    (`emergency_exit`, `risk_off`, `risk_reduce`) — executor owns exits

#### LLM Prompt Changes

`generate_strategy_plan_activity`:
- When `use_hypothesis_model=True` (new env flag: `PAPER_TRADING_USE_HYPOTHESIS_MODEL`, default `False`):
  - Change schema block to request `hypotheses[]` instead of `triggers[]`
  - New prompt template: `prompts/hypothesis_plan.txt`
  - Required fields: `thesis`, `entry_rule`, `stop_price`, `target_price`, `structure.stop_anchor_source`, `structure.target_anchor_source`
  - Prohibited fields in output: exit-class triggers (`direction: flat/exit`)
  - Compile `HypothesisCompileResult` after generation; repair if violations

#### Deprecation path for exit trigger types

When hypothesis model is active for a symbol:
- `TriggerEngine` filters out all `direction in {flat, exit, flat_exit}` triggers for that symbol
- `category in {emergency_exit, risk_off, risk_reduce}` also filtered — executor owns these
- Only `direction in {long, short}` entry triggers pass through (for mixed-mode sessions)

**Exit criteria:**
- Single symbol session with `PAPER_TRADING_USE_HYPOTHESIS_MODEL=true` produces:
  - Plan has `hypotheses[]` list, not just `triggers[]`
  - `stop_price` and `target_price` are set on every hypothesis at plan time
  - `rr_ratio` ≥ 1.2 on every compiled hypothesis
  - Position opens on `entry_rule` fire
  - Position closes when price crosses `stop_price` (loss) or `target_price` (win)
  - No `no_target_rr_undefined` blocks
  - No `priority_skip` or `priority_skip_invalid_candidate` blocks (only 1 hypothesis per symbol)
- Existing trigger-mode sessions unaffected (flag off by default)

---

## Phase 3b — Hypothesis Synthesis + Cadence Governor (R77, reframed)

**Prerequisite: R78 complete and validated on single-symbol session.**

### R77 — Hypothesis Generation from `SessionIntent` + `CadenceGovernor`

**Goal:** Plans generated from `SessionIntent.symbol_intents`, not user template.
Cadence governor tracks round-trip completion and adjusts symbol breadth to hit 5–10/day.

#### Hypothesis Generation from Intent

`generate_strategy_plan_activity` changes:
- When `session_intent` is present AND `use_hypothesis_model=True`:
  - One `TradeHypothesis` generated per symbol in `session_intent.selected_symbols`
  - `playbook_id` taken from `SymbolIntent.playbook_id` (not LLM-selected)
  - `direction` from `SymbolIntent.direction_bias`
  - `thesis_summary` from `SymbolIntent.thesis_summary` seeded into prompt
  - `structure` resolved from `StructureSnapshot` for that symbol
  - LLM fills: `entry_rule`, `invalidation_rule`, `estimated_bars_to_resolution`
  - Compiler fills: `stop_price`, `target_price`, `rr_ratio` from structure levels

#### `services/cadence_governor.py` (new file)

```python
class CadenceGovernorState(BaseModel):
    round_trips_completed: int = 0
    round_trips_target_min: int = 5
    round_trips_target_max: int = 10
    session_start: datetime
    hours_elapsed: float
    projected_24h_rate: float           # extrapolated from current pace
    cadence_status: Literal["on_track", "below_target", "above_target"]
    last_adaptation: str | None         # what the governor last changed

class CadenceGovernor:
    """Tracks round-trip completion and suggests breadth adjustments.

    Hard invariant: NEVER lowers quality gates to hit cadence targets.
    Only permitted adaptations:
    - Increase symbol breadth (add more symbols from scanner ranking)
    - Reallocate risk budgets (smaller per-trade, more symbols)
    - Reduce opportunity_score cutoff threshold (within quality floor)

    Prohibited adaptations (enforced by assertion):
    - Reducing min R:R below configured floor
    - Permitting hypotheses without stop_price or target_price
    - Bypassing compile-time invariants
    """

    def record_round_trip_complete(self, symbol, outcome): ...

    def get_adaptation_recommendation(self) -> dict:
        """Returns {'action': 'widen_breadth' | 'throttle' | 'hold', 'reason': str}"""

    def should_widen_symbol_set(self) -> bool:
        """True when pace is below target and session has > 4h remaining."""
```

#### Integration into `PaperTradingWorkflow`

- Add `cadence_governor_state` to `SessionState`
- On fill (position opened or closed): call `governor.record_round_trip_complete()`
- On `_generate_plan()`: call `governor.get_adaptation_recommendation()`
  - If `widen_breadth`: add next N symbols from last opportunity ranking
  - If `throttle`: reduce `total_risk_budget_fraction` for next plan interval

**Exit criteria:**
- Governor state visible in session queries
- Controlled 8h session achieves 4–12 round trips
- Governor never lowers `min_rr_ratio` or removes stop/target requirements
- `cadence_status` emitted in plan events

---

## Phase 4 — Hypothesis Feedback Loop (R79)

**Prerequisite: R77 complete, round trips being recorded.**

### R79 — Trade Attribution Updates Scanner and Planner Priors

**Goal:** Each completed round trip updates the scanner's prior for that symbol/playbook,
so future sessions prefer symbols and playbooks with positive attribution.

#### `HypothesisOutcomeRecord` → extends `EpisodeMemoryRecord`

Add to episode record:
- `hypothesis_id`
- `thesis_text`
- `stop_hit: bool`
- `target_hit: bool`
- `invalidation_hit: bool`
- `attribution_tags: List[str]` — e.g. `["model_quality", "regime_mismatch", "timing_bias"]`
- `r_multiple_realized: float`
- `bars_held: int`
- `expected_bars: int | None`
- `timing_accuracy: float | None` — `expected / actual` ratio

#### Attribution engine (`services/hypothesis_attribution.py`, new file)

```python
def classify_outcome(record: HypothesisOutcomeRecord) -> List[str]:
    """Tag the round trip with attribution labels.

    - stop_hit + bars_held < 3: "premature_stop" (too tight?)
    - target_hit + r_multiple < 1.0: "target_too_close"
    - invalidation_hit before stop/target: "regime_mismatch"
    - bars_held >> expected_bars: "timing_bias" (entry too early/late)
    - r_multiple > 2.0 + target_hit: "model_quality_strong"
    """
```

#### Scanner prior adjustment

`OpportunityCard` scorer gains:
- `playbook_win_rate: float | None` — from last 20 episodes for this symbol/playbook
- `playbook_r_expectancy: float | None` — mean r_multiple from episodes
- Score formula gain: `+0.05 * clamp(playbook_r_expectancy, 0, 2) / 2`
  (bonus for symbols with strong historical expectancy; capped to prevent overfitting)

#### Feedback write path

After `HypothesisExecutor.on_exit()`:
1. Build `HypothesisOutcomeRecord`
2. Call `classify_outcome()` → tags
3. Persist to `EpisodeMemoryStore` (in-session) + event store
4. `PlaybookOutcomeAggregator` updates `## Validation Evidence` in playbook `.md`
5. Next scanner run picks up updated priors

**Exit criteria:**
- Each round trip emits `hypothesis_outcome` event with attribution tags
- Scanner cards show `playbook_win_rate` and `r_expectancy` when ≥ 5 episodes exist
- Playbook `.md` files updated with validation evidence after each session

---

## LLM Prompt Strategy (Cross-cutting for R77/R78)

### New prompt template: `prompts/hypothesis_plan.txt`

The LLM is asked to produce a different artifact than today.

**Today's ask (trigger soup):**
> "Generate a list of entry and exit triggers as DSL rule strings."

**New ask (hypothesis-first):**
> "For each selected symbol, identify one trade hypothesis. Describe:
> - What you see in the structure and indicators that creates an edge
> - What price action would confirm the thesis (entry_rule)
> - Where the thesis is wrong (stop_price — must be a structural level)
> - Where the thesis is complete (target_price — must be a structural level)
> - What would invalidate the thesis before stop/target is reached
>
> You do NOT need to generate exit triggers. Stop and target are explicit prices,
> not rule expressions. The execution engine handles them automatically."

**Prohibited outputs (compiler rejects):**
- `direction: flat` or `direction: exit` in any trigger/hypothesis
- `hypotheses[]` entry with `stop_price` missing or on wrong side of entry
- `hypotheses[]` entry with `target_price` missing or on wrong side of entry
- `rr_ratio < 1.2` (compiler rejects at validation time)

---

## Data and Snapshot Quality (Pre-condition for all phases)

The following issues reduce the quality of LLM input and should be addressed during R74/R78:

### 1. `text_signals` and `visual_signals` — permanently absent
These are placeholder sections in `build_policy_snapshot()` hardcoded as always-missing.
They represent future integrations (news/sentiment, chart pattern recognition). Not blocking.

### 2. `memory_bundle` — spuriously absent
`memory_bundle_id` is never passed to `build_policy_snapshot()` even when memory was retrieved.
**Fix in R78:** Pass a content-addressed ID (SHA256 of bundle) when `_validation_bundle` is populated.

### 3. Indicator staleness
`EXECUTION_MAX_STALENESS_SECONDS` (default 1800s) allows 30-minute-old indicators.
For 1h candle strategies this is fine but for 5m strategies it's too permissive.
**Fix in R77:** Set staleness threshold relative to `indicator_timeframe`:
`max(indicator_tf_seconds * 2, 300)`.

### 4. Structure snapshot coverage
Structure engine only runs when `_raw_ohlcv_data` is present. Cold starts have no structure.
**Fix in R74:** Ensure `fetch_indicator_snapshots_activity` always returns OHLCV rows,
even for minimal/fallback snapshots.

---

## Key File Locations for Implementation

| Area | File |
|------|------|
| New hypothesis schema | `schemas/hypothesis.py` (new) |
| New opportunity schema | `schemas/opportunity.py` (new) |
| New session intent schema | `schemas/session_intent.py` (new) |
| Hypothesis compiler | `services/hypothesis_compiler.py` (new) |
| Hypothesis executor | `services/hypothesis_executor.py` (new) |
| Cadence governor | `services/cadence_governor.py` (new) |
| Opportunity scanner | `services/opportunity_scanner.py` (new) |
| Session planner | `services/session_planner.py` (new) |
| Hypothesis attribution | `services/hypothesis_attribution.py` (new) |
| Plan generation activity | `tools/paper_trading.py` — `generate_strategy_plan_activity` |
| Workflow execution | `tools/paper_trading.py` — `_evaluate_and_execute`, `_generate_plan` |
| LLM hypothesis prompt | `prompts/hypothesis_plan.txt` (new) |
| Scanner UI | `ui/src/components/ScannerPanel.tsx` (new) |
| Scanner ops API | `ops_api/routers/scanner.py` (new) |
| Core plan schema | `schemas/llm_strategist.py` — add `hypotheses[]` to `StrategyPlan` |

---

## Success Metrics (end state)

| Metric | Target |
|--------|--------|
| Round trips per 24h session | Median 5–10 |
| `block_class_counts.quality_gate` | 0 (all blocks are valid risk blocks) |
| `block_class_counts.infra` | 0 in normal conditions |
| Dead sessions (zero actionable triggers) | < 5% of sessions |
| `rr_ratio` on all opened hypotheses | ≥ 1.2 (enforced by compiler) |
| Attribution coverage | 100% of round trips have attribution tags |
| Schema pass rate by prompt template (Phase 5) | Tracked and improving |
| Hallucination findings per plan (Phase 6) | < 1 REVISE finding per 10 plans |
| Judge approval rate (Phase 7) | Within 10% of target base rate |

---

## Phase 5 — Prompt Hygiene

*Source: `docs/analysis/PROMPT_AUDIT_2026-04-23.md`*
*Prerequisite for: Phase 6 (scratchpad, hallucination scorer) — you want the prompt surface clean before adding new sections to it.*

### 5.1 — Clean Up the Live Path (immediate)

Three changes, no new features:

1. **Re-point prompt API to runtime default.** `ops_api/routers/prompts.py` currently exposes `prompts/llm_strategist_prompt.txt` as the editable strategist prompt, but `llm_client.py` defaults to `prompts/llm_strategist_simple.txt`. Align them — make the ops API serve and accept `llm_strategist_simple.txt`.

2. **Collapse duplicate schema files.** `strategy_plan_schema.txt` and `strategy_plan_schema_core.txt` are currently identical. Delete `strategy_plan_schema.txt` and make `STRATEGIST_SCHEMA_MODE` env var point to `strategy_plan_schema_core.txt` in both branches. Remove the dead `verbose` branch.

3. **Remove duplicate `STRATEGY_GUIDANCE` injection.** `prompt_builder.py` can append guidance that `llm_client.py` also appends from `LLM_STRATEGIST_PROMPT`. Consolidate to one path; `prompt_builder.py` is the canonical appender; `llm_client.py` delegates.

### 5.2 — Prune the Template Surface

1. Gate these templates behind `PROMPT_ENABLE_BROAD_ARCHETYPE_TEMPLATES=false` (default off):
   - `aggressive_active`, `balanced_hybrid`, `mean_reversion`, `momentum_trend_following`, `volatility_breakout`
2. Update `ui/src/lib/presets.ts` fast-timeframe presets to use `scalper_fast` not `aggressive_active`.
3. Gate directional breakout variants behind `PROMPT_ENABLE_DIRECTIONAL_BREAKOUT_TEMPLATES=false` until stale identifiers are fixed (Phase 5.3).

**Prompt registry manifest** — introduce `prompts/registry.yaml` with metadata per prompt/template:

```yaml
- id: llm_strategist_simple
  kind: strategist_base
  status: active
  enabled_by_default: true
  visible_in_ui: true

- id: scalper_fast
  kind: template
  status: active
  enabled_by_default: true
  visible_in_ui: true

- id: aggressive_active
  kind: template
  status: legacy
  enabled_by_default: false
  visible_in_ui: false
  flag: PROMPT_ENABLE_BROAD_ARCHETYPE_TEMPLATES
```

The ops API and UI both read this manifest for the allowed template list.

### 5.3 — Strengthen the Surviving Prompts

Rewrite three templates against the current allowed-identifier contract:

1. **`scalper_fast`** — tighten to fast-timeframe realism; explicit VWAP/mid-band/SMA mean targets; remove generic breakout language; update `position == 'flat'` → `is_flat`.

2. **`range_long` + `range_short`** — valid allowed identifiers only; explicit invalidation behavior; target toward `bollinger_middle` and `sma_medium`; remove `price_position_in_range` references.

3. **One breakout family** — choose `compression_breakout_long/short`, fix stale identifiers, re-enable; archive the others until needed.

### 5.4 — Add Prompt Evaluation Telemetry

Track per-prompt-template:
- Schema pass rate
- Validation pass rate (judge approval)
- Blocked-trigger rate
- Empty-plan rate
- Stand-down rate
- Realized R multiple (when trades close)

Emit `prompt_cohort_id` (hash of prompt template name + version) on every `plan_generated` event. The ops API aggregates by cohort for comparison dashboards.

**Exit criteria for Phase 5:**
- Prompt API edits land in the actual runtime default
- UI shows only first-class templates by default
- `cmp strategy_plan_schema.txt strategy_plan_schema_core.txt` shows a diff (one is canonical, one is deleted)
- `prompt_cohort_id` present on `plan_generated` events

---

## Phase 6 — SOTA Foundation

*Source: `docs/analysis/AGENTIC_SOTA_ALIGNMENT_PLAN.md` — Runbooks R88, R90*
*Prerequisite for: Phases 7, 8, 9, 10.*

### R88 — Strategist Chain-of-Thought Scratchpad

Add a `<reasoning>…</reasoning>` block to the LLM response before the JSON plan. The scratchpad is a short (4–8 sentence) derivation covering regime rationale, playbook selection, and primary risk. The judge gate stores it for downstream hallucination checking and reflexion memory.

**Key files:** `prompts/llm_strategist_simple.txt`, `agents/strategies/llm_client.py::_extract_plan_json()`, `agents/strategies/plan_provider.py`, `schemas/llm_strategist.py` (add `scratchpad: str | None`).

**Exit criteria:** `plan_generated` events contain non-empty `scratchpad_text`.

### R90 — Trading-Domain Hallucination Taxonomy + Section Scorer

Deterministic fine-grained hallucination scorer covering all six FG-PRM types adapted to the trading domain. Runs as Layer 0 before the 5-layer judge gate at zero additional LLM cost.

| Type | Check |
|---|---|
| Context Inconsistency | `plan.regime` conflicts with majority-vote regime from `LLMInput.assets` |
| Logical Inconsistency | Opposite-direction triggers on same symbol without mutually exclusive entry conditions |
| Instruction Inconsistency | Trigger category ∈ `judge_constraints.disabled_categories` |
| Calculation Error | `stop_loss_pct` outside allowed band or R:R below floor |
| Factual Inconsistency | `plan.allowed_symbols` contains symbol not in `LLMInput.assets` |
| Fabrication | `plan.playbook_id` not in `PlaybookRegistry` |

**Key files:** `services/plan_hallucination_scorer.py` (new), `schemas/judge_feedback.py`, `services/judge_validation_service.py`.

**Exit criteria:** Unit tests cover all 6 types; REJECT findings propagate to structural_violation.

---

## Phase 7 — Memory Closure

*Source: `docs/analysis/AGENTIC_SOTA_ALIGNMENT_PLAN.md` — Runbooks R91, R93, R97*
*Depends on: Phase 6 (scratchpad required for reflexion; hallucination scorer required for logprob integration).*
*Companion support track: `docs/analysis/TRADE_LIFECYCLE_REPLAY_ALIGNMENT_PLAN.md`
— use lifecycle identity + replay packs to improve loser/winner episode linkage
before broadening retrieval scope.*

### R91 — Reflexion Memory: Structured Self-Critique Persistence

When a position closes, synthesise a `PlanReflexionSummary` from the scratchpad, revision history, and outcome. Persist as a new memory record type. Inject top-3 reflexion summaries from loser episodes into future strategist prompts as `## Lessons from Similar Episodes`.

**Key files:** `schemas/episode_memory.py`, `services/episode_memory_service.py`, `services/memory_retrieval_service.py`, `agents/strategies/llm_client.py`.

### R93 — Per-Field Uncertainty Quantification

Extend scratchpad to include a `confidence_map` JSON (`{"regime": 0.0–1.0, "stance": 0.0–1.0, "allowed_directions": 0.0–1.0}`). Cross-reference LLM-stated confidence with cluster win-rate from memory bundle. Scale `risk_multiplier` down proportionally to mean field uncertainty.

**Key files:** `prompts/llm_strategist_simple.txt`, `agents/strategies/llm_client.py`, `services/plan_hallucination_scorer.py`, `services/judge_feedback_service.py`.

### R97 — Logprob-Based Token-Level Uncertainty Extraction

Enable `logprobs=True` on the strategist LLM call. Extract mean log-probability per token for `regime`, `stop_loss_pct`, and `target_pct`. Threshold < −1.5 mean logprob → `Factual Inconsistency` finding. Feeds into `PlanHallucinationReport` (R90) and `FieldUncertaintyScore` (R93) as a zero-training hardware-level cross-check.

**Key files:** `agents/strategies/llm_client.py`, `services/logprob_extractor.py` (new), `services/plan_hallucination_scorer.py`.

**Exit criteria for Phase 7:** Losing trade produces non-empty `## Lessons from Similar Episodes` in next plan; `field_logprobs` present on `plan_generated` event; `risk_multiplier` scales below 1.0 on low-confidence plans.

---

## Phase 8 — Generation Quality

*Source: `docs/analysis/AGENTIC_SOTA_ALIGNMENT_PLAN.md` — Runbooks R89, R92*
*Depends on: Phase 6/7 (need hallucination scorer and uncertainty scores to rank candidates).*
*Cost: ~2–4× LLM calls per policy loop. Gate behind env vars; validate Phase 6/7 value first.*

### R89 — Multi-Candidate Plan Generation + Best-of-N Selection

Generate N=3 candidate plans per policy loop (temperatures 0.10/0.15/0.20). Score each with `JudgePlanValidationService`. Select highest-confidence approved plan.

Gate: `STRATEGIST_BEST_OF_N=1` (default). Set to `3` to enable.

### R92 — Challenger Debate Pass (Two-Hypothesis Deliberation)

After primary plan, generate one challenger plan arguing the opposite or defensive stance. Run both through hallucination scorer. If challenger scores equally or higher, escalate approval threshold.

Gate: `STRATEGIST_DEBATE=false` (default).

**Note (open question):** "argue opposite stance" can produce degenerate flat/null plans that trivially win on hallucination score. Prompt engineering for meaningful contrasting hypotheses is required before enabling in production.

---

## Phase 9 — Operational Improvements

*Source: `docs/analysis/AGENTIC_SOTA_ALIGNMENT_PLAN.md` — Runbooks R94, R95*
*Can run in parallel with Phase 8.*

### R94 — Proactive Intra-Session Goal Reformulation

Extend `CadenceGovernor` to detect regime drift mid-session (cosine similarity of fingerprints). On drift signal during non-`THESIS_ARMED` state, trigger lightweight `SessionIntent` refresh without a full strategist plan cycle.

### R95 — FG-PRM-Style Aggregate Confidence Score + Operator Transparency Surface

Aggregate per-section hallucination scores + field uncertainty into a single `PlanConfidenceScore` (log-sum of non-hallucination probabilities, following FG-PRM's R_Φ formula). Expose via `GET /plan-audit/{plan_id}` — returns scratchpad, hallucination report, deliberation verdict, confidence score, revision history.

The companion lifecycle/replay workstream should provide the operator-facing
link from `plan_id` to executed `trade_set_id` lifecycles so `plan-audit`
surfaces can show not just a plan verdict, but the concrete entries/exits and
replay artifacts derived from that plan.

---

## Phase 10 — Trigger Architecture

*Source: `docs/analysis/AGENTIC_SOTA_ALIGNMENT_PLAN.md` — Runbook R96*
*Depends on: all prior phases (requires stable prompt surface and validated plan quality before changing the trigger model).*
*Architectural lift: highest of all phases. Implement last.*
*Companion support track: `docs/analysis/TRADE_LIFECYCLE_REPLAY_ALIGNMENT_PLAN.md`
— use lifecycle identity and replay packs as the acceptance harness for trigger
catalog + registry continuity across replans.*

### R96 — Canonical Trigger Catalog + Incremental Trigger Registry

Replace free-form trigger list generation with:
1. A **canonical trigger catalog** — 9 pre-defined archetypes with stable IDs (`mean_reversion_long`, `breakout_long`, `momentum_long`, `emergency_exit`, etc.)
2. A **stateful `TriggerRegistry`** — the LLM outputs a diff (`triggers_to_add`, `triggers_to_remove`, `triggers_to_modify`) not a full replacement
3. **`PolicyStateMachine` guard** — in `POSITION_OPEN`, exit trigger removal is blocked

This closes the gap where accumulated thesis confidence evaporates on every replan, and prevents the LLM's tendency to recompose slightly different phrasings of the same trigger that defeat deduplication and audit trails.

**Exit criteria:** Open position survives a mid-session replan without losing its stop/exit trigger binding; registry state serialises deterministically for Temporal CaN.

---

## Reference Documents

| Document | Contents |
|---|---|
| `docs/analysis/PROMPT_AUDIT_2026-04-23.md` | Full prompt inventory, findings, and execution order for Phase 5 |
| `docs/analysis/AGENTIC_SOTA_ALIGNMENT_PLAN.md` | Gap analysis, FG-PRM/LLM-as-a-Judge integration, and full R88–R97 runbook specs |
| `docs/analysis/TRADE_LIFECYCLE_REPLAY_ALIGNMENT_PLAN.md` | Companion workstream for lifecycle identity, replay packs, and pattern retrieval supporting R91, R95, and R96 |

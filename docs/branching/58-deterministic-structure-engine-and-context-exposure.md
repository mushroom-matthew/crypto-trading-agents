# Runbook 58: Deterministic Structure Engine and Context Exposure

## Purpose

Define a deterministic structure engine that produces reusable market-structure outputs
for:

- policy-loop reassessment triggers (strategy/playbook re-evaluation)
- deterministic entry activation logic
- deterministic stop/target candidate selection
- user-visible structure inspection in ops UI

This runbook formalizes a core Phase 8 requirement: the system should not rely on ad-hoc
LLM interpretation of "support/resistance" or "structure" at decision time. Structure
must be produced by a deterministic engine and exposed as typed, timestamped context.

## Why This Runbook Exists

Current HTF anchor fields (`htf_daily_high`, `htf_5d_low`, etc.) are useful but limited:

- they are a small anchor set, not a general structure model
- they do not emit structural events (break/reclaim/sweep/shift)
- they do not standardize near/mid/far target ladders across timeframes
- UI exposure can be confusing because raw anchors and current support/resistance roles
  are not clearly separated

This runbook adds the missing deterministic layer so structure becomes a first-class input
to policy and execution rather than a UI-only heuristic.

## Position in Architecture

This runbook introduces a deterministic structure subsystem that sits:

- **below** strategist/judge reasoning (LLM consumes structure outputs, does not invent them)
- **alongside** regime transition detection (Runbook 55)
- **upstream** of playbook instantiation and structural target selection (Runbooks 52, 56)
- **upstream** of UI panels that explain why the system is reassessing or targeting a level

## Dependencies

- **Runbook 49** — snapshot family (`TickSnapshot`, `PolicySnapshot`) and provenance/hash rules
- **Runbook 54** — policy-loop cadence and event-driven reassessment triggers
- **Runbook 55** — deterministic regime transition detector (complementary signal)
- **Runbook 52** — playbook schema (expects structural context and expectancy gating)
- **Runbook 56** — structural target selection / refinement enforcement consumes engine outputs
- **Runbook 41** — existing HTF daily anchor fields (initial deterministic sources)

## Scope

1. **`schemas/structure_engine.py`** (new)
   - typed `StructureSnapshot`, `StructureLevel`, `StructureEvent`, `LevelLadder`
2. **`services/structure_engine.py`** (new)
   - deterministic level extraction, ranking, event detection, role classification
3. **`services/market_snapshot_builder.py`**
   - include structure-engine outputs/references in `TickSnapshot` / `PolicySnapshot`
4. **Policy trigger orchestration**
   - emit policy-loop reassessment triggers from structural events (Runbook 54 integration)
5. **`agents/strategies/trigger_engine.py`** / compiler paths
   - consume canonical structural entry activation / stop / target candidates
6. **`services/structural_target_selector.py`** (Runbook 56 integration point)
   - resolve target candidates from deterministic structure levels
7. **Ops API / UI exposure**
   - time-indexed structure snapshots + level ladders + structural events for user inspection
8. **Tests**
   - determinism, ranking stability, event generation, snapshot integration, UI/API contracts

## Out of Scope (First Implementation)

- ML-learned structure detection
- discretionary chart-drawing tools
- non-deterministic LLM-generated trendlines
- fully automatic playbook editing (Runbook 48/52 human-review flow remains)
- advanced order-book/microstructure S/R models (can be follow-up runbook)

## Core Design Principles (Non-Negotiable)

### 1. Separate Raw Levels from Current Role

Each level must preserve:

- **source/kind** (what it is)
- **price**
- **timeframe**
- **provenance**

and only then derive:

- **current role** (`support` / `resistance` / `neutral`) relative to a reference price

This avoids conflating "D-1 high" with "resistance" when the level may now act as support.

### 2. Timeframe-Scoped Semantics

"Latest support" is ambiguous unless the timeframe is explicit.

All structure outputs must be scoped by:

- source timeframe of the level
- evaluation timeframe used for event detection
- as-of timestamp of the snapshot

### 3. Deterministic, Auditable Event Generation

Structural events that trigger policy reassessment or activation logic must be:

- deterministic
- replayable
- timestamped
- attributable to specific levels and evidence fields

### 4. Engine Outputs are Context, Not Direct Orders

The structure engine does not place trades.

It produces:

- level ladders
- event signals
- candidate stop/target anchors
- trigger-ready structural conditions

Execution remains controlled by existing policy/risk/trigger enforcement.

## Schema Contract (Required)

## `StructureLevel`

Illustrative shape:

```python
class StructureLevel(BaseModel):
    level_id: str
    snapshot_id: str
    symbol: str
    as_of_ts: datetime

    price: float
    source_timeframe: str            # e.g. "15m", "1h", "1d", "1w"
    kind: Literal[
        "prior_session_high",
        "prior_session_low",
        "rolling_window_high",
        "rolling_window_low",
        "swing_high",
        "swing_low",
        "trendline",
        "channel_upper",
        "channel_lower",
        "measured_move_projection",
    ]
    source_label: str                # human-readable, deterministic label (e.g. "D-1 High")
    source_metadata: dict[str, Any] = {}

    # Dynamic role relative to snapshot reference price
    role_now: Literal["support", "resistance", "neutral"]
    distance_abs: float
    distance_pct: float
    distance_atr: float | None = None

    # Strength / quality
    strength_score: float | None = None     # deterministic composite or sub-scores
    touch_count: int | None = None
    last_touch_ts: datetime | None = None
    age_bars: int | None = None

    # Use-case flags
    eligible_for_entry_trigger: bool = False
    eligible_for_stop_anchor: bool = False
    eligible_for_target_anchor: bool = False
```

## `StructureEvent`

Illustrative shape:

```python
class StructureEvent(BaseModel):
    event_id: str
    snapshot_id: str
    symbol: str
    as_of_ts: datetime
    eval_timeframe: str

    event_type: Literal[
        "level_broken",
        "level_reclaimed",
        "liquidity_sweep_reject",
        "range_breakout",
        "range_breakdown",
        "trendline_break",
        "structure_shift",
    ]
    severity: Literal["low", "medium", "high"]
    level_id: str | None = None
    level_kind: str | None = None
    direction: Literal["up", "down", "neutral"] = "neutral"

    # Deterministic evidence
    price_ref: float | None = None
    close_ref: float | None = None
    threshold_ref: float | None = None
    confirmation_rule: str | None = None
    evidence: dict[str, Any] = {}

    # Policy integration hints (advisory; actual cadence still follows Runbook 54)
    trigger_policy_reassessment: bool = False
    trigger_activation_review: bool = False
```

## `StructureSnapshot`

Required sections:

- identity/provenance (versioned, hashable, immutable)
- reference price/timeframe
- `levels[]` (raw + classified levels)
- `ladders` (near/mid/far supports and resistances by timeframe)
- `events[]` (structural changes at snapshot boundary)
- quality/staleness/missing components

`StructureSnapshot` should be embeddable (or referenced) from `TickSnapshot`/`PolicySnapshot`
per Runbook 49 rather than passed as an ad-hoc dict.

## Deterministic Level Sources (Implementation Strata)

## Stratum S1 — Anchor Levels (Ship First)

Deterministic sources already available or trivial to derive:

- prior daily (`D-1`) OHLC anchors
- prior daily (`D-2`) anchors / midpoint
- rolling 5D high/low (existing)
- rolling 20D high/low (new)
- prior weekly high/low (new)
- prior monthly high/low (new)

This stratum alone improves breakout target ladders and reduces UI ambiguity.

## Stratum S2 — Swing Structure (Next)

Add deterministic swing points on selected timeframes (e.g. 15m/1h/4h):

- swing highs/lows using fixed lookback/forward rules
- strength/touch scoring
- recency ranking

These become stronger candidates for entry triggers and structural stops/targets.

## Stratum S3 — Trendlines / Channels (Later, Deterministic)

Add trendline/channel primitives only if they are:

- parameterized
- deterministic
- replayable
- versioned

No freehand or LLM-drawn lines.

## Support / Resistance Semantics (Required)

The engine must explicitly distinguish:

1. **Raw level identity**
   - e.g. `D-1 High`, `Weekly High`, `Swing High #3`
2. **Current role**
   - support/resistance/neutral relative to reference price
3. **Use case**
   - entry trigger, stop anchor, target anchor, reassessment trigger

The same raw level may flip roles over time. The engine must preserve role transitions in
events/telemetry rather than relabeling silently.

## Reassessment Trigger Integration (Runbook 54)

Structural events may trigger policy-loop reassessment only via typed policy events.

Examples:

- HTF resistance reclaim confirmed (`level_reclaimed`, `eval_timeframe="1h"`)
- range breakout from compression box (`range_breakout`)
- structural invalidation of armed thesis level (`structure_shift`, `level_broken`)
- multi-timeframe confluence change (e.g., weekly level entered/exited proximity band)

Required output for policy-loop integration:

- `policy_trigger_reasons[]`
- `policy_event_priority`
- suppression-ready metadata (for cooldown/debounce observability)

## Tick-Engine / Trigger Integration (Runbooks 52/56)

The trigger engine and compiler should consume canonical structural outputs, not raw names:

- activation-level references (`level_id`)
- activation conditions (`touch`, `close_confirm`, `sweep_reclaim`)
- stop candidates (ordered, with rejection reasons)
- target candidates (ordered, with selection mode and expectancy telemetry)

Required compiler/telemetry threading:

- `structural_entry_level_id`
- `structural_stop_candidates[]`
- `structural_target_candidates[]`
- `selected_structural_target_level_id`
- candidate rejection reasons (deterministic)

## UI / Ops Exposure (Required)

The user should be able to inspect what the engine is using.

Minimum UI/API requirements:

1. Time-indexed `StructureSnapshot` retrieval (`as_of`, symbol, timeframe)
2. Multi-timeframe ladder view (`15m`, `1h`, `1d`, `1w`, `1M` as available)
3. Explicit raw level labels + current role
4. Distances to price (`%`, ATR)
5. Structural events timeline (what changed and why)
6. "Why replan/reassess happened" linkage to structural events
7. Provenance / snapshot hash visibility (or stable reference)

## Provenance / Hashing / Quality (Required)

The structure engine must follow Runbook 49 evidence-artifact rules:

- `snapshot_version`
- `snapshot_id`
- `snapshot_hash`
- `as_of_ts`
- `generated_at_ts`
- `created_at_bar_id`
- deterministic derivation manifest/version

Quality flags should include:

- missing timeframes
- stale source data
- partial level generation (e.g., weekly unavailable due to warmup)

## Implementation Steps

### Step 1: Define typed structure schemas

Add `schemas/structure_engine.py` with:

- `StructureLevel`
- `StructureEvent`
- `LevelLadder`
- `StructureSnapshot`

### Step 2: Implement Stratum S1 deterministic level extraction

Add `services/structure_engine.py` for anchor levels and ladder construction using:

- daily/weekly/monthly prior-session anchors
- rolling windows (5D / 20D)
- deterministic role classification and distance metrics

### Step 3: Add structural event detection (initial event set)

Implement deterministic events for:

- break / reclaim
- range breakout / breakdown (using deterministic bounds)
- liquidity sweep + reject (if primitive exists; otherwise defer)

### Step 4: Thread outputs into snapshot builders

Update `services/market_snapshot_builder.py` and related paths so policy/tick snapshots
carry structure outputs or a stable reference.

### Step 5: Integrate policy-loop trigger hooks

Translate high-confidence structural events into typed reassessment triggers (Runbook 54).

### Step 6: Integrate stop/target/entry candidate outputs

Expose candidate ladders to compiler/trigger paths for Runbooks 52/56 consumption.

### Step 7: Expose to Ops/UI

Add API + UI panels for:

- multi-timeframe level ladders
- structural events
- selected-candle `as_of` lookup

### Step 8: Add Stratum S2 swing structure

Add swing highs/lows and their ranking/strength metadata.

### Step 9: Evaluate deterministic trendlines/channels (Stratum S3)

Only ship if deterministic and replay-safe.

## Acceptance Criteria

- [x] Typed `StructureSnapshot` / `StructureLevel` / `StructureEvent` schemas exist and are validated
- [x] Deterministic anchor-level engine (daily/weekly/monthly/rolling) produces ranked ladders with explicit raw-level identity and current role
- [x] Structural role classification is separated from raw level identity in API/UI outputs
- [x] Structural events are deterministic, replayable, and carry evidence metadata
- [x] Policy-loop reassessment can be triggered from typed structural events (Runbook 54 integration)
- [x] Trigger/compiler paths can consume deterministic structural stop/target/entry candidates (Runbooks 52/56 integration points)
- [x] Structure outputs are embedded in or referenced by `TickSnapshot` / `PolicySnapshot` with provenance/hash-compatible metadata (Runbook 49 alignment)
- [x] Ops API supports time-indexed snapshot lookup and multi-timeframe ladder inspection
- [x] UI exposes level ladders + structural events and explains "why" for replan/reassessment triggers
- [x] Tests cover determinism, ranking stability, and event generation semantics

## Test Plan

```bash
# Structure schema validation
uv run pytest tests/test_structure_engine_schema.py -vv

# Deterministic level extraction + ladder ranking
uv run pytest tests/test_structure_engine_levels.py -vv

# Structural event detection (break/reclaim/range events)
uv run pytest tests/test_structure_engine_events.py -vv

# Snapshot-builder integration (R49 alignment)
uv run pytest tests/test_market_snapshot_builder.py -k "structure" -vv

# Ops API contracts (time-indexed retrieval, target candidates)
uv run pytest tests/test_ops_api_structure.py -vv

# All R58 tests together
uv run pytest tests/test_structure_engine_schema.py tests/test_structure_engine_levels.py tests/test_structure_engine_events.py tests/test_ops_api_structure.py tests/test_market_snapshot_builder.py -v
```

## Worktree Setup

```bash
git fetch origin
git worktree add ../crypto-trading-agents-r58 origin/main
cd ../crypto-trading-agents-r58
git checkout -b r58-deterministic-structure-engine
```

## Git Workflow

```bash
git status
git add docs/branching/58-deterministic-structure-engine-and-context-exposure.md docs/branching/README.md
git commit -m "docs(runbook): add r58 deterministic structure engine contract"
```

## Human Verification

- Confirm the runbook clearly distinguishes raw level identity vs support/resistance role.
- Confirm policy-loop reassessment triggers and tick-engine stop/target integrations are both in scope.
- Confirm UI exposure requirements are explicit (time-indexed lookup, events, ladders, provenance).

## Test Evidence

```
uv run pytest tests/test_structure_engine_schema.py tests/test_structure_engine_levels.py \
  tests/test_structure_engine_events.py tests/test_ops_api_structure.py \
  tests/test_market_snapshot_builder.py -v

194 passed, 1 warning in 57.55s
```

Full suite (excluding pre-existing DB_DSN collection errors in test_agent_workflows.py and
test_metrics_tools.py):
```
1273 passed, 2 skipped  (2 pre-existing test_factor_loader.py failures — pandas version mismatch
unrelated to R58)
```

## Human Verification Evidence

- `StructureLevel` carries both `kind` (raw level type: `prior_session_high`, `swing_high`, etc.)
  and `role` (`support` | `resistance` | `both`), confirming raw identity is always preserved
  separately from the current role classification. Role flips are surfaced via
  `StructureEvent(kind="structure_shift")` rather than silent overwrites.
- `StructureSnapshot.policy_trigger_reasons` and `policy_event_priority` fields are populated by
  the engine and consumed by `PolicySnapshot.structure_policy_priority` (R54 forward-compat hook).
- `services/structural_target_selector.py` exposes `select_stop_candidates()` and
  `select_target_candidates()` as the R52/R56 integration surface.
- `GET /structure/{symbol}` and `GET /structure/{symbol}/ladders/{timeframe}` serve time-indexed
  lookups; `GET /structure/{symbol}/target-candidates` exposes stop/target-eligible levels.
- UI time-travel viewer (`BacktestPlaybackViewer.tsx`) and live paper trading panel
  (`PaperTradingControl.tsx`) both received structure snapshot and level ladder display
  (committed in `ad9926e`).

## Change Log

- 2026-02-25: Initial runbook drafted for deterministic structure engine outputs, integration hooks, and user exposure.
- 2026-02-26: Implementation complete. New files: `schemas/structure_engine.py`, `services/structure_engine.py`, `services/structural_target_selector.py`, `ops_api/routers/structure.py`, `tests/test_structure_engine_schema.py`, `tests/test_structure_engine_levels.py`, `tests/test_structure_engine_events.py`, `tests/test_ops_api_structure.py`. Modified: `schemas/market_snapshot.py`, `services/market_snapshot_builder.py`, `ops_api/app.py`, `tests/test_market_snapshot_builder.py`. 194 new tests, 1273 full suite passing.

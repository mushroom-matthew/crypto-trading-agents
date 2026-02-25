# Runbook 57: Screener Timeframe Threading

## Purpose

The universe screener derives an `expected_hold_timeframe` per candidate (e.g., `1m`, `15m`,
`1h`, `4h`) based on hypothesis type, volatility state, and ATR expansion. When a user applies
a screener candidate to a session via the "Use" button, the UI converts this to a
`plan_interval_hours` value (the replan cadence) but **drops the timeframe itself** — it never
reaches the backend.

As a result, `fetch_indicator_snapshots_activity` always fetches `1h` OHLCV candles regardless
of the screener's recommendation, and the LLM generates triggers defaulting to `1h`. A `1m`
screener candidate ends up traded on `1h` triggers — the strategy evaluates far less frequently
than the setup requires, misses short-lived entries, and holds positions far longer than the
screener's thesis implies.

## Scope

1. **`tools/paper_trading.py`** — `PaperTradingConfig` gains `indicator_timeframe: str = "1h"`;
   workflow stores and uses it for `fetch_indicator_snapshots_activity`; `generate_strategy_plan_activity`
   gains `indicator_timeframe` param and injects a `TIMEFRAME:` hint into the prompt
2. **`ops_api/routers/paper_trading.py`** — `PaperTradingSessionConfig` gains
   `indicator_timeframe: str = "1h"`; wired into `workflow_config`
3. **`ui/src/lib/api.ts`** — `PaperTradingSessionConfig` interface gains `indicator_timeframe?: string`
4. **`ui/src/components/PaperTradingControl.tsx`** — `applyScreenerCandidateToForm` sets
   `indicator_timeframe` from `item.expected_hold_timeframe`; `recommendedPlanIntervalHours`
   extended to handle `1m`/`5m`; session start payload includes `indicator_timeframe`
5. **`tests/test_screener_timeframe_threading.py`** — new test file

## Out of Scope

- Changing the candle-clock evaluation cadence per timeframe (already correct: the trigger's
  `timeframe` field drives evaluation frequency via `last_eval_candle_by_tf`)
- Multi-timeframe indicators within a single session (all indicators at the same timeframe)
- Changing the live `_refresh_live_indicators` call (already fetches `1m` for the structure UI)
- Per-instrument timeframe (all symbols in a session share one `indicator_timeframe`)

## Key Files

- `tools/paper_trading.py` — workflow + activities (primary)
- `ops_api/routers/paper_trading.py` — session config (add field)
- `ui/src/lib/api.ts` — API interface (add field)
- `ui/src/components/PaperTradingControl.tsx` — "Use" button handler + session start

## Implementation Steps

### Step 1: `PaperTradingConfig` in `tools/paper_trading.py`

Add field after `plan_interval_hours`:

```python
indicator_timeframe: str = Field(
    default="1h",
    description=(
        "OHLCV timeframe used for indicator computation and trigger generation. "
        "Should match the screener's expected_hold_timeframe for the selected candidates. "
        "Valid values: '1m', '5m', '15m', '1h', '4h', '1d'."
    ),
)
```

### Step 2: Workflow `run()` — store the field

In `PaperTradingSessionWorkflow.run()`, after `self.plan_interval_hours = parsed_config.plan_interval_hours`, add:

```python
self.indicator_timeframe = parsed_config.indicator_timeframe
```

### Step 3: Workflow `_generate_plan()` — use it for indicator fetch

Replace the hardcoded `"1h"` at the `fetch_indicator_snapshots_activity` call site:

```python
# Before:
args=[self.symbols, "1h", 300],

# After:
args=[self.symbols, self.indicator_timeframe, 300],
```

### Step 4: `generate_strategy_plan_activity` — propagate timeframe to LLM

Add parameter:
```python
async def generate_strategy_plan_activity(
    symbols: List[str],
    portfolio_state: Dict[str, Any],
    strategy_prompt: Optional[str],
    market_context: Dict[str, Any],
    llm_model: Optional[str] = None,
    session_id: Optional[str] = None,
    repair_instructions: Optional[str] = None,
    indicator_timeframe: Optional[str] = None,   # ← NEW
) -> Dict[str, Any]:
```

Fix the hardcoded fallback in `snap_init`:
```python
# Before:
snap_init.setdefault("timeframe", "1h")

# After:
snap_init.setdefault("timeframe", indicator_timeframe or "1h")
```

Inject a timeframe hint block at the bottom of `effective_prompt` (before calling
`llm_client.generate_plan()`):
```python
tf = indicator_timeframe or "1h"
timeframe_hint = (
    f"\nTIMEFRAME: Use '{tf}' as the timeframe for all triggers in this plan. "
    f"The indicator snapshot was computed on {tf} candles. "
    f"All entry_rule and exit_rule expressions refer to {tf}-close values.\n"
)
effective_prompt = (effective_prompt or "") + timeframe_hint
```

### Step 5: Workflow `_generate_plan()` — pass `indicator_timeframe` to activity

In both the primary and repair calls to `generate_strategy_plan_activity`, add `self.indicator_timeframe` as the last arg:

```python
args=[
    self.symbols,
    portfolio_state,
    self.strategy_prompt,
    market_context,
    None,                       # llm_model
    self.session_id,
    None,                       # repair_instructions (primary call only)
    self.indicator_timeframe,   # ← NEW
],
```

### Step 6: `PaperTradingSessionConfig` in `ops_api/routers/paper_trading.py`

Add field after `plan_interval_hours`:

```python
indicator_timeframe: str = Field(
    default="1h",
    description=(
        "OHLCV timeframe for indicator computation — should match the screener's "
        "expected_hold_timeframe. Valid: '1m', '5m', '15m', '1h', '4h', '1d'."
    ),
)
```

Wire into `workflow_config`:
```python
"indicator_timeframe": config.indicator_timeframe,
```

### Step 7: UI — `ui/src/lib/api.ts`

Add to `PaperTradingSessionConfig` interface:
```typescript
indicator_timeframe?: string;  // OHLCV timeframe for indicator fetch and trigger generation
```

### Step 8: UI — `ui/src/components/PaperTradingControl.tsx`

**8a. Fix `recommendedPlanIntervalHours` to handle short timeframes:**
```typescript
function recommendedPlanIntervalHours(timeframe: string): number | null {
  const tf = timeframe.trim().toLowerCase();
  if (tf === '1m' || tf === '5m') return 0.5;   // replan every 30 min for short-hold setups
  if (tf === '15m') return 1;
  if (tf === '1h') return 4;
  if (tf === '4h') return 8;
  return null;
}
```

**8b. Set `indicator_timeframe` state:**

Add state variable near other session form state:
```typescript
const [indicatorTimeframe, setIndicatorTimeframe] = useState<string>('1h');
```

**8c. Apply from screener "Use" button:**

In `applyScreenerCandidateToForm`, after setting `planIntervalHours`:
```typescript
setIndicatorTimeframe(item.expected_hold_timeframe);
```

**8d. Include in session start payload:**

In the `startSession` mutation, add to the config object:
```typescript
indicator_timeframe: indicatorTimeframe,
```

## Schema Backwards Compatibility

`indicator_timeframe` defaults to `"1h"` in both `PaperTradingConfig` and
`PaperTradingSessionConfig`. Existing sessions started without this field continue to behave
exactly as before. No migration required.

## Environment Variables

None new. `indicator_timeframe` is a session-level config, not a global env var.

## Test Plan

```bash
# Unit: new field validates and defaults correctly
uv run pytest tests/test_screener_timeframe_threading.py -vv

# Regression: paper trading session start and indicator fetch
uv run pytest -k "paper_trading or indicator" -vv

# Full suite
uv run pytest -q
```

## Tests to Write (`tests/test_screener_timeframe_threading.py`)

```python
def test_paper_trading_config_defaults_to_1h():
    """PaperTradingConfig.indicator_timeframe defaults to '1h'."""
    ...

def test_paper_trading_config_accepts_1m():
    """PaperTradingConfig.indicator_timeframe='1m' validates without error."""
    ...

def test_session_config_defaults_to_1h():
    """PaperTradingSessionConfig.indicator_timeframe defaults to '1h'."""
    ...

def test_generate_plan_activity_injects_timeframe_hint():
    """generate_strategy_plan_activity injects TIMEFRAME: block when indicator_timeframe='1m'."""
    ...

def test_generate_plan_activity_snap_init_uses_timeframe():
    """snap_init uses indicator_timeframe as the timeframe fallback, not hardcoded '1h'."""
    ...
```

## Acceptance Criteria

- [x] `PaperTradingConfig.indicator_timeframe: str = "1h"` — backwards compatible
- [x] Workflow uses `self.indicator_timeframe` for `fetch_indicator_snapshots_activity` call
- [x] `generate_strategy_plan_activity` injects `TIMEFRAME:` hint into effective_prompt
- [x] `snap_init.setdefault("timeframe", ...)` uses `indicator_timeframe`, not `"1h"` literal
- [x] `PaperTradingSessionConfig` accepts `indicator_timeframe` and passes to workflow config
- [x] UI `applyScreenerCandidateToForm` sets `indicatorTimeframe` from `expected_hold_timeframe`
- [x] `recommendedPlanIntervalHours` handles `1m`/`5m` → 0.5h replan interval
- [x] Session start payload includes `indicator_timeframe`
- [x] All existing tests pass (no regression)
- [x] New unit tests pass

## Human Verification Evidence

```
Code inspection verified:
- PaperTradingConfig.indicator_timeframe defaults to "1h" (Field with default="1h")
- SessionState.indicator_timeframe = "1h" (backwards-compatible default)
- _generate_plan() passes self.indicator_timeframe to fetch_indicator_snapshots_activity
  replacing the hardcoded "1h" literal
- generate_strategy_plan_activity appends TIMEFRAME: hint block to effective_prompt
- snap_init.setdefault("timeframe", indicator_timeframe or "1h") — all 3 IndicatorSnapshot
  construction sites updated
- PaperTradingSessionConfig.indicator_timeframe wired into workflow_config dict
- UI: applyScreenerCandidateToForm calls setIndicatorTimeframe(item.expected_hold_timeframe)
- UI: recommendedPlanIntervalHours returns 0.5 for '1m' and '5m'
- UI: session start config includes indicator_timeframe: indicatorTimeframe
- _snapshot_state() and _restore_state() both carry indicator_timeframe across continue-as-new
- Repair pass also threads indicator_timeframe through (both workflow call and recursive activity call)

Paper trading verification pending: start a session using a 1m screener candidate and confirm
plan_generated event shows timeframe="1m" in indicator snapshot and triggers.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-25 | Runbook authored — screener timeframe threading from screener → indicator fetch → LLM prompt | Claude |
| 2026-02-25 | Implemented: indicator_timeframe field on PaperTradingConfig, SessionState, PaperTradingWorkflow; wired into fetch_indicator_snapshots_activity, generate_strategy_plan_activity (hint injection + snap_init fallback); PaperTradingSessionConfig + workflow_config; UI state + applyScreenerCandidateToForm + recommendedPlanIntervalHours; 8 new tests, 934 passed no regressions | Claude |

## Worktree Setup

```bash
git worktree add -b feat/screener-timeframe-threading ../wt-screener-tf main
```

## Git Workflow

```bash
git add \
  tools/paper_trading.py \
  ops_api/routers/paper_trading.py \
  ui/src/lib/api.ts \
  ui/src/components/PaperTradingControl.tsx \
  tests/test_screener_timeframe_threading.py

git commit -m "feat: thread screener expected_hold_timeframe into indicator fetch and LLM prompt (Runbook 57)"
```

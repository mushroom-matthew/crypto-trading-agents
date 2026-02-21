# Backlog: Per-Instrument Workflow Architecture

## Status: Backlog — do not implement until Runbooks 46 + 47 are validated

## Problem Statement

The current architecture attaches strategy plans and trigger evaluation to a `StrategyRun`
that spans multiple symbols. One plan covers BTC-USD and ETH-USD together, sharing the
same regime assessment, template selection, and judge feedback loop. As the system adds
autonomous instrument selection (Runbook 39 screener) and template-matched generation
(Runbooks 46/47), this coupling creates friction:

- **Template conflict**: BTC may warrant `compression_breakout` while ETH is in
  `bull_trending`. A shared plan can only declare one `template_id`.
- **State pollution**: A BTC emergency exit or judge replan restarts the evaluation
  cycle for ETH even if ETH's strategy is unchanged.
- **Screener cadence mismatch**: The screener surfaces new instruments every 15 minutes.
  Adding a new instrument currently requires creating a new `StrategyRun`, which
  is a user-initiated operation, not a system-level one.
- **Signal ledger provenance**: `SetupEvent` and `SignalEvent` both have `symbol` fields,
  but the plan provenance (`run_id`) is shared. Per-instrument workflows make the run_id
  directly instrument-scoped.

## Vision

Each active instrument gets its own Temporal workflow:

```
UniverseScreenerWorkflow
  └── emits InstrumentRecommendation{symbol, template_id, thesis}
        └── spawns (or signals) InstrumentStrategyWorkflow{symbol}
              ├── owns StrategyRun for that symbol
              ├── generates plan via StrategyPlanProvider with template_id hint
              ├── runs TriggerEngine per candle close (candle-clock, same as paper trading)
              ├── receives judge feedback specific to that symbol's performance
              └── signals ExecutionLedgerWorkflow for fills
```

Key invariant: `InstrumentStrategyWorkflow` is the single source of truth for one
symbol's active plan, template, position state, setup events, and signal ledger entries.
The execution ledger (cash, portfolio) remains shared across all instruments.

## Architecture Sketch

### `workflows/instrument_strategy_workflow.py`

```python
@workflow.defn
class InstrumentStrategyWorkflow:
    """Per-symbol strategy state and trigger evaluation loop."""

    @workflow.run
    async def run(self, config: InstrumentStrategyConfig) -> None:
        # 1. Load template from config.template_id
        # 2. Generate plan via activity (LLM call with template hint)
        # 3. Loop: receive candle-close signal → evaluate triggers → emit signal events
        # 4. Receive judge_feedback signal → update plan constraints
        # 5. Receive screener_recommendation signal → update template_id, trigger replan
        # 6. Continue-as-new at HISTORY_LIMIT
        ...

    @workflow.signal
    async def on_candle_close(self, candle: dict) -> None:
        """Receive OHLCV bar for this symbol's timeframe."""
        ...

    @workflow.signal
    async def on_judge_feedback(self, feedback: JudgeFeedback) -> None:
        ...

    @workflow.signal
    async def on_screener_update(self, recommendation: InstrumentRecommendation) -> None:
        """Screener has a new template suggestion for this symbol."""
        ...

    @workflow.query
    def get_instrument_state(self) -> dict:
        """Return current plan, template_id, position state, setup event count."""
        ...
```

### `schemas/instrument_strategy.py`

```python
class InstrumentStrategyConfig(BaseModel):
    symbol: str
    timeframes: list[str]          # e.g., ["1h", "4h"]
    template_id: str | None        # Initial template hint from screener
    screener_run_id: str | None    # Which screener run surfaced this instrument
    risk_limits: RiskLimitSettings
    parent_run_id: str             # StrategyRun ID for audit trail linkage
```

### Routing Layer

The `UniverseScreenerWorkflow` (Runbook 39) gains a new responsibility: on each
screening pass, for each top candidate it either:

- **Starts** an `InstrumentStrategyWorkflow` if none exists for that symbol
- **Signals** the existing workflow with the new template_id if the recommendation changed
- **Sends a stop signal** if the symbol drops below the screener threshold for 3+ passes

The `ExecutionLedgerWorkflow` (mock ledger) is unchanged — it continues to track
portfolio-level cash and positions regardless of which instrument workflows are active.

## Prerequisite Chain

1. Runbook 39 (Universe Screener) — screener must run before it can spawn workflows
2. Runbook 46 (Template-Matched Retrieval) — template_id field must be in screener output
3. Runbook 47 (Hard Template Binding) — per-instrument plans need stable template_id
4. Paper trading validation of 39+46+47 — before committing to workflow architecture change

## Estimated Scope

**New files:**
- `workflows/instrument_strategy_workflow.py`
- `schemas/instrument_strategy.py`
- `tests/test_instrument_strategy_workflow.py`

**Modified files:**
- `workflows/universe_screener_workflow.py` (spawn/signal InstrumentStrategyWorkflow)
- `worker/agent_worker.py` (register InstrumentStrategyWorkflow)
- `agents/strategies/plan_provider.py` (per-symbol cache keyed by symbol+date, not run_id)
- `services/strategist_plan_service.py` (accept InstrumentStrategyConfig)

**Not changed:**
- `agents/workflows/execution_ledger_workflow.py` (shared ledger unchanged)
- `trading_core/trigger_compiler.py` (per-symbol already, no change)
- `trading_core/trigger_engine.py` (per-symbol already, no change)
- `schemas/llm_strategist.py` (StrategyPlan unchanged)

## Open Questions (resolve before implementation)

1. **Workflow ID namespace**: `instrument-strategy-{symbol}` or include screener run ID?
   Using just symbol allows upsert (signal existing workflow if already running). Including
   run ID creates a new workflow each screening cycle — cleaner history but more workflows.

2. **Plan cache key**: Currently `(run_id, date)`. With per-instrument workflows, should
   it be `(symbol, date)` so the plan persists across screener cycles on the same symbol?

3. **Multi-timeframe**: Does each timeframe get its own workflow, or does one
   `InstrumentStrategyWorkflow` handle all timeframes for a symbol?
   Recommendation: one workflow per symbol, multi-timeframe handled internally (same as
   current paper trading candle-clock implementation in `tools/paper_trading.py`).

4. **Judge feedback routing**: Currently judge evaluates a StrategyRun. With per-symbol
   workflows, does the judge evaluate per symbol? Or does it still evaluate the portfolio
   and broadcast constraints to all active instrument workflows?

5. **Instrument deactivation**: When screener stops recommending a symbol, the
   InstrumentStrategyWorkflow should flatten its position then terminate (or pause).
   Flattening requires a position check — who initiates the flatten order?

## Why Deferred

The per-instrument workflow refactor touches the core execution path and requires
answering the questions above with operational evidence (how many instruments run
simultaneously, what is the candle-close fan-out latency). These answers only come from
running the screener in paper trading for 30+ days (Runbook 39 validation gate).

Do not implement this runbook until:
- [ ] Runbook 39 screener is running and surfacing ≥ 3 instruments simultaneously
- [ ] Runbook 46 retrieval routing is validated (correct template selected ≥ 80% of days)
- [ ] Runbook 47 hard binding shows no regressions in a 30-day paper trading run
- [ ] Open questions above are resolved via operational evidence

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-21 | Backlog runbook created — architectural sketch for per-instrument workflows | Claude |

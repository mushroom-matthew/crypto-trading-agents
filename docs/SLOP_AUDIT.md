# Repository Slop Audit - January 2026

**Status**: Critical Issues Identified
**Severity**: HIGH - Blocking effective operations

## Quick Summary

The repository is **60% mature** and in active refactor. There are **10 major sources of slop** preventing effective backtest/live trading UI development. The primary issues are:

1. **Three disconnected UIs** with no shared infrastructure
2. **Incomplete event emission** from agents (can't track decision chains)
3. **Hardcoded fallbacks** in materializer (always shows "running"/"paper")
4. **Zero backtest visualization** (sophisticated engine but invisible)
5. **Fragmented trade tracking** (different schemas for backtest vs live)

## Critical Path to Fix (2-3 weeks)

### Week 1: Foundation
- [ ] Wire all agents to emit events to event store
- [ ] Fix materializer to query actual Temporal status
- [ ] Add missing DB tables (BlockEvent, RiskAllocation, PositionSnapshot, BacktestRun)

### Week 2: API Layer
- [ ] Create unified API endpoints (/backtests, /live, /market, /agents, /wallets)
- [ ] Add WebSocket support for real-time streaming
- [ ] Bootstrap React frontend with core tabs

### Week 3: Integration
- [ ] Wire backtest orchestration to UI
- [ ] Implement live daily reports (match backtest format)
- [ ] Complete wallet reconciliation UI
- [ ] Clean up repository (delete garbage, organize docs)

## Top 10 Sources of Slop

### 1. Three Separate UI Implementations

**Files**: `ui/index.html`, `app/dashboard/templates/`, `ticker_ui_service.py`

**Problem**:
- Ops UI (port 8080): Trading operations - 40% complete, HIGH slop
- Dashboard (port 8081): Infrastructure - 70% complete, MEDIUM slop
- Ticker UI (terminal): Price charts - 85% complete, LOW slop

**Impact**: User must juggle 2 browser tabs + 1 terminal to see full state

**Fix**: Consolidate into single React SPA with 5 tabs (backtest, live, market, agents, wallets)

---

### 2. Agents Don't Emit Events Consistently

**Files**: `agents/execution_agent_client.py`, `agents/broker_agent_client.py`, `agents/judge_agent_client.py`

**Problem**:
- Event store exists (`ops_api/event_store.py`)
- Event schemas defined (`ops_api/schemas.py`)
- But agents emit events sporadically or to JSONL files instead

**Missing Events**:
- `trade_blocked` - Execution agent doesn't emit individual block events
- `order_submitted` - No correlation ID linking to fills
- `fill` - Not consistently emitted
- `intent` - Broker doesn't systematically emit
- `plan_judged` - Judge verdicts not captured

**Impact**: UI cannot show decision chains (broker intent → execution → block/fill)

**Fix**: Add `EventEmitter` to all agent clients, emit at every state transition

---

### 3. Materializer Hardcodes Status/Mode

**File**: `ops_api/materializer.py` lines 105-134

**Problem**:
```python
summaries[rid] = RunSummary(
    status="running",  # ← ALWAYS "running"
    mode="paper",      # ← ALWAYS "paper"
)
```

**Impact**: UI shows inaccurate workflow status and trading mode

**Fix**: Query Temporal client for actual workflow status, read runtime mode from config

---

### 4. Backtest Visualization: Complete Gap

**Files**: `backtesting/simulator.py` (sophisticated), `backtesting/cli.py` (CLI only)

**Problem**:
- Backtests generate rich results: equity curves, daily reports, risk budgets, trigger quality
- Results written to JSON files (`.cache/strategy_plans/{run_id}/`)
- **Zero web endpoints** to trigger or monitor backtests
- **Zero UI components** to visualize results
- Backlog acknowledges: "Integrate historical backtesting orchestration into dashboard" - NOT STARTED

**Impact**: Cannot run backtests from web, cannot see equity curves or compare runs

**Fix**:
- Add `/backtests` endpoints (POST to start, GET for status/results/equity/trades)
- Create `BacktestControl` React component
- Store results in `BacktestRun` DB table

---

### 5. Trade Tracking Schema Mismatch

**Backtest**: `trades_df` with columns: `time, symbol, side, qty, price, fee, pnl, risk_used_abs, trigger_id`

**Live**: `Order` table with columns: `order_id, wallet_id, product_id, side, quantity, fill_price, timestamp`

**Problem**:
- Different field names (`symbol` vs `product_id`, `qty` vs `quantity`)
- Backtest tracks `risk_used_abs` and `trigger_id`, live doesn't
- No unified API to query "all trades" regardless of mode

**Impact**: Cannot compare backtest vs live performance, fragmented data model

**Fix**: Normalize schemas, add `/trades` endpoint that unifies both sources

---

### 6. Block Reasons Not Persisted

**Backtest**: Aggregated counts in `daily_reports[].limit_stats`

**Live**: Ephemeral state in `trading_core/execution_engine.py`

**Problem**:
- No `BlockEvent` table with individual block records
- Cannot answer "Why was BTC trade at 14:32 blocked?"
- Only aggregate counts visible

**Impact**: Cannot debug block decisions, no audit trail for rejected trades

**Fix**: Create `BlockEvent` table, emit event on every block, expose via `/live/blocks` endpoint

---

### 7. Wallet Reconciliation: CLI Only, No History

**File**: `app/ledger/reconciliation.py`

**Problem**:
- Reconciliation is manual CLI command (`uv run python -m app.cli.main reconcile run`)
- No scheduled reconciliation
- No drift history table (can't see drift over time)
- No auto-correction mechanism
- No alerting if drift exceeds threshold

**Impact**: Manual toil, no visibility into drift trends

**Fix**:
- Add `/reconcile` endpoints (POST to trigger, GET for reports)
- Create `DriftHistory` table
- Add scheduled reconciliation (Temporal workflow or cron)
- Build reconciliation UI tab

---

### 8. Agent Decision Chains Fragmented

**Problem**:
- Broker generates intent → logged to JSONL
- Execution evaluates → logged to different JSONL
- Judge approves → logged to yet another JSONL
- No correlation IDs linking events across boundaries

**Impact**: Cannot trace "Why did this trade execute?" from user intent to fill

**Fix**:
- Generate `correlation_id` at broker intent
- Pass correlation_id through all events (intent → trigger → block/submit → fill)
- Create decision chain timeline visualization in Agent Inspector tab

---

### 9. Live Wallet Provider: Empty Stub

**File**: `agents/wallet_provider.py` lines 40-56

**Problem**:
```python
class LiveWalletProvider(WalletProvider):
    def get_balance(self, symbol: str) -> Decimal:
        raise RuntimeError("Live wallet provider not yet implemented")
```

**Impact**: Cannot use live wallets in trading, reconciliation UI would fail

**Fix**: Implement using `app/coinbase/client.py`, test in paper mode first

---

### 10. Repository Root Garbage

**Files to delete**:
- `test.txt` - Old test output
- `thing.JPG`, `Capture.JPG`, `Capture2.JPG` - Orphaned screenshots
- `main.py` - Useless entrypoint (just prints)
- `run_stack.sh` - Deprecated (use docker-compose)

**Files to move to `/docs/`**:
- `AGENTS.md`, `JUDGE_AGENT_README.md`, `README_metrics.md`
- `chat-interactions.md`, `Flow After Preferences Are Set.md`

**Impact**: Cluttered repo, confusing navigation

**Fix**: Clean up in Phase 3 (1 day effort)

---

## Prioritized Fix List

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 🔴 P0 | Wire agent event emission | 3 days | Unblocks decision chain visibility |
| 🔴 P0 | Fix materializer hardcoded values | 1 day | Accurate status/mode display |
| 🔴 P0 | Add missing DB tables | 1 day | Foundation for all tracking |
| 🟡 P1 | Create unified API endpoints | 4 days | Enable frontend development |
| 🟡 P1 | Bootstrap React frontend | 1 day | UI foundation |
| 🟡 P1 | Backtest control component | 2 days | Core feature gap |
| 🟢 P2 | Live monitoring components | 3 days | Operational visibility |
| 🟢 P2 | WebSocket integration | 2 days | Real-time updates |
| 🟢 P2 | Wallet reconciliation UI | 2 days | Reduce manual toil |
| 🔵 P3 | Clean up repository | 1 day | Quality of life |

**Total Effort**: 20 days (4 weeks with buffer)

---

## Validation Checklist

After implementation, verify:

### Event Emission
- [ ] Query event store: `SELECT type, COUNT(*) FROM events GROUP BY type`
- [ ] Verify: trade_blocked, order_submitted, fill, intent, plan_judged all present
- [ ] Check correlation IDs: Events with same correlation_id form complete chain

### Materializer
- [ ] Pause workflow in Temporal UI → Status shows "paused" in API
- [ ] Set `RUN_MODE=live` → Mode shows "live" in API
- [ ] Empty event store → Graceful fallback (no hardcoded "execution" run)

### Backtest UI
- [ ] Start backtest via web → Workflow appears in Temporal UI
- [ ] Poll `/backtests/{id}` → Progress increases from 0% to 100%
- [ ] Completed backtest → Equity curve renders in chart
- [ ] Trade log table → Shows all fills with trigger_id

### Live Monitoring
- [ ] Place trade via broker agent → Fill appears in UI table within 5s
- [ ] Block trade → Block reason appears in histogram
- [ ] Check risk budget → Shows used/available accurately

### Reconciliation
- [ ] Click "Run Reconciliation" → Report generates
- [ ] Drift detected → Shows in drift history table
- [ ] Alert threshold exceeded → UI shows warning badge

---

## References

**Full Implementation Plan**: `docs/UI_UNIFICATION_PLAN.md` (comprehensive 300+ line guide)

**Architecture Docs**: `docs/ARCHITECTURE.md`

**Agent Guidelines**: `docs/AGENTS.md` (to be moved from root)

**Current Status**: `docs/STATUS.md`

**Backlog**: `docs/backlog.md`

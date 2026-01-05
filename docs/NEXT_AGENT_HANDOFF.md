# Next Agent Handoff - 2026-01-04

**Last Session Completed**: 2026-01-04
**Phase Status**: Phase 1 & 2 Complete, Phase 3 In Progress
**System Maturity**: ~70% (up from 60%)

## 🎯 Quick Context

You're working on a **24×7 multi-agent crypto trading system** with three core agents (Broker, Execution, Judge) orchestrated via Temporal workflows. The system supports both **backtesting** and **live trading** with a unified web dashboard.

**Recent Major Accomplishment**: Completed full UI unification with shared components (MarketTicker, EventTimeline) and Week 1 foundation tables. Repository cleanup done. System is now ready for Phase 3 integration work.

---

## ✅ What Was Just Completed (2026-01-04 Session)

### 1. UI Unification (SLOP #1) - COMPLETE
- ✅ **MarketTicker component** integrated into both Backtest Control and Live Trading Monitor tabs
  - Shows real-time prices with auto-refresh every 2 seconds
  - Color-coded price changes (green/red)
  - File: `ui/src/components/MarketTicker.tsx`
- ✅ **EventTimeline component** integrated into both tabs
  - Shows agent events (intent, plan_generated, plan_judged, order_submitted, fill, trade_blocked)
  - Auto-refresh every 3 seconds
  - Filterable by event type
  - File: `ui/src/components/EventTimeline.tsx`
- ✅ Modified `ui/src/components/BacktestControl.tsx` to include both components

### 2. Database Tables (SLOP #6) - COMPLETE
- ✅ Created migration: `app/db/migrations/versions/0002_add_week1_tables.py`
- ✅ Applied migration successfully (verified via `psql -d botdb`)
- ✅ **Four new tables created**:
  - `block_events` - Individual trade block records with trigger_id, reason, detail
  - `risk_allocations` - Risk budget tracking (claimed → used → released)
  - `position_snapshots` - Point-in-time position state
  - `backtest_runs` - Backtest metadata and results storage
- All tables have proper indexes (run_id, correlation_id, timestamps)

### 3. Repository Cleanup (Phase 3.4) - COMPLETE
- ✅ **Deleted garbage**: `test.txt`, `thing.JPG`, `Capture.JPG`, `Capture2.JPG`, `main.py`
- ✅ **Moved to docs/**: `AGENTS.md`, `JUDGE_AGENT_README.md`, `README_metrics.md`
- ✅ **Archived**: `chat-interactions.md`, `Flow After Preferences Are Set.md` → `docs/archive/`
- ✅ **Deprecated**: `run_stack.sh` → `run_stack.sh.DEPRECATED` with README

### 4. Verified Complete
- ✅ **SLOP #2** (Agent event wiring) - Confirmed all agents emit events:
  - Execution agent: `trigger_fired`, `trade_blocked`, `order_submitted`, `fill`
  - Broker agent: `intent`
  - Judge agent: `plan_judged`
- ✅ **SLOP #3** (Materializer fix) - Already queries Temporal for actual status
  - File: `ops_api/materializer.py` has `list_runs_async()` with Temporal integration

---

## 📋 Current System Status

### Working Components
- ✅ React + Vite + TailwindCSS frontend (`ui/`)
- ✅ Backtest Control tab with presets, custom configs, progress monitoring, equity curves
- ✅ Live Trading Monitor tab with positions, fills, blocks, risk budget
- ✅ MarketTicker and EventTimeline components (cross-tab)
- ✅ FastAPI ops-api backend (`ops_api/`) with routers:
  - `/backtests` - Start backtests, get status/results/equity/trades
  - `/live` - Positions, fills, blocks, risk budget, portfolio
  - `/market` - Market ticks
  - `/agents` - Events, workflows, LLM telemetry
  - `/wallets` - Wallet operations (partial)
- ✅ Database tables: All core + 4 new Week 1 tables
- ✅ Agent event emissions wired up and working
- ✅ Temporal workflows: BacktestWorkflow, ExecutionLedgerWorkflow, market streaming

### Known Gaps (What Needs Work)
- ⚠️ **Backtest progress events** - Need to verify BacktestWorkflow emits progress during execution
- ⚠️ **Live daily reports** - Not yet implemented (should match backtest format)
- ⚠️ **Wallet reconciliation UI** - Endpoints exist but no frontend component
- ⚠️ **WebSocket streaming** - Planned but not implemented (currently using polling)
- ⚠️ **Market Monitor tab** - Not yet created (dedicated chart view)
- ⚠️ **Agent Inspector tab** - Not yet created (LLM telemetry, event chains)
- ⚠️ **A/B backtest comparison** - Not yet implemented

---

## 🎯 Prioritized Next Steps

### Priority 1: Verify Backtest Progress Events (HIGH - 2 hours)
**Why**: Need to ensure UI progress bars work correctly during backtest execution

**Tasks**:
1. Read `tools/backtest_execution.py` to verify workflow emits progress
2. Check if `run_simulation_chunk_activity` sends heartbeats with progress data
3. Test: Start a backtest via UI, verify progress bar updates in real-time
4. If missing: Add progress event emissions to workflow
5. Verify `ops_api/routers/backtests.py` GET endpoint returns progress correctly

**Files to check**:
- `tools/backtest_execution.py` - BacktestWorkflow implementation
- `backtesting/activities.py` - run_simulation_chunk_activity
- `ops_api/routers/backtests.py` - GET /backtests/{id} endpoint

**Success criteria**:
- [ ] Backtest progress updates in UI every 2 seconds
- [ ] Progress bar shows accurate percentage (candles_processed / candles_total)
- [ ] Status transitions: queued → running → completed

---

### Priority 2: Implement Live Daily Reports (MEDIUM - 1 day)
**Why**: Provides backtest-style analytics for live trading performance

**Tasks**:
1. Create `services/live_daily_reporter.py`
2. Query `BlockEvent`, `RiskAllocation`, `Order` tables for given date
3. Compute metrics matching backtest daily report format:
   - Trades executed, blocks by reason, risk budget used/available
   - P&L, win rate, risk utilization
4. Add endpoint: `GET /live/daily-report/{date}`
5. Create UI component to display daily reports (reuse backtest format)

**Files to create**:
- `services/live_daily_reporter.py` - Report generation logic
- Add route in `ops_api/routers/live.py`

**Reference**:
- `backtesting/reports.py` - Daily report format to match
- `UI_UNIFICATION_PLAN.md` lines 1103-1163 - Implementation details

**Success criteria**:
- [ ] GET /live/daily-report/2026-01-04 returns structured report
- [ ] Report includes trades, blocks, risk budget, P&L
- [ ] Format matches backtest daily reports

---

### Priority 3: Complete Wallet Reconciliation UI (MEDIUM - 1 day)
**Why**: Critical for live trading safety - detect drift between ledger and exchange

**Current State**:
- CLI reconciliation works: `uv run python -m app.cli.main reconcile run`
- `app/ledger/reconciliation.py` has logic
- No web UI or scheduled reconciliation

**Tasks**:
1. Create `ui/src/components/WalletReconciliation.tsx`
2. Add API endpoints in `ops_api/routers/wallets.py`:
   - `POST /reconcile` - Trigger reconciliation
   - `GET /reconcile/reports?since={ts}` - Historical reports
   - `GET /wallets` - List all wallets with balances
3. Create UI with:
   - Wallet table showing ledger vs exchange balances
   - "Run Reconciliation" button
   - Drift history chart
   - Threshold slider for alerts
4. Wire up to App.tsx as new tab

**Files to create/modify**:
- `ui/src/components/WalletReconciliation.tsx` (new)
- `ops_api/routers/wallets.py` (add endpoints)
- `ui/src/App.tsx` (add tab)

**Success criteria**:
- [ ] User can trigger reconciliation from UI
- [ ] Drift reports displayed in table
- [ ] Historical drift visible over time

---

### Priority 4: WebSocket Integration (LOW - 2 days)
**Why**: Reduce API polling, get true real-time updates for fills/ticks

**Current State**: UI polls every 2-5 seconds via TanStack Query

**Tasks**:
1. Add WebSocket support to `ops_api/app.py`:
   - Connection manager for active clients
   - `/ws/live` endpoint for fills, positions, blocks
   - `/ws/market` endpoint for ticks
2. Modify event emitter to broadcast to WebSocket clients
3. Create `ui/src/hooks/useWebSocket.ts` hook
4. Update `LiveTradingMonitor.tsx` to use WebSocket for fills
5. Update `MarketTicker.tsx` to use WebSocket for ticks

**Reference**: `UI_UNIFICATION_PLAN.md` lines 1006-1089

**Success criteria**:
- [ ] Real-time fill updates without polling
- [ ] Real-time market ticks without polling
- [ ] WebSocket reconnects automatically on disconnect

---

### Priority 5: Agent Inspector Tab (LOW - 2-3 days)
**Why**: Observability into agent decision chains and LLM costs

**Tasks**:
1. Create `ui/src/components/AgentInspector.tsx`
2. Design features:
   - Decision chain timeline (broker intent → execution → judge approval)
   - Event log viewer with filters (type, source, run_id, correlation_id)
   - LLM telemetry table (model, tokens, cost, duration)
   - Workflow status cards (broker, execution, judge)
3. Use existing API endpoints:
   - GET /agents/events
   - GET /agents/llm/telemetry
   - GET /workflows
4. Add as new tab in App.tsx

**Files to create**:
- `ui/src/components/AgentInspector.tsx`

**Success criteria**:
- [ ] Can trace full decision chain via correlation_id
- [ ] LLM costs visible per agent/call
- [ ] Workflow status shows running/paused/completed

---

## 🔧 Development Environment

### Running the System
```bash
# Start full stack (database, temporal, ops-api, worker)
docker compose up

# Start UI dev server (separate terminal)
cd ui && npm run dev

# Access points:
# - UI: http://localhost:5173
# - Ops API: http://localhost:8081
# - Temporal UI: http://localhost:8088
```

### Database
```bash
# Run migrations
uv run alembic upgrade head

# Create new migration
make migrate name="description"

# Check tables
docker compose exec -T db psql -U botuser -d botdb -c "\\dt"
```

### Testing Backtests
```bash
# Via UI: Go to http://localhost:5173, use Backtest Control tab

# Via CLI (if needed):
uv run python -m backtesting.cli --help
```

---

## 📚 Key Documentation

**Must Read**:
- `docs/UI_UNIFICATION_PLAN.md` - Comprehensive UI plan (1380 lines)
- `docs/SLOP_AUDIT.md` - Top 10 sources of slop and fixes
- `CLAUDE.md` - Project overview and conventions
- `docs/ARCHITECTURE.md` - System design

**Reference**:
- `docs/STATUS.md` - Current project status
- `docs/ROADMAP.md` - Long-term vision
- `docs/backlog.md` - Outstanding work items

---

## 🚨 Important Gotchas

### 1. Database User
- **NOT** `postgres` user
- Use `botuser` / `botpass` / `botdb` (see docker-compose.yml)
- Example: `docker compose exec -T db psql -U botuser -d botdb`

### 2. Alembic Migrations
- Template file missing, so autogenerate fails
- **Solution**: Manually create migration files following `0001_initial.py` pattern
- Always set `PYTHONPATH=/home/getzinmw/crypto-trading-agents` when running alembic

### 3. Live Trading Safety
- `LIVE_TRADING_ACK=true` required for destructive operations
- Middleware blocks certain endpoints in live mode without ack
- See `ops_api/app.py` lines 50-104 for safety checks

### 4. Temporal Workers
- **Agent worker** (default): Handles agent workflows and MCP tools
- **Legacy live worker** (optional): Only with `docker compose --profile legacy_live up`
- Worker must be restarted after adding new workflows to `ALLOWED_MODULES`

### 5. Mock vs Production Ledger
- **Mock ledger** (default): `ExecutionLedgerWorkflow` - pure Temporal state
- **Production ledger**: `app/ledger/` - PostgreSQL with double-entry accounting
- Agents use mock by default; production ledger via CLI only
- Wire production ledger to agents: requires `ENABLE_REAL_LEDGER=1` in preferences

---

## 🎨 UI Component Patterns

### TanStack Query Pattern
```typescript
const { data, isLoading } = useQuery({
  queryKey: ['resource', id],
  queryFn: () => api.get(`/resource/${id}`),
  refetchInterval: 2000, // Auto-refresh every 2s
  enabled: !!id // Only query when id exists
})
```

### Event Timeline Filtering
```typescript
const { data: events } = useQuery({
  queryKey: ['events', filterType],
  queryFn: () => agentAPI.getEvents({
    type: filterType || undefined,
    limit: 50
  }),
  refetchInterval: 3000
})
```

### Backtest Status Polling
```typescript
const { data: backtest } = useQuery({
  queryKey: ['backtest', runId],
  queryFn: () => backtestAPI.getStatus(runId),
  enabled: !!runId,
  refetchInterval: (query) => {
    const status = query.state.data?.status
    return status === 'running' ? 2000 : false // Only poll when running
  }
})
```

---

## 🧪 Quick Verification Commands

### Verify Event Emissions
```bash
# Check if agents are emitting events
docker compose logs execution-agent | grep "emit_event"

# Query event store
curl http://localhost:8081/agents/events?limit=10 | jq
```

### Verify Database Tables
```bash
# Check if new tables exist
docker compose exec -T db psql -U botuser -d botdb -c "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"

# Check block_events table
docker compose exec -T db psql -U botuser -d botdb -c "SELECT COUNT(*) FROM block_events;"
```

### Verify UI Components
```bash
# Check if MarketTicker is imported in BacktestControl
grep -n "MarketTicker" ui/src/components/BacktestControl.tsx

# Check if EventTimeline is imported
grep -n "EventTimeline" ui/src/components/BacktestControl.tsx
```

---

## 🏁 Success Metrics

**Phase 1 Foundation**: ✅ COMPLETE (100%)
- Materializer queries Temporal
- DB tables exist and migrated
- Agent events wired

**Phase 2 Frontend**: ✅ COMPLETE (100%)
- React app bootstrapped
- Backtest & Live tabs functional
- Shared components integrated

**Phase 3 Integration**: ⏳ IN PROGRESS (25%)
- [ ] Backtest orchestration wired (Priority 1)
- [ ] Live daily reports implemented (Priority 2)
- [ ] Wallet reconciliation UI (Priority 3)
- [ ] WebSocket streaming (Priority 4)

**Phase 4 Polish**: 📋 NOT STARTED (0%)
- [ ] Testing (unit, integration, e2e)
- [ ] Documentation updates
- [ ] Performance optimization

---

## 💡 Recommended Starting Point

**If you have 2 hours**: Start with **Priority 1** (Verify Backtest Progress Events)
- Quick win to ensure existing UI works perfectly
- Touch minimal files
- Immediate user value

**If you have 1 day**: Do **Priority 1 + Priority 2** (Live Daily Reports)
- Completes core analytics parity between backtest and live
- Sets up foundation for Agent Inspector tab
- High business value

**If you have 2-3 days**: Do **Priority 1-3** (Add Wallet Reconciliation UI)
- Completes Week 1 foundation + critical safety feature
- Makes system production-ready for live trading
- Major milestone

---

## 🤝 Handoff Notes

**What worked well**:
- Incremental approach (UI → DB → Cleanup)
- Using existing components where possible
- Following established patterns in codebase

**What to watch out for**:
- Alembic migration generation (manual creation needed)
- Database connection (use `botuser`, not `postgres`)
- Event correlation IDs (ensure they flow through decision chains)

**Open questions**:
- Should WebSocket be prioritized over live daily reports?
- Do we need A/B backtest comparison in Phase 3 or defer to Phase 4?
- Should we create Market Monitor tab before Agent Inspector?

**Technical debt acknowledged**:
- No WebSocket yet (polling works but not ideal)
- No A/B backtest comparison UI
- Live wallet provider still stubbed out
- No scheduled reconciliation (manual only)

---

**Ready to proceed? Start with Priority 1 and work down the list. Good luck! 🚀**

# UI Unification Plan: Backtest & Live Trading Dashboard

**Status**: ACTIVE IMPLEMENTATION (Phase 1 & 2 completed, Phase 3 in progress)
**Created**: 2026-01-02
**Last Updated**: 2026-01-04
**Priority**: HIGH - Critical for operational visibility

## Current Progress (2026-01-04)

**Completed**:
- ✅ Frontend framework bootstrap (React + Vite + TailwindCSS v4)
- ✅ Backtest Control tab (preset configs, custom params, progress, results, equity curve)
- ✅ Live Trading Monitor tab (positions, fills, blocks, risk budget, portfolio metrics)
- ✅ Tab navigation between Backtest and Live views
- ✅ API proxy configuration (Vite → FastAPI backend)
- ✅ Auto-refresh queries with TanStack Query
- ✅ Market ticker component (cross-tab, real-time prices) - integrated into both tabs
- ✅ Event timeline component (cross-tab, bot events and trade triggers) - integrated into both tabs
- ✅ Week 1 foundation: DB tables (BlockEvent, RiskAllocation, PositionSnapshot, BacktestRun)
- ✅ Database migration applied successfully

**In Progress**:
- ⏳ Phase 3 integration tasks (backtest orchestration, live reports)

**Next**:
- 📋 Market Monitor tab (dedicated chart view)
- 📋 Agent Inspector tab (event chains, LLM telemetry)
- 📋 Wallet Reconciliation tab
- 📋 Live daily reports (matching backtest format)

## Executive Summary

This document outlines a comprehensive plan to build a unified web UI that consolidates fragmented monitoring capabilities across backtesting and live trading. The current repository has **three separate UI implementations** with no shared infrastructure, incomplete event wiring, and significant architectural debt ("slop") that prevents effective monitoring.

**Goal**: Single web application that provides:
1. Backtest initiation and monitoring (predefined + custom configurations)
2. Live market data streaming and visualization
3. Wallet reconciliation UI (test/paper vs live wallets)
4. Multi-agent interaction monitoring (broker ↔ execution ↔ judge)
5. Unified trade tracking (planned → blocked → executed)

---

## Current State: Major Sources of Slop

### 1. **Three Disconnected UI Layers**

| UI Component | Location | Port | Purpose | Status | Slop Level |
|--------------|----------|------|---------|--------|------------|
| **Ops UI** | `ui/index.html` | 8080 | Trading operations (fills, blocks) | 40% complete | HIGH |
| **Dashboard** | `app/dashboard/` | 8081 | Infrastructure (processes, wallets) | 70% complete | MEDIUM |
| **Ticker UI** | `ticker_ui_service.py` | Terminal | Price charts (curses-based) | 85% complete | LOW |

**Slop Issues**:
- No shared state management or authentication
- Different data sources (ops_api vs direct DB vs SSE from MCP)
- Duplicate polling logic (each UI polls independently every 5 seconds)
- No unified theming or component library
- User must open 2 browser tabs + 1 terminal to see full system state

### 2. **Incomplete Event Emission from Agents**

**Problem**: Event infrastructure exists (`ops_api/event_store.py`) but agents don't consistently emit events.

**What Should Happen**:
```python
# Execution Agent should emit:
- "trade_blocked" (with reason, trigger_id, symbol, timestamp)
- "order_submitted" (correlation_id linking to fill)
- "fill" (execution details)

# Broker Agent should emit:
- "intent" (user request or autonomous decision)
- "plan_generated" (new strategy plan created)

# Judge Agent should emit:
- "plan_judged" (approval/rejection with reasoning)
```

**What Actually Happens**:
- Sporadic event emission (grep shows incomplete wiring)
- Some events logged to JSONL files instead of event store
- No correlation IDs linking events across workflow boundaries
- LLM calls partially instrumented via `client_factory.py` but not all callers migrated

**Impact**: UI cannot show complete decision chains (nudge → plan → decision → block/execute).

### 3. **Materializer with Hardcoded Fallbacks**

**File**: `ops_api/materializer.py` lines 105-134

**Slop**:
```python
def list_runs(self) -> List[RunSummary]:
    # For now, synthesize from events; later use Temporal visibility + durable state.
    # ...
    summaries[rid] = RunSummary(
        run_id=rid,
        status="running",  # ← ALWAYS "running", never paused/stopped
        mode="paper",      # ← ALWAYS "paper", ignores actual TRADING_MODE
    )
```

**Issues**:
- Status always shows "running" (no pause/stop detection)
- Mode always shows "paper" (doesn't reflect live trading state)
- Falls back to hardcoded "execution" run if no events found
- No attempt to query Temporal for actual workflow status
- Comment admits this is temporary hack

### 4. **Backtest Visualization: Zero Integration**

**Current State**:
- Backtests run via CLI (`backtest` command)
- Results written to JSON files in `.cache/strategy_plans/{run_id}/daily_reports/`
- Rich metrics computed: equity curve, sharpe ratio, win rate, drawdown, risk budgets, trigger quality
- **NO web endpoints to trigger or monitor backtests**
- **NO UI to visualize equity curves, trade logs, or daily reports**
- **NO API schema for backtest results** (raw Python dicts returned)

**From backlog** (`app/dashboard/server.py` line 109):
```python
"Integrate historical backtesting orchestration into dashboard."  # ← NOT STARTED
```

**Slop**: Complete feature gap. Backtesting is sophisticated but invisible to operations.

### 5. **Trade Tracking Split Across Systems**

**Backtest Path**:
```python
# backtesting/simulator.py
trades_df = pd.DataFrame([
    {"time": ts, "symbol": sym, "side": side, "qty": qty,
     "price": price, "fee": fee, "pnl": pnl,
     "risk_used_abs": risk, "trigger_id": trig_id}
])
```

**Live Path**:
```python
# app/db/models.py
class Order(Base):
    order_id, wallet_id, product_id, side, quantity,
    order_type, status, fill_price, fill_timestamp
```

**Slop**:
- Different schemas (trades_df vs Order table)
- Backtest tracks `risk_used_abs` and `trigger_id`, live doesn't
- No unified query interface to get "all trades" regardless of mode
- Position state: backtest uses `PortfolioTracker.positions` dict, live requires SQL joins

### 6. **Block Reasons Not Persisted**

**Backtest**: Blocks counted in aggregates (`daily_reports[].limit_stats.blocked_by_daily_cap`)

**Live**: Blocks enforced by `trading_core/execution_engine.py` but only visible in ephemeral state:
```python
class DailyExecutionState:
    trades_today: int
    symbol_trades: Dict[str, int]
    skipped_reasons: Dict[str, int]  # ← In-memory only, lost on restart
```

**Slop**:
- NO individual block events with timestamps, trigger IDs, and detailed reasons
- Cannot answer "why was this specific trade at 14:32 blocked?"
- Aggregate counts only visible in verbose debug CSVs (if enabled)
- No API endpoint exposing block history

### 7. **Wallet Reconciliation: Manual CLI Only**

**Current**:
```bash
# CLI operation, no web UI
uv run python -m app.cli.main reconcile run --threshold 0.0001
```

**What it does**:
- `app/ledger/reconciliation.py`: Compare ledger vs Coinbase balances
- Generate `ReconciliationReport` with drift records
- Print to stdout

**Slop**:
- One-time manual operation (no scheduled reconciliation)
- No automatic correction if drift detected
- No historical drift ledger (can't see drift over time)
- No UI to trigger reconciliation or view reports
- No alerting if drift exceeds threshold

### 8. **Agent Interaction Logging Fragmented**

**Current logging**:
1. JSONL daily logs: `logs/{agent}_decisions_YYYY-MM-DD.jsonl`
2. Temporal workflow history (internal, not exported)
3. Event store (incomplete agent wiring)
4. MCP server logs (structured via `AgentLogger`)

**Slop**:
- No correlation IDs linking broker intent → execution decision → judge approval
- Workflow signals not visible in UI
- Agent decision chains require manual log file analysis
- No real-time stream of agent conversations

### 9. **Live Wallet Provider: Stubbed**

**File**: `agents/wallet_provider.py` lines 40-56

```python
class LiveWalletProvider(WalletProvider):
    def get_balance(self, symbol: str) -> Decimal:
        raise RuntimeError("Live wallet provider not yet implemented")

    def debit(self, symbol: str, amount: Decimal) -> None:
        raise RuntimeError("Live wallet provider not yet implemented")
```

**Slop**:
- Interface defined but completely unimplemented
- Gated by runtime mode checks (good) but empty (bad)
- `PaperWalletProvider` works fine, but no live counterpart
- Reconciliation UI would need live wallet integration to be meaningful

### 10. **Repository Root Garbage**

**Files to delete**:
- `test.txt` - Old test error output (no value)
- `thing.JPG`, `Capture.JPG`, `Capture2.JPG` - Screenshots with no context
- `main.py` - Useless entrypoint (just prints greeting)
- `run_stack.sh` - Deprecated tmux approach (superseded by docker-compose)

**Files to move to `/docs/`**:
- `AGENTS.md` (good content, wrong location)
- `JUDGE_AGENT_README.md`
- `README_metrics.md`
- `chat-interactions.md`
- `Flow After Preferences Are Set.md`

---

## Target Architecture: Unified Dashboard

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Web Dashboard                     │
│                    (React/Vue SPA + FastAPI)                 │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Backtest   │   Live       │   Market     │   Agent        │
│   Control    │   Trading    │   Monitor    │   Inspector    │
│              │   Monitor    │              │                │
├──────────────┴──────────────┴──────────────┴────────────────┤
│                  Unified Event Stream API                    │
│                  (WebSocket + REST)                          │
├──────────────────────────────────────────────────────────────┤
│                    Event Store (SQLite)                      │
│   ┌────────────┬────────────┬────────────┬────────────┐    │
│   │  Trades    │  Blocks    │  Risk      │  Agent     │    │
│   │  Ledger    │  Ledger    │  Ledger    │  Events    │    │
│   └────────────┴────────────┴────────────┴────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### Tab/Section Breakdown

#### Tab 1: Backtest Control
**Features**:
- [x] Dropdown for predefined backtest configs (JSON presets) - **COMPLETED**
- [x] Custom config builder (symbol, timeframe, date range, strategy) - **COMPLETED**
- [x] "Start Backtest" button → triggers `BacktestWorkflow` via Temporal - **COMPLETED**
- [x] Progress bar (candles processed / total) - **COMPLETED**
- [x] Results panel: - **COMPLETED**
  - Equity curve chart (line graph)
  - Performance metrics table (sharpe, drawdown, win rate)
  - Daily reports accordion (expand each day)
  - Trade log table (filterable by symbol/side/trigger)
- [x] **Market ticker (persistent)** - Real-time price feed for active symbols - **COMPLETED**
- [x] **Event timeline** - Bot events and trade triggers with timestamps - **COMPLETED**
- [ ] A/B comparison view (select 2 backtests, diff metrics)

**API Endpoints Needed**:
- `POST /backtests` - Start new backtest
- `GET /backtests/{id}` - Get status and results
- `GET /backtests/{id}/equity` - Get equity curve data
- `GET /backtests/{id}/trades` - Get trade log
- `GET /backtests/{id}/daily-reports` - Get daily reports
- `GET /backtests` - List all backtests with filters

#### Tab 2: Live Trading Monitor
**Features**:
- [x] Real-time position table (symbol, qty, entry price, current PnL, mark price) - **COMPLETED**
- [x] Portfolio metrics (cash, equity, day P&L, total P&L) - **COMPLETED**
- [x] Recent fills table (last 50 trades with timestamp, symbol, side, qty, price) - **COMPLETED**
- [x] Risk budget gauge (used / available for day) - **COMPLETED**
- [x] Block reasons summary (counts by reason: daily_cap, risk_budget, etc.) - **COMPLETED**
- [x] Rejected trades log (blocked with reasons and timestamps) - **COMPLETED**
- [x] **Market ticker (persistent)** - Real-time price feed for active symbols - **COMPLETED**
- [x] **Event timeline** - Bot events and trade triggers with timestamps - **COMPLETED**
- [ ] Planned trades queue (approved triggers awaiting execution)

**API Endpoints Needed**:
- `GET /live/positions` - Current positions
- `GET /live/portfolio` - Portfolio summary
- `GET /live/fills?limit=50` - Recent fills
- `GET /live/risk-budget` - Daily risk allocation
- `GET /live/blocks?since={timestamp}` - Block events
- `GET /live/planned-trades` - Pending triggers
- `WebSocket /ws/live` - Real-time position/fill updates

#### Tab 3: Market Monitor
**Features**:
- [ ] Multi-symbol price charts (candlestick or line)
- [ ] Real-time tick feed (price + volume + timestamp)
- [ ] Technical indicators overlay (optional: MA, ATR, rolling high/low)
- [ ] Symbol selector (add/remove symbols from view)
- [ ] Time range selector (1h, 4h, 1d, 1w)

**API Endpoints Needed**:
- `GET /market/ticks?symbol={sym}&since={ts}` - Historical ticks
- `WebSocket /ws/market` - Real-time tick stream
- `GET /market/candles?symbol={sym}&timeframe={tf}&start={ts}` - OHLCV data

#### Tab 4: Agent Inspector
**Features**:
- [ ] Decision chain timeline (broker intent → execution decision → judge approval)
- [ ] Event log viewer (all events with filters: type, source, run_id, correlation_id)
- [ ] Agent conversation viewer (messages between agents with timestamps)
- [ ] LLM telemetry (model, tokens, cost, duration per call)
- [ ] Workflow status cards (broker, execution, judge workflows with status)
- [ ] Signal controls (pause/resume workflows, rotate plan, trigger evaluation)

**API Endpoints Needed**:
- `GET /events?type={type}&source={source}&run_id={id}&since={ts}` - Filtered events
- `GET /agent-logs?agent={name}&date={YYYY-MM-DD}` - JSONL logs
- `GET /llm/telemetry?since={ts}` - LLM call stats
- `GET /workflows` - List workflows with status
- `POST /workflows/{id}/pause` - Pause workflow
- `POST /workflows/{id}/resume` - Resume workflow
- `POST /workflows/{id}/rotate-plan` - Regenerate plan

#### Tab 5: Wallet Reconciliation
**Features**:
- [ ] Wallet list (ID, name, currency, ledger balance, Coinbase balance, drift)
- [ ] "Run Reconciliation" button
- [ ] Drift history chart (drift over time for each wallet/currency)
- [ ] Threshold slider (set drift alert threshold)
- [ ] Manual correction controls (if drift > threshold, approve/reject correction)
- [ ] Transaction log (recent debits/credits with source)

**API Endpoints Needed**:
- `GET /wallets` - List all wallets
- `GET /wallets/{id}/balance` - Current balance
- `POST /reconcile` - Trigger reconciliation
- `GET /reconcile/reports?since={ts}` - Historical reports
- `GET /wallets/{id}/transactions?limit=100` - Transaction history
- `POST /wallets/{id}/correct-drift` - Manual drift correction

---

## Implementation Plan

### Phase 1: Foundation (Critical Path)

#### 1.1 Unify Event Emission (HIGH PRIORITY)
**Effort**: 3-4 days
**Files to modify**:
- `agents/execution_agent_client.py`
- `agents/broker_agent_client.py`
- `agents/judge_agent_client.py`
- `agents/event_emitter.py` (enhance if needed)

**Tasks**:
```python
# agents/execution_agent_client.py
from agents.event_emitter import EventEmitter

class ExecutionAgentClient:
    def __init__(self):
        self.event_emitter = EventEmitter()

    async def _handle_trade_block(self, trigger, reason, detail):
        await self.event_emitter.emit({
            "type": "trade_blocked",
            "source": "execution_agent",
            "run_id": self.run_id,
            "correlation_id": trigger.correlation_id,
            "payload": {
                "trigger_id": trigger.id,
                "symbol": trigger.symbol,
                "side": trigger.side,
                "qty": trigger.qty,
                "reason": reason.value,
                "detail": detail,
                "timestamp": datetime.utcnow().isoformat()
            }
        })

    async def _handle_order_submission(self, trigger, order_id):
        await self.event_emitter.emit({
            "type": "order_submitted",
            "source": "execution_agent",
            "correlation_id": trigger.correlation_id,
            "payload": {
                "trigger_id": trigger.id,
                "order_id": order_id,
                "symbol": trigger.symbol,
                "side": trigger.side,
                "qty": trigger.qty
            }
        })
```

**Validation**:
- [ ] Query event store: `SELECT * FROM events WHERE type='trade_blocked' LIMIT 10`
- [ ] Verify correlation_ids link order_submitted → fill events
- [ ] Check all agent sources emit events consistently

#### 1.2 Fix Materializer (MEDIUM PRIORITY)
**Effort**: 1-2 days
**Files to modify**:
- `ops_api/materializer.py`
- Add: `ops_api/temporal_client.py` (helper to query Temporal)

**Tasks**:
```python
# ops_api/temporal_client.py
from temporalio.client import Client
from agents.workflows.broker_agent_workflow import BrokerAgentWorkflow

async def get_workflow_status(workflow_id: str) -> dict:
    client = await Client.connect("localhost:7233")
    handle = client.get_workflow_handle(workflow_id)

    try:
        desc = await handle.describe()
        return {
            "status": desc.status.name.lower(),  # running, paused, completed, failed
            "run_id": desc.run_id,
            "type": desc.workflow_type
        }
    except Exception as e:
        return {"status": "unknown", "error": str(e)}

# ops_api/materializer.py
from ops_api.temporal_client import get_workflow_status
from agents.runtime_mode import get_runtime_mode

async def list_runs(self) -> List[RunSummary]:
    events = self.store.list_events(limit=500)
    summaries: dict[str, RunSummary] = {}

    # Get actual runtime mode
    runtime = get_runtime_mode()
    actual_mode = runtime.mode  # "dev", "paper", or "live"

    for event in events:
        rid = event.run_id or "default"
        if rid not in summaries:
            # Query Temporal for actual status
            wf_status = await get_workflow_status(f"execution-agent-{rid}")

            summaries[rid] = RunSummary(
                run_id=rid,
                status=wf_status.get("status", "unknown"),  # ← ACTUAL status
                mode=actual_mode,  # ← ACTUAL mode from runtime config
                last_updated=event.ts
            )

    # Only fall back if truly no events found
    if not summaries:
        summaries["default"] = RunSummary(
            run_id="default",
            status="no_events",
            mode=actual_mode,
            last_updated=datetime.utcnow()
        )

    return list(summaries.values())
```

**Validation**:
- [ ] Start a workflow, pause it via Temporal UI, verify status shows "paused"
- [ ] Set `RUN_MODE=live` and `LIVE_TRADING_ACK=true`, verify mode shows "live"
- [ ] Check that default fallback only triggers when event store is empty

#### 1.3 Add Missing Database Tables (✅ COMPLETED 2026-01-04)
**Effort**: 1 day
**Files modified**:
- `app/db/models.py` - Tables already existed
- Created migration: `app/db/migrations/versions/0002_add_week1_tables.py`
- Applied migration successfully

**New Tables** (all created and verified):
```python
# app/db/models.py

class BlockEvent(Base):
    """Individual trade block events with full context."""
    __tablename__ = "block_events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    run_id = Column(String, nullable=False, index=True)
    correlation_id = Column(String, nullable=True, index=True)
    trigger_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # buy/sell
    qty = Column(Numeric(precision=18, scale=8), nullable=False)
    reason = Column(String, nullable=False)  # BlockReason enum value
    detail = Column(Text, nullable=True)  # JSON detail

    __table_args__ = (
        Index("ix_block_events_ts_reason", "timestamp", "reason"),
    )

class RiskAllocation(Base):
    """Risk budget tracking (claimed → used → released)."""
    __tablename__ = "risk_allocations"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    correlation_id = Column(String, nullable=True, index=True)
    trigger_id = Column(String, nullable=False)
    claim_timestamp = Column(DateTime, nullable=False)
    claim_amount = Column(Numeric(precision=18, scale=8), nullable=False)
    release_timestamp = Column(DateTime, nullable=True)
    release_amount = Column(Numeric(precision=18, scale=8), nullable=True)
    status = Column(String, nullable=False)  # claimed, used, released, expired

class PositionSnapshot(Base):
    """Point-in-time position state (for live trading)."""
    __tablename__ = "position_snapshots"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    run_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    qty = Column(Numeric(precision=18, scale=8), nullable=False)
    avg_entry_price = Column(Numeric(precision=18, scale=8), nullable=False)
    mark_price = Column(Numeric(precision=18, scale=8), nullable=True)
    unrealized_pnl = Column(Numeric(precision=18, scale=8), nullable=True)

    __table_args__ = (
        Index("ix_position_snapshots_ts_symbol", "timestamp", "symbol"),
    )

class BacktestRun(Base):
    """Backtest metadata and configuration."""
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True, nullable=False)
    config = Column(Text, nullable=False)  # JSON config
    status = Column(String, nullable=False)  # queued, running, completed, failed
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    candles_total = Column(Integer, nullable=True)
    candles_processed = Column(Integer, nullable=True)
    results = Column(Text, nullable=True)  # JSON results summary
```

**Validation**:
- [ ] Run migration: `make migrate name="add_unified_tracking_tables"`
- [ ] Verify tables exist: `psql -d botdb -c "\dt"`
- [ ] Test insert: `INSERT INTO block_events (...) VALUES (...)`

#### 1.4 Create Unified API Endpoints (HIGH PRIORITY)
**Effort**: 3-4 days
**Files to create**:
- `ops_api/routers/backtests.py`
- `ops_api/routers/live.py`
- `ops_api/routers/market.py`
- `ops_api/routers/agents.py`
- `ops_api/routers/wallets.py`

**Example: Backtest Router**
```python
# ops_api/routers/backtests.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from temporalio.client import Client
from agents.workflows.backtest_workflow import BacktestWorkflow

router = APIRouter(prefix="/backtests", tags=["backtests"])

class BacktestConfig(BaseModel):
    symbols: list[str]
    timeframe: str
    start_date: str
    end_date: str
    initial_cash: float
    strategy: Optional[str] = "baseline"

class BacktestCreateResponse(BaseModel):
    run_id: str
    status: str

@router.post("", response_model=BacktestCreateResponse)
async def start_backtest(config: BacktestConfig, background_tasks: BackgroundTasks):
    """Start a new backtest run."""
    run_id = f"backtest-{uuid4()}"

    # Save to DB
    async with get_db() as db:
        backtest = BacktestRun(
            run_id=run_id,
            config=config.json(),
            status="queued",
            started_at=None
        )
        db.add(backtest)
        await db.commit()

    # Start Temporal workflow
    client = await Client.connect("localhost:7233")
    await client.start_workflow(
        BacktestWorkflow.run,
        config.dict(),
        id=run_id,
        task_queue="mcp-tools"
    )

    return BacktestCreateResponse(run_id=run_id, status="queued")

@router.get("/{run_id}")
async def get_backtest(run_id: str):
    """Get backtest status and results."""
    async with get_db() as db:
        backtest = await db.execute(
            select(BacktestRun).where(BacktestRun.run_id == run_id)
        )
        backtest = backtest.scalar_one_or_none()

        if not backtest:
            raise HTTPException(404, "Backtest not found")

        return {
            "run_id": backtest.run_id,
            "status": backtest.status,
            "progress": (backtest.candles_processed / backtest.candles_total * 100)
                        if backtest.candles_total else 0,
            "results": json.loads(backtest.results) if backtest.results else None
        }

@router.get("/{run_id}/equity")
async def get_equity_curve(run_id: str):
    """Get equity curve data for charting."""
    # Load from results JSON or compute from trades
    async with get_db() as db:
        backtest = await db.execute(
            select(BacktestRun).where(BacktestRun.run_id == run_id)
        )
        backtest = backtest.scalar_one_or_none()

        if not backtest or not backtest.results:
            raise HTTPException(404, "Results not available")

        results = json.loads(backtest.results)
        equity_curve = results.get("equity_curve", [])

        return {
            "timestamps": [point["time"] for point in equity_curve],
            "values": [point["equity"] for point in equity_curve]
        }
```

**Similar patterns for**:
- `live.py`: `/live/positions`, `/live/fills`, `/live/risk-budget`, `/live/blocks`
- `market.py`: `/market/ticks`, `/market/candles`
- `agents.py`: `/events`, `/workflows`, `/llm/telemetry`
- `wallets.py`: `/wallets`, `/reconcile`

**Validation**:
- [ ] Test each endpoint with `curl` or Postman
- [ ] Verify OpenAPI docs at `http://localhost:8080/docs`
- [ ] Check response schemas match Pydantic models

### Phase 2: Frontend Development

#### 2.0 Shared Components (Cross-Tab)
**Status**: ✅ COMPLETED (2026-01-04)
**Effort**: 2-3 days

These components are now visible in both Backtest Control and Live Trading Monitor tabs.

**Component 1: MarketTicker**
- **Purpose**: Real-time price ticker showing live market data
- **Data source**: `/market/ticks` endpoint (polls every 1-2 seconds)
- **Features**:
  - Horizontal ticker bar displaying active symbols
  - Current price, volume, timestamp for each symbol
  - Color-coded price changes (green for up, red for down)
  - Compact format (fits at top of page)
  - Auto-updates via TanStack Query `refetchInterval`
- **UI position**: Top of both Backtest and Live tabs (persistent)
- **Implementation**:
  ```typescript
  // src/components/MarketTicker.tsx
  export function MarketTicker() {
    const { data: ticks } = useQuery({
      queryKey: ['market-ticks'],
      queryFn: () => api.get('/market/ticks?limit=10'),
      refetchInterval: 2000 // 2 seconds
    })
    // Render horizontal ticker with price updates
  }
  ```

**Component 2: EventTimeline**
- **Purpose**: Chronological timeline of bot events and trade triggers
- **Data source**: `/agents/events` endpoint with filtering
- **Event types displayed**:
  - `intent` - Broker agent requests
  - `plan_generated` - Strategy plans created
  - `plan_judged` - Judge agent decisions with scores
  - `order_submitted` - Trade orders placed
  - `fill` - Executed trades with fill prices
  - `trade_blocked` - Blocked trades with reasons and details
- **Features**:
  - Timeline view with timestamps and event details
  - Color-coded by event type (e.g., green for fills, red for blocks)
  - Filterable by event type, source, time range
  - Shows correlation IDs to link related events
  - Auto-refreshes every 3-5 seconds
- **UI position**: Side panel or bottom section in both tabs
- **Implementation**:
  ```typescript
  // src/components/EventTimeline.tsx
  export function EventTimeline() {
    const { data: events } = useQuery({
      queryKey: ['agent-events'],
      queryFn: () => api.get('/agents/events?limit=50'),
      refetchInterval: 3000 // 3 seconds
    })
    // Render timeline with event cards
  }
  ```

**Integration**:
- Both components will be imported into `BacktestControl.tsx` and `LiveTradingMonitor.tsx`
- Shared positioning/styling for consistency across tabs
- API client functions added to `src/lib/api.ts`

#### 2.1 Choose Framework and Bootstrap
**Status**: COMPLETED
**Effort**: 1 day
**Framework**: React + Vite + TailwindCSS v4

**Completed Tasks**:
```bash
cd ui
npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss @tailwindcss/postcss autoprefixer
npm install @tanstack/react-query axios recharts
npm install lucide-react clsx tailwind-merge
```

**Structure**:
```
ui/dashboard/
├── src/
│   ├── components/
│   │   ├── BacktestControl.tsx
│   │   ├── LiveTradingMonitor.tsx
│   │   ├── MarketMonitor.tsx
│   │   ├── AgentInspector.tsx
│   │   └── WalletReconciliation.tsx
│   ├── hooks/
│   │   ├── useBacktests.ts
│   │   ├── useLiveData.ts
│   │   ├── useMarketTicks.ts
│   │   └── useWebSocket.ts
│   ├── lib/
│   │   ├── api.ts
│   │   └── utils.ts
│   └── App.tsx
└── package.json
```

#### 2.2 Implement Core Components
**Status**: PARTIALLY COMPLETED
**Effort**: 5-7 days

**Completed Components**:
- [x] `BacktestControl.tsx` - Fully functional with preset configs, custom parameters, progress monitoring, and equity curve visualization
- [x] `LiveTradingMonitor.tsx` - Real-time positions, fills, blocks, risk budget gauge, and portfolio metrics
- [x] Tab navigation in `App.tsx` - Clean tab switching between Backtest and Live views

**Completed Shared Components**:
- [x] `MarketTicker.tsx` - Real-time price ticker (integrated into both tabs) - **COMPLETED**
- [x] `EventTimeline.tsx` - Bot event timeline (integrated into both tabs) - **COMPLETED**

**Pending Components**:
- [ ] `MarketMonitor.tsx` - Dedicated market data tab (future work)
- [ ] `AgentInspector.tsx` - Dedicated agent monitoring tab (future work)
- [ ] `WalletReconciliation.tsx` - Wallet reconciliation tab (future work)

**Example: BacktestControl.tsx (COMPLETED)**
```typescript
import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend } from 'recharts'
import { api } from '@/lib/api'

interface BacktestConfig {
  symbols: string[]
  timeframe: string
  start_date: string
  end_date: string
  initial_cash: number
  strategy?: string
}

export function BacktestControl() {
  const [config, setConfig] = useState<BacktestConfig>({
    symbols: ['BTC-USD'],
    timeframe: '15m',
    start_date: '2024-01-01',
    end_date: '2024-01-31',
    initial_cash: 10000,
    strategy: 'baseline'
  })

  const [selectedRun, setSelectedRun] = useState<string | null>(null)

  // Mutation to start backtest
  const startBacktest = useMutation({
    mutationFn: (config: BacktestConfig) =>
      api.post('/backtests', config),
    onSuccess: (data) => {
      setSelectedRun(data.run_id)
    }
  })

  // Query backtest status (poll every 2s when running)
  const { data: backtest } = useQuery({
    queryKey: ['backtest', selectedRun],
    queryFn: () => api.get(`/backtests/${selectedRun}`),
    enabled: !!selectedRun,
    refetchInterval: (data) =>
      data?.status === 'running' ? 2000 : false
  })

  // Query equity curve when complete
  const { data: equity } = useQuery({
    queryKey: ['equity', selectedRun],
    queryFn: () => api.get(`/backtests/${selectedRun}/equity`),
    enabled: backtest?.status === 'completed'
  })

  return (
    <div className="p-6 space-y-6">
      <div className="grid grid-cols-2 gap-4">
        {/* Config Form */}
        <div className="border rounded-lg p-4">
          <h3 className="font-bold mb-4">Backtest Configuration</h3>

          <div className="space-y-3">
            <div>
              <label>Symbols (comma-separated)</label>
              <input
                value={config.symbols.join(',')}
                onChange={(e) => setConfig({
                  ...config,
                  symbols: e.target.value.split(',').map(s => s.trim())
                })}
                className="w-full border rounded px-3 py-2"
              />
            </div>

            <div>
              <label>Timeframe</label>
              <select
                value={config.timeframe}
                onChange={(e) => setConfig({...config, timeframe: e.target.value})}
                className="w-full border rounded px-3 py-2"
              >
                <option value="1m">1 minute</option>
                <option value="5m">5 minutes</option>
                <option value="15m">15 minutes</option>
                <option value="1h">1 hour</option>
                <option value="4h">4 hours</option>
              </select>
            </div>

            <div>
              <label>Date Range</label>
              <div className="flex gap-2">
                <input
                  type="date"
                  value={config.start_date}
                  onChange={(e) => setConfig({...config, start_date: e.target.value})}
                  className="flex-1 border rounded px-3 py-2"
                />
                <span className="self-center">to</span>
                <input
                  type="date"
                  value={config.end_date}
                  onChange={(e) => setConfig({...config, end_date: e.target.value})}
                  className="flex-1 border rounded px-3 py-2"
                />
              </div>
            </div>

            <div>
              <label>Initial Cash</label>
              <input
                type="number"
                value={config.initial_cash}
                onChange={(e) => setConfig({...config, initial_cash: +e.target.value})}
                className="w-full border rounded px-3 py-2"
              />
            </div>

            <button
              onClick={() => startBacktest.mutate(config)}
              disabled={startBacktest.isPending}
              className="w-full bg-blue-600 text-white rounded py-2 hover:bg-blue-700 disabled:opacity-50"
            >
              {startBacktest.isPending ? 'Starting...' : 'Start Backtest'}
            </button>
          </div>
        </div>

        {/* Results Panel */}
        <div className="border rounded-lg p-4">
          <h3 className="font-bold mb-4">Results</h3>

          {!selectedRun && (
            <p className="text-gray-500">Configure and start a backtest to see results</p>
          )}

          {backtest && backtest.status === 'running' && (
            <div>
              <p>Status: Running</p>
              <div className="mt-2 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{width: `${backtest.progress}%`}}
                />
              </div>
              <p className="text-sm text-gray-600 mt-1">
                {backtest.progress.toFixed(1)}% complete
              </p>
            </div>
          )}

          {backtest?.status === 'completed' && backtest.results && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 rounded p-3">
                  <div className="text-sm text-gray-600">Return</div>
                  <div className="text-2xl font-bold">
                    {backtest.results.equity_return_pct.toFixed(2)}%
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-3">
                  <div className="text-sm text-gray-600">Sharpe Ratio</div>
                  <div className="text-2xl font-bold">
                    {backtest.results.sharpe_ratio.toFixed(2)}
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-3">
                  <div className="text-sm text-gray-600">Max Drawdown</div>
                  <div className="text-2xl font-bold text-red-600">
                    {backtest.results.max_drawdown_pct.toFixed(2)}%
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-3">
                  <div className="text-sm text-gray-600">Win Rate</div>
                  <div className="text-2xl font-bold">
                    {(backtest.results.win_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {equity && (
                <div className="mt-4">
                  <h4 className="font-semibold mb-2">Equity Curve</h4>
                  <LineChart width={400} height={200} data={equity.timestamps.map((t, i) => ({
                    time: t,
                    equity: equity.values[i]
                  }))}>
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="equity" stroke="#2563eb" />
                  </LineChart>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

**Similar patterns for**:
- `LiveTradingMonitor.tsx`: Position table, fills table, risk gauge
- `MarketMonitor.tsx`: Candlestick charts, real-time ticks
- `AgentInspector.tsx`: Event timeline, LLM telemetry
- `WalletReconciliation.tsx`: Wallet table, drift chart, reconcile button

#### 2.3 WebSocket Integration
**Effort**: 2-3 days

**Backend**: Add WebSocket support to ops_api
```python
# ops_api/app.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# In event recording, broadcast to websocket clients
@app.post("/signal/{name}")
async def record_signal(name: str, payload: dict):
    # ... existing logic ...

    # Broadcast to websocket clients
    await manager.broadcast({
        "type": name,
        "payload": payload,
        "timestamp": datetime.utcnow().isoformat()
    })

    return {"status": "recorded"}
```

**Frontend**: WebSocket hook
```typescript
// src/hooks/useWebSocket.ts
import { useEffect, useState } from 'react'

export function useWebSocket<T>(url: string, topic: string) {
  const [data, setData] = useState<T | null>(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const ws = new WebSocket(url)

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)
      if (msg.type === topic) {
        setData(msg.payload)
      }
    }

    return () => ws.close()
  }, [url, topic])

  return { data, connected }
}

// Usage in component
const { data: latestFill } = useWebSocket('ws://localhost:8080/ws/live', 'fill')
```

### Phase 3: Integration & Polish

#### 3.1 Wire Backtest Orchestration
**Effort**: 2 days

**Tasks**:
- [ ] Update `agents/workflows/backtest_workflow.py` to accept web config
- [ ] Emit progress events during backtest (`candles_processed` / `candles_total`)
- [ ] Persist results to `BacktestRun` table on completion
- [ ] Handle failures gracefully (mark as "failed", store error message)

#### 3.2 Implement Live Daily Reports
**Effort**: 2-3 days
**Create**: `services/live_daily_reporter.py`

**Logic**:
```python
# services/live_daily_reporter.py
from app.db.models import Order, BlockEvent, RiskAllocation
from backtesting.reports import generate_daily_report

async def generate_live_daily_report(run_id: str, date: str) -> dict:
    """Generate backtest-style daily report for live trading."""
    async with get_db() as db:
        # Get all orders for this day
        orders = await db.execute(
            select(Order)
            .where(Order.timestamp >= date)
            .where(Order.timestamp < date + timedelta(days=1))
        )
        orders = orders.scalars().all()

        # Get block events
        blocks = await db.execute(
            select(BlockEvent)
            .where(BlockEvent.timestamp >= date)
            .where(BlockEvent.timestamp < date + timedelta(days=1))
        )
        blocks = blocks.scalars().all()

        # Get risk allocations
        risk = await db.execute(
            select(RiskAllocation)
            .where(RiskAllocation.claim_timestamp >= date)
            .where(RiskAllocation.claim_timestamp < date + timedelta(days=1))
        )
        risk = risk.scalars().all()

        # Compute report (mimic backtest format)
        return {
            "date": date,
            "trades": len([o for o in orders if o.status == "filled"]),
            "blocks": len(blocks),
            "block_breakdown": {
                reason: len([b for b in blocks if b.reason == reason])
                for reason in set(b.reason for b in blocks)
            },
            "risk_budget": {
                "used_abs": sum(r.claim_amount for r in risk),
                "budget_abs": 1000,  # TODO: from config
                "used_pct": sum(r.claim_amount for r in risk) / 1000 * 100
            },
            # ... more fields matching backtest report format
        }
```

**Endpoint**:
```python
@app.get("/live/daily-report/{date}")
async def get_live_daily_report(date: str):
    report = await generate_live_daily_report("live-run", date)
    return report
```

#### 3.3 Complete Wallet Reconciliation UI
**Effort**: 2 days

**Tasks**:
- [ ] Implement `LiveWalletProvider` (integrate with `app/coinbase/client.py`)
- [ ] Add scheduled reconciliation (cron job or Temporal workflow)
- [ ] Create drift history table and populate via reconciliation runs
- [ ] Add manual correction endpoint with approval workflow

#### 3.4 Clean Up Repository
**Effort**: 1 day

**Tasks**:
```bash
# Delete garbage
rm test.txt thing.JPG Capture.JPG Capture2.JPG main.py

# Move docs
mv AGENTS.md docs/
mv JUDGE_AGENT_README.md docs/
mv README_metrics.md docs/
mv chat-interactions.md docs/archive/
mv "Flow After Preferences Are Set.md" docs/archive/

# Mark deprecated
echo "# DEPRECATED: Use docker-compose.yml instead" > run_stack.sh.DEPRECATED
mv run_stack.sh run_stack.sh.DEPRECATED

# Update README with clear navigation
# Update CLAUDE.md to reference new UI
```

---

## Testing Strategy

### Unit Tests
- [ ] Event emission: Mock event store, verify all agents emit correctly
- [ ] Materializer: Mock Temporal client, verify status/mode resolution
- [ ] API endpoints: Test request/response schemas with pytest fixtures

### Integration Tests
- [ ] Start backtest via API, poll status, verify results
- [ ] Emit live trade events, query via API, verify response
- [ ] WebSocket: Connect client, emit event, verify client receives

### End-to-End Tests
- [ ] Full backtest flow: config → start → monitor → results → chart
- [ ] Live monitoring: start agent → emit fills → see in UI
- [ ] Reconciliation: drift wallet → run reconcile → see drift report

---

## Success Criteria

### Must Have (MVP)
- [ ] Single web UI consolidating all monitoring
- [ ] Backtest can be started and monitored via web
- [ ] Live fills visible in real-time table
- [ ] Block reasons visible with counts and details
- [ ] Agent events queryable via API
- [ ] No more hardcoded "running"/"paper" in materializer

### Should Have (V1.1)
- [ ] WebSocket streaming for fills and ticks
- [ ] Live daily reports matching backtest format
- [ ] Wallet reconciliation UI with drift history
- [ ] Agent decision chain timeline visualization
- [ ] A/B backtest comparison view

### Nice to Have (V1.2)
- [ ] Equity curve overlays (compare multiple backtests)
- [ ] Candlestick charts with trade markers
- [ ] LLM cost breakdown per agent
- [ ] Automated reconciliation with alerting
- [ ] Workflow pause/resume controls in UI

---

## Migration Path

### From Current State
1. ~~**Week 1**: Phase 1 foundation (event wiring, materializer fix, new tables)~~ - **COMPLETED 2026-01-04**
2. ~~**Week 2**: Phase 1 API endpoints + Phase 2 frontend bootstrap~~ - **COMPLETED**
3. ~~**Week 3**: Phase 2 core components (backtest control, live monitor)~~ - **COMPLETED**
4. **Current Status (2026-01-04)**:
   - ✅ React + Vite + TailwindCSS frontend bootstrapped
   - ✅ Backtest Control tab fully functional
   - ✅ Live Trading Monitor tab fully functional
   - ✅ Tab navigation implemented
   - ✅ Market ticker + Event timeline shared components **COMPLETED**
   - ✅ Week 1 DB tables (BlockEvent, RiskAllocation, PositionSnapshot, BacktestRun) **COMPLETED**
   - ⏳ **Next up**: Phase 3 integration tasks
5. **Week 4+**: Phase 3 integration (backtest orchestration, live reports, reconciliation)
6. **Week 5+**: Testing, polish, cleanup, documentation

### Deprecation Timeline
- **Immediate**: Stop using `ticker_ui_service.py` once market monitor tab is ready
- **After 2 weeks**: Archive `ui/index.html` once new dashboard is feature-complete
- **After 1 month**: Decide whether to keep `app/dashboard/` for infrastructure or merge

---

## Appendix: Code Snippets

### Example Event Correlation
```python
# Create correlation ID when broker generates intent
correlation_id = str(uuid4())

# Broker emits intent
await emit_event({
    "type": "intent",
    "correlation_id": correlation_id,
    "payload": {"action": "evaluate_btc"}
})

# Execution agent receives intent, generates trigger
trigger_id = f"trigger-{uuid4()}"
await emit_event({
    "type": "trigger_generated",
    "correlation_id": correlation_id,
    "payload": {"trigger_id": trigger_id, "symbol": "BTC-USD"}
})

# If blocked
await emit_event({
    "type": "trade_blocked",
    "correlation_id": correlation_id,
    "payload": {"trigger_id": trigger_id, "reason": "risk_budget"}
})

# If executed
order_id = await place_order(trigger)
await emit_event({
    "type": "order_submitted",
    "correlation_id": correlation_id,
    "payload": {"trigger_id": trigger_id, "order_id": order_id}
})

# When filled
await emit_event({
    "type": "fill",
    "correlation_id": correlation_id,
    "payload": {"order_id": order_id, "fill_price": 42000, "qty": 0.1}
})
```

### Example Unified Trade Query
```python
@app.get("/trades")
async def get_all_trades(
    run_id: Optional[str] = None,
    symbol: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
):
    """Unified trade query across backtest and live."""
    async with get_db() as db:
        # Try live orders first
        query = select(Order)
        if run_id:
            query = query.where(Order.run_id == run_id)
        if symbol:
            query = query.where(Order.product_id == symbol)
        if start:
            query = query.where(Order.timestamp >= start)
        if end:
            query = query.where(Order.timestamp <= end)

        orders = await db.execute(query)
        live_trades = [
            {
                "timestamp": o.timestamp,
                "symbol": o.product_id,
                "side": o.side,
                "qty": float(o.quantity),
                "price": float(o.fill_price) if o.fill_price else None,
                "source": "live"
            }
            for o in orders.scalars().all()
        ]

        # If backtest run_id, load from backtest results
        if run_id and run_id.startswith("backtest-"):
            backtest = await db.execute(
                select(BacktestRun).where(BacktestRun.run_id == run_id)
            )
            backtest = backtest.scalar_one_or_none()
            if backtest and backtest.results:
                results = json.loads(backtest.results)
                backtest_trades = [
                    {
                        "timestamp": t["time"],
                        "symbol": t["symbol"],
                        "side": t["side"],
                        "qty": t["qty"],
                        "price": t["price"],
                        "source": "backtest"
                    }
                    for t in results.get("trades", [])
                ]
                return {"trades": backtest_trades}

        return {"trades": live_trades}
```

---

## Next Steps for Implementation Agent

1. **Read this document fully** - Understand all identified slop and proposed fixes
2. **Prioritize Phase 1** - Foundation work is critical path
3. **Start with event emission** - Fix agent wiring first (highest ROI)
4. **Test incrementally** - Validate each endpoint before moving to next
5. **Document as you go** - Update CLAUDE.md with new endpoints/components
6. **Ask questions** - If any requirement is unclear, clarify before coding

**Success looks like**: A single web dashboard where a user can start a backtest, monitor live trading, see all agent decisions, and reconcile wallets—all without opening multiple terminals or parsing log files.

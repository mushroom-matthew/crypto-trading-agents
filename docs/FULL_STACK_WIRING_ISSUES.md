# Full Stack Wiring Issues & Fix Plan

## Executive Summary

The backtest system has a **critical data format mismatch** between the Temporal workflow persistence layer and the API endpoints that read from it. This is the primary reason why running backtests via the UI doesn't produce meaningful results.

## Architecture Diagram (Current)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ UI (port 3000)                                                              │
│ BacktestControl.tsx                                                         │
│   ├─ POST /backtests         → starts backtest                              │
│   ├─ GET /backtests/{id}     → polls status ✅                              │
│   ├─ GET /backtests/{id}/results → gets metrics ⚠️ (falls back to disk)    │
│   ├─ GET /backtests/{id}/equity  → gets chart data ❌ BROKEN               │
│   ├─ GET /backtests/{id}/trades  → gets trade log ❌ BROKEN                │
│   └─ GET /backtests/{id}/playback/candles → ❌ BROKEN                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │ (vite proxy)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Ops API (port 8081)                                                         │
│ ops_api/routers/backtests.py                                                │
│   ├─ start_backtest() → starts Temporal workflow ✅                         │
│   ├─ get_backtest_status() → queries workflow ✅                            │
│   ├─ get_backtest_results() → queries workflow OR disk cache ✅             │
│   ├─ get_equity_curve() → reads disk cache ❌ FORMAT MISMATCH              │
│   ├─ get_backtest_trades() → reads disk cache ❌ FORMAT MISMATCH           │
│   └─ get_playback_candles() → reads disk cache ❌ FORMAT MISMATCH          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Temporal (port 7233)                                                        │
│ BacktestWorkflow                                                            │
│   ├─ run() → orchestrates backtest ✅                                       │
│   ├─ get_status() → query returns status dict ✅                            │
│   ├─ get_results() → query returns metrics dict ✅                          │
│   └─ persist_results_activity() → saves to disk ❌ WRONG FORMAT            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Disk Cache (.cache/backtests/*.pkl)                                         │
│                                                                             │
│ WORKFLOW SAVES:                    │  API EXPECTS:                          │
│ {                                  │  {                                     │
│   "run_id": "...",                 │    "result": PortfolioBacktestResult(  │
│   "equity_curve": [...],  ← list   │      equity_curve=pd.Series(...),      │
│   "trades": [...],        ← list   │      trades=pd.DataFrame(...),         │
│   ...                              │    ),                                  │
│ }                                  │    "config": {...},                    │
│                                    │    "status": "completed",              │
│                                    │  }                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Critical Issues Found

### 1. **DATA FORMAT MISMATCH** (Severity: CRITICAL)

**Location:** `tools/backtest_execution.py:277` vs `ops_api/routers/backtests.py:455`

**Problem:**
- The workflow saves a flat dict with `equity_curve` as a list of dicts
- The API expects `cached["result"]` to be a `PortfolioBacktestResult` object with `.equity_curve` as a pandas Series

**Impact:** `/equity`, `/trades`, `/playback/*` endpoints all fail with errors

**Fix Required:** Either:
- A) Modify `persist_results_activity` to save in the format the API expects
- B) Modify API endpoints to handle the workflow's output format

### 2. **DISK CACHE NOT SHARED ACROSS CONTAINERS** (Severity: HIGH)

**Problem:**
- `persist_results_activity` writes to `.cache/backtests/` inside the **worker** container
- Playback endpoints read `.cache/backtests/` from the **ops-api** container
- Without a shared volume, the API never sees the files

**Impact:** `/equity`, `/trades`, `/playback/*` appear empty or 404 even when backtests succeed

**Short-term Fix:** Add a shared Docker volume for `.cache/backtests` between worker + ops-api

**Long-term Fix (Required):** Replace disk cache dependencies with Temporal query-backed responses for playback endpoints

### 3. **INCOMPLETE DOCKER SERVICES** (Severity: HIGH)

**Current Status:** Only `worker` service is running

```bash
$ docker compose ps
NAME                             STATUS
crypto-trading-agents-worker-1   Up
# Missing: temporal, db, ops-api, app
```

**Impact:** No Temporal server, no database, no API

**Fix Required:** Run full stack:
```bash
docker compose up temporal db ops-api worker
```

### 4. **MISSING PROXY ROUTES** (Severity: MEDIUM)

**Location:** `ui/vite.config.ts`

**Problem:** No proxy for `/workflows` endpoint used by `agentAPI.listWorkflows()`

**Impact:** Agent workflow listing fails in dev mode

**Fix Required:** Add to vite.config.ts:
```typescript
'/workflows': {
  target: 'http://localhost:8081',
  changeOrigin: true,
},
'/llm': {
  target: 'http://localhost:8081',
  changeOrigin: true,
},
```

### 5. **WEBSOCKET DIRECT CONNECTION** (Severity: LOW)

**Location:** `ui/src/lib/websocket.ts:36-37`

**Current Behavior:** In dev mode, WebSockets bypass vite proxy and connect directly to `localhost:8081`

**Status:** This is actually correct - Vite proxy doesn't support WebSocket well. The WebSocket connection will work IF ops-api is running.

---

## Services Dependency Map

```
┌──────────────┐
│     db       │ (postgres:5432)
└──────────────┘
       │
       ▼
┌──────────────┐
│   temporal   │ (grpc:7233, ui:8088)
└──────────────┘
       │
       ├───────────────────┐
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│    worker    │    │   ops-api    │ (http:8081, ws:8081)
└──────────────┘    └──────────────┘
       │                   │
       └─────────┬─────────┘
                 │
       ┌─────────▼─────────┐
       │       app         │ (mcp:8080) - optional for backtesting
       └───────────────────┘
```

---

## Data Sources

### OHLCV Data

**Location:** `data/backtesting/`

```
data/backtesting/
├── BTC-USD_1h.csv  (22KB)
└── ETH-USD_1h.csv  (107KB)
```

**Source:** CCXT API (Coinbase) with disk caching
**Loader:** `data_loader/api_loader.py` → `CCXTAPILoader`

**Status:** ✅ Data exists for BTC-USD and ETH-USD (1h timeframe)

### Backtest Results Cache

**Location:** `.cache/backtests/`

**Status:** Empty (no completed backtests persisted)

---

## Fix Plan

### Phase 1: Fix Data Format Mismatch (Priority 1)

**File:** `ops_api/routers/backtests.py`

Update endpoints to handle the workflow's output format:

```python
# get_equity_curve() - line 442
@router.get("/{run_id}/equity", response_model=List[EquityCurvePoint])
async def get_equity_curve(run_id: str):
    cached = get_backtest_cached(run_id)
    if not cached:
        raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

    # Handle both formats
    if "result" in cached:
        # Legacy format (PortfolioBacktestResult object)
        result = cached["result"]
        equity_series = result.equity_curve
        return [
            EquityCurvePoint(timestamp=ts.isoformat(), equity=float(eq))
            for ts, eq in equity_series.items()
        ]
    else:
        # New format (list of dicts from workflow)
        equity_curve = cached.get("equity_curve", [])
        return [
            EquityCurvePoint(
                timestamp=point.get("timestamp") or point.get("time", ""),
                equity=float(point.get("equity", 0))
            )
            for point in equity_curve
        ]
```

Similar updates needed for:
- `get_backtest_trades()` (line 476)
- `get_playback_candles()` (line 528)
- `get_playback_events()` (line 632)
- `get_portfolio_state_snapshot()` (line 698)

### Phase 2: Update Vite Proxy Config

**File:** `ui/vite.config.ts`

```typescript
proxy: {
  // ... existing routes ...
  '/workflows': {
    target: 'http://localhost:8081',
    changeOrigin: true,
  },
  '/llm': {
    target: 'http://localhost:8081',
    changeOrigin: true,
  },
  '/events': {
    target: 'http://localhost:8081',
    changeOrigin: true,
  },
}
```

### Phase 3: Local Development Startup Script

Create `scripts/dev-start.sh`:

```bash
#!/bin/bash
# Start full stack for local development

echo "Starting database and Temporal..."
docker compose up -d db temporal temporal-ui

echo "Waiting for Temporal to be ready..."
until docker compose exec -T temporal tctl namespace list > /dev/null 2>&1; do
  sleep 2
done

echo "Starting worker and ops-api..."
docker compose up -d worker ops-api

echo "Starting UI dev server..."
cd ui && npm run dev
```

---

## Testing Checklist

After fixes, verify:

1. [ ] `POST /backtests` starts workflow (check Temporal UI at :8088)
2. [ ] `GET /backtests/{id}` returns status
3. [ ] `GET /backtests/{id}/results` returns metrics after completion
4. [ ] `GET /backtests/{id}/equity` returns chart data
5. [ ] `GET /backtests/{id}/trades` returns trade log
6. [ ] WebSocket `/ws/live` connects and receives events
7. [ ] WebSocket `/ws/market` connects and receives ticks
8. [ ] UI equity chart renders correctly
9. [ ] UI trade table populates correctly

---

## Quick Start (After Fixes)

```bash
# Start infrastructure
docker compose up -d db temporal temporal-ui

# Wait for Temporal
sleep 10

# Start backend services
docker compose up -d worker ops-api

# Start UI (separate terminal)
cd ui && npm run dev

# Access UI at http://localhost:3000
# Access Temporal UI at http://localhost:8088
# Access Ops API at http://localhost:8081
```

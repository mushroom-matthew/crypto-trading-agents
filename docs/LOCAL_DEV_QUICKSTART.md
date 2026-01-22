# Local Dev Quickstart (UI + Backend)

This guide is for local feature/UI testing with the Vite dev server plus the
backend stack (Ops API, MCP server, Temporal, worker, Postgres) running in
Docker.

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  UI (3000)  │────▶│ Ops API     │────▶│  Temporal   │
│  Vite Dev   │     │   (8081)    │     │   (7233)    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Postgres   │     │   Worker    │
                    │   (5432)    │     │             │
                    └─────────────┘     └─────────────┘
```

**Port Reference:**
- `3000` - Vite UI dev server
- `8081` - Ops API (backtesting, monitoring, dashboards)
- `8080` - MCP Server (agent tools - not used by UI)
- `8088` - Temporal UI
- `7233` - Temporal gRPC
- `5432` - PostgreSQL

## 1) Prereqs

- Docker + Docker Compose
- Node + npm (for the UI in `ui/`)

## 2) Configure env

Copy the example file and set required values:

```bash
cp .env.example .env
```

At minimum, set:

```bash
OPENAI_API_KEY=your_key
DB_DSN=postgresql+psycopg://botuser:botpass@db:5432/botdb
```

Optional (only if you test live Coinbase flows):

```bash
COINBASE_API_KEY=
COINBASE_API_SECRET=
COINBASE_WALLET_SECRET=
```

## 3) Start backend (detached)

```bash
docker compose up -d db temporal temporal-ui ops-api worker
docker compose ps
```

**Note:** The `app` service (MCP server) is optional for backtesting. Only needed if you're testing agent tools.

Wait for Temporal to be healthy before running backtests:

```bash
# Check Temporal is ready (should show "default" namespace)
docker compose exec temporal tctl namespace list
```

Tail only the services you care about:

```bash
docker compose logs -f ops-api worker
```

## 3.5) Paper trading (default)

The stack runs in paper mode by default (`TRADING_MODE=paper`) and uses mock
fills while streaming live market data. To enable real orders, set
`TRADING_MODE=live` and `LIVE_TRADING_ACK=true`.

## 4) Run database migrations (first time only)

```bash
docker compose exec ops-api uv run alembic upgrade head
```

## 5) (Optional) Seed wallet rows

If you want wallet/portfolio panels to show data immediately, edit
`app/db/seed_wallets.py` and then run:

```bash
docker compose exec ops-api uv run python -m app.db.seed_wallets
```

## 6) Start the Vite UI

```bash
cd ui
npm install
npm run dev
```

Open `http://localhost:3000/`.

The Vite dev server proxies API calls to `http://localhost:8081` by default
(see `ui/vite.config.ts`), and WebSocket URLs auto-target port 8081.

If you need to override the backend host/port, create `ui/.env`:

```bash
VITE_API_URL=http://localhost:8081
# or
VITE_WS_URL=ws://localhost:8081
```

## 7) Running Backtests

### Available Historical Data

The system caches OHLCV data in `data/backtesting/`:

```
data/backtesting/
├── BTC-USD_1h.csv
└── ETH-USD_1h.csv
```

If data doesn't exist, it will be fetched from Coinbase via CCXT on first backtest.

### Starting a Backtest via UI

1. Open http://localhost:3000
2. Configure backtest parameters (symbols, date range, strategy)
3. Click "Start Backtest"
4. Monitor progress via the status panel
5. View results when complete (equity curve, trades, metrics)

### Verifying Backtest Execution

Check the Temporal UI at http://localhost:8088:
- Look for workflows with ID `backtest-*`
- View workflow status, history, and queries

Check worker logs for simulation progress:
```bash
docker compose logs -f worker
```

### Backtest Data Flow

```
UI (POST /backtests)
    │
    ▼
Ops API (starts Temporal workflow)
    │
    ▼
BacktestWorkflow
    ├─ load_ohlcv_activity (fetches data)
    ├─ run_simulation_chunk_activity (runs strategy)
    └─ persist_results_activity (saves to .cache/backtests/)
    │
    ▼
UI polls GET /backtests/{id} for status
    │
    ▼
UI fetches GET /backtests/{id}/equity, /trades when complete
```

## 8) Health checks

- Ops API: `http://localhost:8081/health`
- MCP server: `http://localhost:8080/healthz` (if running)
- Temporal UI: `http://localhost:8088/`
- Temporal gRPC: `docker compose exec temporal tctl namespace list`

## 9) Troubleshooting

### Backtest stuck at "queued"

Worker isn't running or can't connect to Temporal:
```bash
docker compose logs worker
docker compose restart worker
```

### "Backtest not found" after completion

Results aren't being persisted. Check worker logs for errors in `persist_results_activity`:
```bash
docker compose logs worker | grep -i "persist"
```

### Equity curve / trades not showing

The API endpoints read from `.cache/backtests/`. Verify the file exists:
```bash
ls -la .cache/backtests/
```

### WebSocket not connecting

WebSockets connect directly to port 8081 (not through Vite proxy). Ensure ops-api is running:
```bash
curl http://localhost:8081/ws/stats
```

### "No OHLCV data available"

Data for the requested symbol/timeframe doesn't exist. The system will try to fetch from Coinbase:
- Ensure you have internet connectivity
- Check if the symbol is valid (e.g., `BTC-USD`, not `BTCUSD`)
- Try a different date range (data may not be available for all periods)

### Import errors in tests

Some tests require `OPENAI_API_KEY` to even import. Set a dummy key:
```bash
OPENAI_API_KEY=test uv run pytest tests/test_specific.py
```

## 10) Stop everything

```bash
docker compose down
```

To also remove volumes (database data):
```bash
docker compose down -v
```

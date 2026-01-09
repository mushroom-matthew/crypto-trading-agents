# Local Dev Quickstart (UI + Backend)

This guide is for local feature/UI testing with the Vite dev server plus the
backend stack (Ops API, MCP server, Temporal, worker, Postgres) running in
Docker.

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
docker compose up -d db temporal temporal-ui app ops-api worker
docker compose ps
```

Tail only the services you care about:

```bash
docker compose logs -f ops-api app worker
```

## 3.5) Paper trading (default)

The stack runs in paper mode by default (`TRADING_MODE=paper`) and uses mock
fills while streaming live market data. To enable real orders, set
`TRADING_MODE=live` and `LIVE_TRADING_ACK=true`.

## 4) Run database migrations (first time only)

```bash
docker compose exec app uv run alembic upgrade head
```

## 5) (Optional) Seed wallet rows

If you want wallet/portfolio panels to show data immediately, edit
`app/db/seed_wallets.py` and then run:

```bash
docker compose exec app uv run python -m app.db.seed_wallets
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

## 7) Health checks

- Ops API: `http://localhost:8081/health`
- MCP server: `http://localhost:8080/healthz`
- Temporal UI: `http://localhost:8088/`

## 8) Stop everything

```bash
docker compose down
```

# Dashboard Behavior and Troubleshooting

The FastAPI dashboard launched via `uv run python -m app.dashboard` supervises *local* subprocesses that mirror the Temporal + MCP stack. When the main stack already runs via Docker Compose, the processes surfaced by the dashboard are still defined in `app/dashboard/server.py`, but starting them simply launches another copy on the host.

Because of that separation:

- Docker Compose services (Temporal, worker, MCP server, etc.) are already running inside containers. The dashboard is unaware of that state, so the “Start” buttons spin up additional local copies rather than reflecting container health. When running everything in Docker, leave those entries alone or replace them with health probes that check the containers. Set `DASHBOARD_ENABLE_SUPERVISOR=1` only when you intentionally want the dashboard to spawn local processes.
- The “Start” buttons now stream stdout/stderr into `logs/dashboard/<process>.log`. Inspect those files whenever a process immediately reverts to `stopped`—it usually means the process exited because ports were occupied by Docker or required env vars were missing.
- Wallet and balance information comes directly from the ledger database configured via `DB_DSN`. With the default migrations there is **no seed data**, so the “Portfolios & Wallets” panel remains empty until you insert rows through Coinbase sync jobs or manual SQL. Confirm data by running `docker compose exec db psql -U botuser -d botdb -c "SELECT * FROM wallets;"`.

## Recommended workflow

1. Use `docker compose up -d db temporal temporal-ui app worker` to run the production stack.
2. If you still want the dashboard view, run it separately but treat the process list as informational unless `DASHBOARD_ENABLE_SUPERVISOR=1` is set. The log files referenced in the UI show why a given Start command failed.
3. Seed ledger data via `uv run python -m app.db.seed_wallets` (edit `SEED_WALLETS` first) or your Coinbase onboarding scripts so that the “Portfolios & Wallets” table has something to display. Document any funding transactions (wallet names, expected balances, tradeable %).

Documenting these expectations up front prevents confusion about “missing” cash or apparently failing start buttons when the real services are already containerized.

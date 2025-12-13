# AI Slop Refactor Plan (Living Document) — Updated for UI-First Ops

This replaces prior drafts; keep it current as changes land.

## Guiding Principles
- Default to safe, paper trading; make live execution an explicit, latched choice.
- Declare what runs (allowlists) instead of importing the world.
- One canonical bootstrap (Compose) and one canonical control plane (UI + Ops API), replacing pane-watching.
- Preserve capability (wallets, legacy) behind explicit profiles and gates; eliminate unused or unwired surfaces.
- Instrument all LLM calls consistently via Langfuse; keep dependencies/workspace clean.
- Contract-first: schemas and APIs are defined before implementations to prevent drift.

## Refactor Tracks (ordered for ROI)

### Authoritative modes and safety latches
- `TRADING_STACK={agent|legacy_live}` (default agent), `TRADING_MODE={paper|live}` (default paper).
- Require `LIVE_TRADING_ACK=true` for live wallet initialization; optional second latch via UI “unlock.”
- Central config module consumed by workers, MCP server, Ops API, and wallet providers.
**Acceptance criteria**
- In paper mode, live wallet provider cannot be instantiated (hard fail).
- In live mode, startup fails unless `LIVE_TRADING_ACK=true`.
- Mode/banner exposed in Ops API `/health` or `/status`.

### Kill auto-registration/import side effects
- Split worker entrypoints: `worker/run.py`, `worker/agent_worker.py`, `worker/legacy_live_worker.py`.
- Explicit allowlists per worker (workflows, activities, tool modules).
- CI: `TRADING_STACK=agent` must not import `legacy/*` (import-time assertion).
**Acceptance criteria**
- Running the agent worker imports only the allowlisted modules.
- “Legacy” workflows/services do not load unless `TRADING_STACK=legacy_live`.

### Quarantine legacy code (reduce surface early)
- Move unused/legacy workflows/services into `legacy/`.
- Add `legacy/README.md` and `legacy/manifest.yml` (purpose + sunset criteria per module).
- Ensure agent mode never imports legacy by default.
**Acceptance criteria**
- `legacy/` is reachable only via `TRADING_STACK=legacy_live`.
- Manifest exists with owner rationale and a deletion trigger/date per item.

### Contract-first: events + read models + Ops API spec
- Define event schema (append-only): `event_id, ts, source, type, payload, dedupe_key, run_id, correlation_id`.
- Core event types: ticks, intents, plan_generated, plan_judged, trigger_fired, trade_blocked(reason), order_submitted, fill, position_update, llm_call.
- Read model schema: runs, plan summaries, triggers, block reason aggregates, orders/fills, positions, risk usage.
- Define Ops API endpoints + response models (FastAPI pydantic models) before implementation.
**Acceptance criteria**
- Schemas exist in-code (and optionally `docs/contracts/`).
- Breaking schema changes require version bump.

### Single canonical bootstrap (Compose + UI control plane)
- `docker-compose.yml` becomes the only supported runtime path.
- Compose starts: Temporal deps, agent worker(s), durable store, `ops_api`, `ui`.
- Remove tmux as control path once UI parity is achieved; archive tmux scripts/docs.
**Acceptance criteria**
- A new contributor can run `docker compose up` and reach UI + `/health`.
- README references only Compose for running the stack.

### Ops API backend (thin, read-first, Temporal-aware)
- `ops_api/` (FastAPI):
  - Read endpoints: workflow status/list, last plan/judge artifacts, risk usage, block reasons, orders/fills/positions, LLM telemetry.
  - Command endpoints (whitelist): pause/resume trading, rotate/regenerate plan, start/stop backtests.
- Commands implemented via Temporal signals/starts; no trading logic in `ops_api`.
**Acceptance criteria**
- UI can answer: “Is it running?” and “What plan is active?” without container log access.
- Commands are gated by mode (paper/live) and (if live) UI unlock.

### Durable signals/events + materialization
- Implement append-only event log (SQLite WAL acceptable initially).
- Replace in-memory signal log and HTTP fan-out; producers write events, consumers read events.
- Add a single “materializer” for derived read models (library or service). Avoid duplicate state computation paths.
- Avoid routing Temporal activity traffic through HTTP callbacks unless explicitly justified.
**Acceptance criteria**
- Restarting services does not lose events.
- UI can show block reasons/triggers/trades across restarts.
- Exactly one codepath builds read models from events.

### LLM client instrumentation
- `agents/llm/client_factory.py` always returns a Langfuse-instrumented client.
- Refactor all LLM callers to use the factory.
- Standardize LLM call records (including prompt hash and cost estimate).
**Acceptance criteria**
- Every strategist/judge/backtest LLM call emits a telemetry event and Langfuse span.
- Ops API can display daily token/cost totals per run.

### Wallet providers + gating
- WalletProvider interface with PaperWalletProvider and LiveWalletProvider.
- Live provider gated by `TRADING_MODE=live` + `LIVE_TRADING_ACK=true`.
- Wallet actions exposed only via controlled layers (Temporal activities / service layer), not ad hoc tools.
**Acceptance criteria**
- Paper runs never touch live wallet codepaths (tested).
- Live wallet functions are unavailable until explicitly enabled and unlocked.

### Prune or wire dead/unwired modules + workspace hygiene
- For each suspect module: decide Keep+Wire, Archive, or Delete.
- Move historical notes to `docs/archive/`; keep repo root clean.
- Remove stdlib deps (`asyncio`) and unused libs; add CI checks:
  - lint/dead imports
  - root file allowlist
  - unused entrypoints detection
**Acceptance criteria**
- Repo root contains only approved files.
- CI fails on new root artifacts or unused scripts.

## Suggested PR Sequencing (updated)
1. Worker allowlists + profile split
2. Quarantine legacy behind profile + CI import fence
3. Contract-first schemas (events/read models/Ops API spec)
4. Compose canonical + initial UI scaffold (even if minimal)
5. Ops API read endpoints + UI parity for status/plan/block reasons
6. Durable event log + materializer; remove in-memory signal log
7. LLM client factory + Langfuse spans
8. WalletProvider + paper/live gating + UI unlock
9. Slop purge + dependency/workspace hygiene

## Done criteria for removing tmux (UI-driven)
- Can answer: Is it running? (workers/Temporal connectivity)
- What plan is active and why? (plan/judge visibility)
- Why didn’t it trade? (block reasons, caps, risk budget)
- What did it do? (orders/fills/positions)
- Can I pause it safely? (kill-switch)

## Two specific calls to prevent future slop
- Add correlation IDs everywhere. Without them, event forensics are brittle. Correlate plan → triggers → trade decision → order → fills.
- Make “block reasons” a first-class event type. Treat it as schema, not a log line.

If helpful, we can also add a short “initial UI parity spec” (cards/tables/controls) so UI work stays bounded.

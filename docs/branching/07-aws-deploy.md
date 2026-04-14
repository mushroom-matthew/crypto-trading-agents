# Branch: aws-deploy

## Purpose

Move from a purely local stack to a crash-resilient cloud deployment. WSL2 crashes have repeatedly killed active paper trading sessions. Temporal Cloud solves this immediately by making workflow state independent of the local machine.

**Implementation order matches urgency:**
1. **Phase 0 — Temporal Cloud** (immediate, ~2 hours) — workflows survive crashes
2. **Phase 1 — Remote Worker** (Fly.io or EC2, ~half day) — process survives machine restarts
3. **Phase 2 — ECS Fargate** (full cloud ops-api + mcp-server, ~1-2 days)
4. **Phase 3+4 — RDS + Live Trading Controls** (pre-live-trading)

## Source Plans

- `docs/AWS_DEPLOYMENT_SCOPE.md` — full architecture, cost estimates, checklist

## Scope (Phase 0 — implement first)

- Wire Temporal Cloud TLS into `worker/agent_worker.py`
- Wire TLS into `get_temporal_client()` (shared by ops-api and mcp-server)
- Add `TEMPORAL_TLS_CERT`, `TEMPORAL_TLS_KEY` env var support
- Make `temporal` + `temporal-ui` docker-compose services optional (local dev only)
- Update `.env.example` with Temporal Cloud vars

## Scope (Phase 1)

- Add `fly.toml` for worker deployment
- Dockerfile already exists — worker builds from existing image
- Document secret provisioning for remote worker

## Scope (Phase 2+)

- `infra/ecs/task-definitions/worker.json`, `ops-api.json`, `mcp-server.json`
- `infra/terraform/*.tf` (vpc, ecs, alb, secrets, cloudwatch, iam)
- CI/CD via `.github/workflows/deploy-paper.yml` and `deploy-live.yml`

## Out of Scope

- Trading logic changes
- Changes to workflow business logic
- Multi-wallet changes

## Key Files

**Phase 0:**
- `worker/agent_worker.py` — TLS client connect
- `worker/run.py` — may need TLS config passed through
- `mcp_server/app.py` — `get_temporal_client()` used here
- `ops_api/routers/paper_trading.py` — `get_temporal_client()` used here
- `.env.example` — add Temporal Cloud vars
- `docker-compose.yml` — make temporal services optional

**Phase 1:**
- `fly.toml` (new)
- `Dockerfile` (already exists, no changes expected)

**Phase 2+:**
- `infra/ecs/task-definitions/*.json`
- `infra/terraform/*.tf`
- `.github/workflows/deploy*.yml`

## Architecture Context (current — as of 2026-04-13)

Two separate HTTP servers:
- **MCP Server** (`mcp_server/app.py`, port 8080) — agent/programmatic interface
- **Ops API** (`ops_api/app.py`, port 8081) — human operator dashboard

Three ECS tasks needed (no self-hosted Temporal after Phase 0):
- `worker` — runs all Temporal workflows and activities
- `ops-api` — FastAPI for UI
- `mcp-server` — FastMCP for agent tools

Key workflow durability facts:
- `PaperTradingWorkflow` uses ContinueAsNew every ~40 cycles — history is bounded
- `SessionState` is fully serializable — CaN snapshots survive worker restarts cleanly
- All in-session state (WorldState, EpisodeMemory, RegimeTrajectory, trailing stop states, AI planner intent) lives inside Temporal — no external DB needed for paper trading

## Dependencies

- Temporal Cloud account (free tier sufficient for paper trading)
- Fly.io account (Phase 1)
- AWS account (Phase 2+)

## Acceptance Criteria

**Phase 0:**
- Worker connects to Temporal Cloud with TLS
- Kill the local worker mid-session → restart it → session resumes at same cycle
- Kill docker compose entirely → restart → session resumes
- Local dev still works with `docker-compose.yml` temporal services (via env var toggle)

**Phase 1:**
- Worker deployed to remote VM
- Local machine can be shut down → sessions continue running (triggers keep evaluating)
- Worker reconnects automatically after own restart

**Phase 2:**
- All services running in ECS Fargate
- Ops API accessible via ALB HTTPS endpoint
- React UI pointed at cloud ops-api

## Test Plan

**Phase 0:**
```bash
# 1. Start session, confirm it's running
curl http://localhost:8081/paper-trading/sessions

# 2. Kill everything
docker compose down

# 3. Restart only the worker (no local temporal)
uv run python -m worker.main

# 4. Confirm session resumed at same cycle
curl http://localhost:8081/paper-trading/sessions
```

**Phase 1:**
```bash
# Deploy worker to Fly.io
fly deploy

# Confirm worker is polling Temporal Cloud
fly logs

# Shut down local machine (or just kill all processes)
# Wait 1 candle interval (5 minutes for 5m sessions)
# Restart local ops-api
# Confirm session cycle count is still incrementing
curl http://localhost:8081/paper-trading/sessions/<session_id>
```

**Phase 2:**
```bash
# Terraform
cd infra/terraform && terraform fmt -recursive
cd infra/terraform && terraform init
cd infra/terraform && terraform validate
cd infra/terraform && terraform plan

# Paste plan output in Test Evidence before applying
```

## Worktree Setup

```bash
git fetch
git worktree add -b aws-deploy ../wt-aws-deploy main
cd ../wt-aws-deploy

# When finished
git worktree remove ../wt-aws-deploy
```

## Git Workflow

```bash
git checkout main
git pull
git checkout -b aws-deploy

# Phase 0 commit
git add worker/agent_worker.py mcp_server/app.py ops_api/routers/paper_trading.py \
    docker-compose.yml .env.example
git commit -m "feat: Temporal Cloud TLS wiring (Phase 0 crash resilience)"

# Phase 1 commit
git add fly.toml
git commit -m "feat: Fly.io worker deploy config (Phase 1)"

# Phase 2 commit (after terraform validate passes)
git add infra/ .github/workflows/
git commit -m "feat: ECS Fargate + Terraform infrastructure (Phase 2)"
```

## Change Log

- 2026-04-13: Rewritten from scratch. Original scope was ECS-first; reprioritized to Temporal Cloud (Phase 0) as immediate fix for WSL2 crash problem. Added Phase 1 (Fly.io worker). Updated architecture to reflect two-server design (MCP 8080 + Ops API 8081), ContinueAsNew durability, AI planner, trailing stops, min_rr_ratio gate, SessionState serialization.
- 2026-04-14: Phase 0 implemented.
  - `agents/temporal_utils.py` — added `_build_tls_config()` helper; wired TLS into both `get_temporal_client()` and `connect_temporal()`. TLS activates only when `TEMPORAL_TLS_CERT` + `TEMPORAL_TLS_KEY` are set; plain gRPC otherwise (backward-compatible).
  - `worker/agent_worker.py` — replaced bare `Client.connect()` with `get_temporal_client()` from temporal_utils; TLS now flows through the shared helper.
  - `.env.example` — added commented-out Temporal Cloud block with `TEMPORAL_ADDRESS`, `TEMPORAL_NAMESPACE`, `TEMPORAL_TLS_CERT`, `TEMPORAL_TLS_KEY`, `TEMPORAL_TLS_CA`.
  - `docker-compose.yml` — `temporal` and `temporal-ui` services moved to `profiles: ["local"]`; hard `depends_on: temporal` removed from `app`, `ops-api`, `worker` so they start cleanly against Temporal Cloud.

## Test Evidence

**Full suite run (2026-04-14):**
```
1 failed, 2336 passed, 1 skipped in 633.94s
```
Failure: `tests/integration/test_high_budget_activity.py::test_high_budget_activity_consumes_budget` — **pre-existing flaky test**, passes in isolation (`1 passed in 12.36s`). Cause: shared state contamination from earlier tests in full suite. Unrelated to TLS wiring changes.

Changed files (`agents/temporal_utils.py`, `worker/agent_worker.py`, `.env.example`, `docker-compose.yml`) contain no business logic changes — only client connection setup and compose service configuration.

## Human Verification Evidence

**Phase 0 (TLS wiring) — implementation complete, cloud smoke test pending:**
- [ ] Create Temporal Cloud account at cloud.temporal.io and provision namespace
- [ ] Download client.pem + client.key from the Temporal Cloud console
- [ ] Set `TEMPORAL_ADDRESS`, `TEMPORAL_NAMESPACE`, `TEMPORAL_TLS_CERT`, `TEMPORAL_TLS_KEY` in `.env`
- [ ] Start stack: `docker compose up` (no `--profile local` needed)
- [ ] Confirm worker log shows: `Connecting to Temporal Cloud at <namespace>.tmprl.cloud:7233 (ns=...) with mTLS`
- [ ] Start a paper trading session, kill all processes, restart worker only → confirm session cycle count continues
- [ ] Review Terraform plan for cost/security implications before `terraform apply` (Phase 2)

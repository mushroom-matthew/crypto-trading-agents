# AWS Deployment Scope — Paper & Live Trading

**Last updated:** 2026-04-13
**Status:** Phase 0 (Temporal Cloud) is immediate priority — solves WSL2 crash problem. Full ECS deployment follows.

---

## Problem Statement

Local WSL2 crashes repeatedly kill the entire stack mid-session. Temporal's durable execution guarantees only hold as long as the Temporal server itself is running. Moving Temporal to a managed cloud service (Temporal Cloud) makes workflows crash-proof regardless of what happens on the local machine.

---

## Deployment Phases

| Phase | Scope | Urgency |
|-------|-------|---------|
| **0** | Temporal Cloud — workflows survive local crashes | **Now** |
| **1** | Worker on EC2/Fly.io — process survives machine restarts | Soon |
| **2** | Ops API + MCP Server on ECS Fargate — remove local dependency | Medium |
| **3** | RDS PostgreSQL — production ledger | Pre-live-trading |
| **4** | CI/CD, CloudWatch, live trading safety controls | Pre-live-trading |

---

## Phase 0: Temporal Cloud (Immediate)

**Goal:** Workflows (`PaperTradingWorkflow`, `ExecutionLedgerWorkflow`) run in Temporal Cloud. Worker can be local or remote — either way, a crash just means the worker reconnects and resumes from the last checkpoint.

**Steps:**
1. Create Temporal Cloud account at cloud.temporal.io
2. Create a namespace (e.g., `crypto-trading.acctid`)
3. Download TLS certs (client cert + key + CA cert)
4. Update `TEMPORAL_ADDRESS` in `.env` to `<namespace>.tmprl.cloud:7233`
5. Add cert paths to worker config

**Env changes (`.env`):**
```bash
TEMPORAL_ADDRESS=<namespace>.tmprl.cloud:7233
TEMPORAL_NAMESPACE=<your-namespace>
TEMPORAL_TLS_CERT=/path/to/client.pem
TEMPORAL_TLS_KEY=/path/to/client.key
# TEMPORAL_TLS_CA=/path/to/ca.pem  # optional, for mTLS
```

**Worker connection change (`worker/agent_worker.py`):**
```python
from temporalio.client import Client, TLSConfig

tls = TLSConfig(
    client_cert=open(os.environ["TEMPORAL_TLS_CERT"], "rb").read(),
    client_private_key=open(os.environ["TEMPORAL_TLS_KEY"], "rb").read(),
)
client = await Client.connect(
    os.environ["TEMPORAL_ADDRESS"],
    namespace=os.environ["TEMPORAL_NAMESPACE"],
    tls=tls,
)
```

Same change needed in `mcp_server/app.py` and `ops_api/routers/paper_trading.py` wherever `get_temporal_client()` is called.

**Docker compose change:** Remove the `temporal` and `temporal-ui` services from `docker-compose.yml` (or keep them for local dev with a `TEMPORAL_ADDRESS=temporal:7233` override).

**Cost:** Temporal Cloud free tier = 10K actions/month. Paper trading at 5m bars ≈ ~288 workflow task executions/day × 30 = ~8.6K/month. Fits free tier for a single session.

**What survives a crash after Phase 0:**
- All active `PaperTradingWorkflow` sessions ✓
- All `ExecutionLedgerWorkflow` states (cash, positions) ✓
- `ExecutionAgentWorkflow`, `JudgeAgentWorkflow`, `BrokerAgentWorkflow` ✓
- Session intent, world state, episode memory, trailing stop states ✓

**What still dies in a crash (fixed in Phase 1+):**
- Worker process (stops evaluating triggers — resumes on restart)
- Ops API (UI goes dark — resumes on restart)
- Local React UI dev server

---

## Phase 1: Worker on Remote VM

**Goal:** The worker process never goes down with the local machine.

**Options (cheapest first):**

| Option | Cost | Notes |
|--------|------|-------|
| Fly.io Machine (shared-cpu-1x, 256MB) | ~$2/mo | Simple deploy, good for single worker |
| EC2 t3.micro (spot) | ~$3/mo | Familiar, easy IAM integration |
| EC2 t3.small (on-demand) | ~$15/mo | More headroom for 5+ simultaneous sessions |
| Railway | ~$5/mo | Easy GitHub-based deploy |

**Fly.io is the fastest path** — `fly deploy` from the repo root with a minimal `fly.toml`.

**Worker sizing:**
- Current: worker runs all workflows + activities for paper trading
- 1m timeframe, 3 symbols: ~2 activity executions/minute → easily fits 256MB
- 5m timeframe, 5+ symbols: ~1 activity execution/5m → trivial load

**What the worker needs (env vars):**
```bash
TEMPORAL_ADDRESS=<cloud-endpoint>
TEMPORAL_NAMESPACE=<namespace>
TEMPORAL_TLS_CERT=...
TEMPORAL_TLS_KEY=...
OPENAI_API_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
COINBASEEXCHANGE_API_KEY=...  # for market data fetch
COINBASEEXCHANGE_SECRET=...
POSTGRES_HOST=...  # local or RDS
```

---

## Phase 2: Ops API + MCP Server on ECS Fargate

**Architecture (updated — reflects current two-server design):**

```
┌─────────────────────────────────────────────────────────────────────┐
│                           AWS VPC                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Public Subnets                                              │   │
│  │  ├── ALB → Ops API  (port 8081, operator dashboard)         │   │
│  │  └── ALB → MCP Server (port 8080, agent/programmatic API)   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Private Subnets                                             │   │
│  │  ├── ECS Fargate: worker (Temporal task queue consumer)      │   │
│  │  ├── ECS Fargate: ops-api (FastAPI, port 8081)              │   │
│  │  └── ECS Fargate: app/mcp-server (FastMCP, port 8080)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Database Subnets                                            │   │
│  │  └── RDS PostgreSQL (Phase 3+)                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
          │                                │
          ▼                                ▼
┌─────────────────────┐      ┌─────────────────────────┐
│  Temporal Cloud     │      │  External APIs           │
│  (workflows + state)│      │  Coinbase, OpenAI,       │
└─────────────────────┘      │  Langfuse               │
                             └─────────────────────────┘
```

**Note:** Temporal is NOT self-hosted after Phase 0. The `temporal.json` task definition from the original scope is dropped.

**ECS Task Definitions (`infra/ecs/task-definitions/`):**

| Task | CPU | Memory | Port | Purpose |
|------|-----|--------|------|---------|
| `worker.json` | 512 | 1024 | — | Temporal workflow + activity runner |
| `ops-api.json` | 256 | 512 | 8081 | Human operator dashboard API |
| `mcp-server.json` | 256 | 512 | 8080 | Agent/programmatic MCP tools |

**Key env vars that must be in Secrets Manager:**
```
trading/openai      → OPENAI_API_KEY
trading/coinbase    → COINBASEEXCHANGE_API_KEY, COINBASEEXCHANGE_SECRET
trading/langfuse    → LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY
trading/temporal    → TEMPORAL_ADDRESS, TEMPORAL_NAMESPACE, TEMPORAL_TLS_CERT, TEMPORAL_TLS_KEY
trading/database    → DB_DSN (Phase 3+)
trading/config      → LIVE_TRADING_ACK, ENABLE_REAL_LEDGER (Phase 4)
```

**Terraform files (`infra/terraform/`):**

| File | Purpose |
|------|---------|
| `main.tf` | Provider config, module composition |
| `vpc.tf` | VPC, subnets, NAT gateway, route tables |
| `ecs.tf` | Cluster, services, task definitions |
| `alb.tf` | ALBs for ops-api and mcp-server |
| `secrets.tf` | Secrets Manager secrets |
| `cloudwatch.tf` | Log groups, alarms |
| `iam.tf` | Task roles, execution roles |
| `variables.tf` | Input variables |
| `outputs.tf` | ALB DNS, cluster ARN, etc. |

---

## Phase 3: RDS PostgreSQL

Only needed for live trading (production ledger). Paper trading uses Temporal in-memory state only.

**Paper trading ledger = `ExecutionLedgerWorkflow` in-memory state, serialized through Temporal Cloud.** No DB needed.

**Live trading ledger = `app/ledger/` PostgreSQL tables** (`wallets`, `ledger_entries`, `orders`, `reservations`, `cost_estimates`).

| Setting | Paper | Live |
|---------|-------|------|
| Instance | Not needed | `db.t3.medium` |
| Multi-AZ | — | Yes |
| Backup retention | — | 30 days |
| Deletion protection | — | Yes |

---

## Phase 4: Live Trading Safety Controls + CI/CD

### Safety Controls

| Control | Paper | Live |
|---------|-------|------|
| `LIVE_TRADING_ACK=true` | Not required | Required |
| `ENABLE_REAL_LEDGER=1` | Not required | Required |
| Tradeable fraction | Not applicable | Required (e.g., 20%) |
| Cost gating (`app/costing/gate.py`) | Not applicable | Required |
| Reconciliation (every 15 min) | Not applicable | Required |
| Min R:R gate (`min_rr_ratio`) | 1.75 default | 1.75+ (tighten for live) |
| Multi-AZ RDS | — | Required |
| PagerDuty alerts | Optional | Required |

### New since original scope

The following controls exist in code and need surfacing in the cloud config:

- **`min_rr_ratio`** (default 1.75) — entry gate, must be in session config or env for live
- **AI planner** (`use_ai_planner`) — LLM selects symbols from candidate list; adds OpenAI calls at session start
- **Trailing stops** (`default_trailing_config`) — ATR/pct/step modes; session-level config
- **ContinueAsNew** — workflows self-trim history; no manual intervention needed
- **SessionState** fully serializable — CaN snapshot survives worker restarts cleanly
- **WorldState / RegimeTrajectory / EpisodeMemory** — in-workflow state, no external DB needed for paper
- **Judge agent** — evaluates performance every N trades; prompt versioning in `JudgeAgentWorkflow`

### CI/CD

```
main push → build + test → ECR push → deploy worker + ops-api (paper)
                                     ↓
                              manual approval gate
                                     ↓
                              deploy live (separate ECS cluster)
```

---

## Cost Estimates (Updated)

### Minimal (Temporal Cloud + Fly.io worker — Phase 0+1)

| Resource | Cost/month |
|----------|------------|
| Temporal Cloud (free tier) | $0 |
| Fly.io worker (shared-cpu-1x) | ~$2 |
| **Total** | **~$2/month** |

Covers: crash-resilient sessions, persistent workflows, automatic reconnect on local crash.

### Paper Trading Full Cloud (Phase 2)

| Resource | Specification | Cost/month |
|----------|---------------|------------|
| Temporal Cloud (developer tier) | 100K actions | ~$50 |
| ECS Fargate — worker | 0.5 vCPU, 1GB | ~$15 |
| ECS Fargate — ops-api | 0.25 vCPU, 512MB | ~$8 |
| ECS Fargate — mcp-server | 0.25 vCPU, 512MB | ~$8 |
| ALB (2) | ops-api + mcp | ~$20 |
| NAT Gateway | 1 | ~$35 |
| CloudWatch Logs | 5 GB/month | ~$5 |
| Secrets Manager | 6 secrets | ~$3 |
| **Total** | | **~$144/month** |

### Live Trading (Phase 3+4, additional)

| Resource | Specification | Additional cost/month |
|----------|---------------|----------------------|
| RDS PostgreSQL | db.t3.medium, Multi-AZ | +$70 |
| NAT Gateway (2nd for HA) | | +$35 |
| CloudWatch Alarms (15) | | +$5 |
| PagerDuty | | varies |
| **Additional** | | **~+$110/month** |

### LLM Cost (OpenAI)

| Usage | Est. tokens/week | Cost/week |
|-------|-----------------|-----------|
| 1 session, 5m timeframe, AI planner | ~400K | ~$0.60 |
| 3 sessions running simultaneously | ~1.2M | ~$1.80 |

---

## Deployment Checklist

### Phase 0: Temporal Cloud

- [ ] Create Temporal Cloud account
- [ ] Create namespace
- [ ] Download TLS certs
- [ ] Update `TEMPORAL_ADDRESS`, `TEMPORAL_NAMESPACE` in `.env`
- [ ] Wire TLS into `worker/agent_worker.py` and `get_temporal_client()`
- [ ] Test: start local worker → kill it → restart → confirm session resumed
- [ ] Remove `temporal` + `temporal-ui` services from `docker-compose.yml` (or make optional)

### Phase 1: Remote Worker

- [ ] Choose host (Fly.io recommended)
- [ ] Add `fly.toml` or EC2 deploy script
- [ ] Store secrets in host secret store or AWS Secrets Manager
- [ ] Deploy worker
- [ ] Test: crash local machine → verify triggers resume within one candle

### Phase 2: ECS Fargate

- [ ] Write Terraform in `infra/terraform/`
- [ ] Write ECS task definitions in `infra/ecs/task-definitions/`
- [ ] `terraform apply` for staging
- [ ] Deploy ops-api and mcp-server to ECS
- [ ] Configure ALB with HTTPS (ACM cert)
- [ ] Verify UI works against cloud ops-api endpoint

### Phase 3+4: Live Trading

- [ ] Legal/compliance review
- [ ] Terraform for Multi-AZ RDS
- [ ] Database migrations (`alembic upgrade head`)
- [ ] Seed wallets from Coinbase
- [ ] Set tradeable fractions
- [ ] Configure cost gate
- [ ] Configure scheduled reconciliation
- [ ] Test with $100 minimum capital
- [ ] Monitor 7 days before scaling capital

---

## References

- [Temporal Cloud](https://temporal.io/cloud)
- [Temporal Python TLS docs](https://python.temporal.io/temporalio.client.TLSConfig.html)
- [Fly.io Python deploy](https://fly.io/docs/languages-and-frameworks/python/)
- [AWS ECS Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs)

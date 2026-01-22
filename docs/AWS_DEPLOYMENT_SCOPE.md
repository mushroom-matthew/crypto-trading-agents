# AWS Deployment Scope - Paper & Live Trading

This document outlines the infrastructure, observability, and wiring required to deploy the trading system to AWS for both **paper trading** (simulated) and **live trading** (real capital).

**Key Insight:** The AWS infrastructure is largely the same for both modes. The difference is in:
1. Environment variables and secrets
2. Database schema activation (production ledger)
3. Safety controls and operational procedures

---

## Infrastructure Overview

### Architecture (Both Modes)

```
┌─────────────────────────────────────────────────────────────────────┐
│                           AWS VPC                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Public Subnets                                                  │ │
│  │  ├── Application Load Balancer (HTTPS)                           │ │
│  │  └── NAT Gateway                                                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Private Subnets                                                 │ │
│  │  ├── ECS Fargate: Worker (Temporal workflows)                   │ │
│  │  ├── ECS Fargate: Ops API (FastAPI)                             │ │
│  │  ├── ECS Fargate: MCP Server (Agent tools)                      │ │
│  │  └── Temporal Server (or Temporal Cloud)                        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Database Subnets                                                │ │
│  │  └── RDS PostgreSQL (Multi-AZ for live trading)                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   Coinbase API          │     │   OpenAI API            │
│   (Market Data + Orders)│     │   (LLM Strategy)        │
└─────────────────────────┘     └─────────────────────────┘
```

---

## Phase 5: AWS Infrastructure

### 5.1 ECS Task Definitions

**Directory:** `infra/ecs/task-definitions/`

| Task Definition | Purpose | CPU | Memory | Port |
|-----------------|---------|-----|--------|------|
| `worker.json` | Temporal worker (PaperTradingWorkflow, ExecutionLedgerWorkflow) | 512 | 1024 | - |
| `ops-api.json` | Ops API (FastAPI endpoints) | 256 | 512 | 8081 |
| `mcp-server.json` | MCP Server (agent tools) | 256 | 512 | 8080 |
| `temporal.json` | Temporal server (if self-hosting) | 512 | 1024 | 7233 |

**Environment Variables by Mode:**

| Variable | Paper Trading | Live Trading |
|----------|---------------|--------------|
| `TRADING_MODE` | `paper` | `live` |
| `LIVE_TRADING_ACK` | `false` | `true` |
| `ENABLE_REAL_LEDGER` | `0` | `1` |
| `COINBASE_API_KEY` | Optional (market data only) | Required |
| `COINBASE_API_SECRET` | Optional | Required |
| `DATABASE_URL` | Required | Required |
| `OPENAI_API_KEY` | Required | Required |

**Tasks:**
- [ ] Create `worker.json` task definition
- [ ] Create `ops-api.json` task definition
- [ ] Create `mcp-server.json` task definition
- [ ] Create `temporal.json` task definition (if self-hosting)
- [ ] Configure health checks for each container
- [ ] Set resource limits appropriate for workload
- [ ] Create separate task definition variants for paper vs live (or use env var injection)

---

### 5.2 Terraform Infrastructure

**Directory:** `infra/terraform/`

**Files to Create:**

| File | Purpose |
|------|---------|
| `main.tf` | Provider configuration, module composition |
| `vpc.tf` | VPC, subnets, NAT gateway, route tables |
| `ecs.tf` | ECS cluster, services, task definitions |
| `rds.tf` | PostgreSQL RDS instance |
| `alb.tf` | Application Load Balancer, target groups |
| `secrets.tf` | Secrets Manager secrets |
| `cloudwatch.tf` | Log groups, dashboards, alarms |
| `iam.tf` | Task roles, execution roles |
| `variables.tf` | Input variables |
| `outputs.tf` | Output values |

**RDS Configuration by Mode:**

| Setting | Paper Trading | Live Trading |
|---------|---------------|--------------|
| Instance Class | `db.t3.micro` | `db.t3.medium` or higher |
| Multi-AZ | No | **Yes** (required) |
| Backup Retention | 7 days | 30 days |
| Encryption | Yes | Yes |
| Deletion Protection | No | **Yes** |

**Tasks:**
- [ ] Create VPC with public/private/database subnets
- [ ] Configure NAT Gateway for private subnet internet access
- [ ] Create ECS Fargate cluster
- [ ] Create ECS services for worker, ops-api, mcp-server
- [ ] Create RDS PostgreSQL with appropriate settings
- [ ] Create ALB with HTTPS listener (ACM certificate)
- [ ] Configure target groups and health checks
- [ ] Create Security Groups with least-privilege rules
- [ ] Set up CloudWatch Log Groups
- [ ] Create IAM roles with minimal permissions

---

### 5.3 Secrets Manager Configuration

**Secrets to Store:**

| Secret Name | Keys | Paper | Live |
|-------------|------|-------|------|
| `trading/openai` | `OPENAI_API_KEY` | Required | Required |
| `trading/coinbase` | `COINBASE_API_KEY`, `COINBASE_API_SECRET`, `COINBASE_WALLET_SECRET` | Optional | **Required** |
| `trading/langfuse` | `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` | Optional | Recommended |
| `trading/database` | `DATABASE_URL` | Required | Required |
| `trading/config` | `LIVE_TRADING_ACK`, `TRADING_MODE`, `ENABLE_REAL_LEDGER` | - | **Required** |

**Implementation:**

```python
# app/core/config.py
import os
import json
import boto3
from functools import lru_cache

@lru_cache
def get_secret(secret_name: str) -> dict:
    """Fetch secret from AWS Secrets Manager."""
    if os.environ.get("AWS_SECRETS_ENABLED", "false").lower() != "true":
        return {}

    client = boto3.client("secretsmanager", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

def get_trading_config() -> dict:
    """Get trading configuration with safety checks."""
    config = {
        "TRADING_MODE": os.environ.get("TRADING_MODE", "paper"),
        "LIVE_TRADING_ACK": os.environ.get("LIVE_TRADING_ACK", "false"),
        "ENABLE_REAL_LEDGER": os.environ.get("ENABLE_REAL_LEDGER", "0"),
    }

    # Override with Secrets Manager if enabled
    secrets = get_secret("trading/config")
    config.update({k: v for k, v in secrets.items() if v})

    # CRITICAL SAFETY CHECK
    if config["TRADING_MODE"] == "live" and config["LIVE_TRADING_ACK"] != "true":
        raise RuntimeError(
            "SAFETY VIOLATION: Live trading mode requires LIVE_TRADING_ACK=true"
        )

    return config
```

**Tasks:**
- [ ] Add boto3 to dependencies (`pyproject.toml`)
- [ ] Implement `get_secret()` function with caching
- [ ] Implement safety checks for live trading
- [ ] Create secrets in AWS Secrets Manager
- [ ] Configure IAM roles for Secrets Manager access
- [ ] Test fallback to env vars for local development

---

### 5.4 CloudWatch Alerting

**Alarms (Both Modes):**

| Alarm | Metric | Threshold | Action |
|-------|--------|-----------|--------|
| WorkflowFailures | `TemporalWorkflowsFailed` | > 5 in 5 min | SNS → Email/Slack |
| HighLLMCost | Custom metric | > $10/day | SNS → Email |
| DatabaseConnections | `DatabaseConnections` | > 80% | SNS → Email |
| ECSTaskUnhealthy | `UnhealthyHostCount` | > 0 for 5 min | SNS → Email |
| HighMemoryUsage | `MemoryUtilization` | > 85% | SNS → Email |
| HighCPUUsage | `CPUUtilization` | > 80% | SNS → Email |

**Additional Alarms (Live Trading Only):**

| Alarm | Metric | Threshold | Action |
|-------|--------|-----------|--------|
| **OrderFailures** | `CoinbaseOrdersFailed` | > 0 | SNS → PagerDuty/SMS |
| **LedgerDrift** | `ReconciliationDrift` | > 0.01% | SNS → Email + PagerDuty |
| **HighDrawdown** | `PortfolioDrawdownPct` | > 10% | SNS → PagerDuty |
| **DailyLossLimit** | `DailyRealizedLoss` | > $500 | SNS → PagerDuty + Auto-halt |
| **CostGateBlocks** | `CostGateBlockedOrders` | > 10 in 1h | SNS → Email |
| **APIRateLimit** | `CoinbaseRateLimitHits` | > 0 | SNS → Email |

**Dashboard Widgets:**
- Active trading sessions (paper and live)
- Portfolio equity over time
- Trade count per hour
- Order success/failure rates
- LLM API calls and costs
- Workflow execution latency
- Error rates
- **Live Only:** Real P&L, reconciliation status, cost gate decisions

**Tasks:**
- [ ] Create SNS topics for alerts (standard, urgent)
- [ ] Create CloudWatch alarms for each metric
- [ ] Create CloudWatch dashboard
- [ ] Configure Slack/email notifications
- [ ] Configure PagerDuty for live trading alerts
- [ ] Add custom metrics for trading-specific events

---

### 5.5 CI/CD Pipeline

**File:** `.github/workflows/deploy.yml`

**Pipeline Stages:**
1. **Build** - Run tests, build Docker image
2. **Push** - Push to ECR
3. **Deploy Dev** - Update ECS service in dev (paper mode)
4. **Integration Tests** - Run against dev
5. **Deploy Staging** - Update ECS service in staging (paper mode with real market data)
6. **Deploy Prod** - Update ECS service in prod (manual approval, **separate workflow for live**)

**Live Trading Deployment Requirements:**
- Separate approval workflow
- Requires 2+ approvers
- Automatic rollback on failure
- Blue/green deployment
- Database migration verification

**Tasks:**
- [ ] Create ECR repositories
- [ ] Create GitHub Actions workflow for paper trading
- [ ] Create separate workflow for live trading with approval gates
- [ ] Configure OIDC for AWS authentication
- [ ] Add deployment scripts
- [ ] Add rollback procedures

---

## Phase 6: Live Trading Wiring

### 6.1 Production Ledger Setup

**Files Involved:**
- `app/ledger/engine.py` - LedgerEngine class
- `app/db/models.py` - Wallet, LedgerEntry, Order, Reservation
- `app/ledger/reconciliation.py` - Coinbase sync

**Database Migrations:**

```bash
# Run migrations to create production ledger tables
uv run alembic upgrade head
```

**Tables Created:**
- `wallets` - Trading wallets linked to Coinbase accounts
- `balances` - Balance snapshots from Coinbase
- `ledger_entries` - Double-entry postings
- `reservations` - Fund locks during trade execution
- `orders` - Coinbase order records
- `cost_estimates` - Pre-trade cost evaluations

**Tasks:**
- [ ] Verify all migrations are up to date
- [ ] Create migration for any new fields
- [ ] Set up RDS with production schema
- [ ] Test migration rollback procedures
- [ ] Document schema for audit purposes

---

### 6.2 Coinbase Integration

**Files Involved:**
- `app/coinbase/client.py` - CoinbaseClient
- `app/coinbase/advanced_trade.py` - Order placement
- `app/coinbase/accounts.py` - Account queries

**Required Credentials:**
```bash
# Coinbase Advanced Trade API (CDP App)
COINBASE_API_KEY=organizations/{org_id}/apiKeys/{key_id}
COINBASE_API_SECRET=-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----
COINBASE_WALLET_SECRET=<optional, for wallet-level auth>
```

**Safety Controls Built Into CoinbaseClient:**
```python
# In app/coinbase/advanced_trade.py
if runtime.is_live:
    if not runtime.live_trading_ack:
        raise RuntimeError(
            "COINBASE ORDER BLOCKED: Cannot place real Coinbase order "
            "without explicit LIVE_TRADING_ACK=true environment variable"
        )
```

**Tasks:**
- [ ] Generate Coinbase API credentials (CDP App)
- [ ] Store credentials in Secrets Manager
- [ ] Verify API permissions (trade, read accounts)
- [ ] Test connectivity from ECS tasks
- [ ] Configure rate limiting and retry logic

---

### 6.3 Cost Gating Configuration

**Files Involved:**
- `app/costing/gate.py` - CostGate class
- `app/costing/fees.py` - FeeService
- `app/costing/slippage.py` - Slippage simulation

**How Cost Gating Works:**
```
For each trade:
1. Estimate total cost = exchange_fee + slippage + spread + transfer_fee
2. Compare against expected_edge parameter
3. Decision: proceed only if expected_edge >= total_cost * (1 + safety_buffer)
4. Log decision in cost_estimates table for audit
```

**Configuration:**

| Setting | Default | Description |
|---------|---------|-------------|
| `COST_GATE_SAFETY_BUFFER` | 0.1 (10%) | Extra margin required above costs |
| `COST_GATE_ENABLED` | true | Enable/disable cost gating |
| `COST_GATE_OVERRIDE_ALLOWED` | false | Allow manual override |

**Tasks:**
- [ ] Configure cost gate parameters for production
- [ ] Test cost gate with various market conditions
- [ ] Set up alerting for blocked trades
- [ ] Document override procedures

---

### 6.4 Wallet Seeding and Tradeable Fractions

**Initial Setup:**
```bash
# 1. Seed wallets from Coinbase accounts
uv run python -m app.cli.main ledger seed-from-coinbase

# 2. List wallets to get IDs
uv run python -m app.cli.main wallet list

# 3. Set tradeable fraction (e.g., 20% of wallet can be traded)
uv run python -m app.cli.main wallet set-tradeable-fraction 1 0.20
```

**Tradeable Fraction Safety:**
- Limits how much capital can be used for trading
- Prevents accidental full-account trades
- Can be adjusted per wallet
- Reservations further limit within the fraction

**Tasks:**
- [ ] Document wallet seeding procedure
- [ ] Set appropriate tradeable fractions for production
- [ ] Create runbook for adjusting fractions
- [ ] Test reservation logic under load

---

### 6.5 Reconciliation Setup

**Files Involved:**
- `app/ledger/reconciliation.py` - Reconciler class

**Reconciliation Process:**
```python
# Run reconciliation
uv run python -m app.cli.main reconcile run --threshold 0.0001

# What it does:
# 1. Fetch current balances from Coinbase
# 2. Compare against ledger balances
# 3. Report any drift > threshold
# 4. Optionally auto-correct (with approval)
```

**Scheduled Reconciliation:**
- Run every 15 minutes in production
- Alert on any drift > 0.01%
- Auto-halt trading on drift > 1%

**Tasks:**
- [ ] Set up scheduled reconciliation (CloudWatch Events or cron)
- [ ] Configure drift thresholds
- [ ] Create alerting for reconciliation failures
- [ ] Document manual correction procedures

---

### 6.6 TradeExecutor Wiring

**Files Involved:**
- `app/strategy/trade_executor.py` - TradeExecutor class

**Execution Pipeline:**
```
1. acquire_tradable_lock() - Reserve funds in ledger
2. cost_gate.evaluate() - Check if trade is profitable
3. place_order() - Send to Coinbase API
4. persist_order() - Record in database
5. handle_fills() - Post double-entry transactions
6. release_reservation() - Free the lock
```

**Integration Points:**

| Component | Paper Trading | Live Trading |
|-----------|---------------|--------------|
| Order Execution | `ExecutionLedgerWorkflow.record_fill()` | `TradeExecutor.execute_trade()` |
| Ledger Updates | In-memory workflow state | PostgreSQL via LedgerEngine |
| Cost Checking | None | CostGate.evaluate() |
| Coinbase API | Market data only | Full trading API |

**Tasks:**
- [ ] Wire PaperTradingWorkflow to optionally use TradeExecutor
- [ ] Add feature flag for paper vs live execution
- [ ] Test end-to-end trade flow
- [ ] Verify idempotency handling

---

## Phase 7: Monitoring & Observability

### 7.1 Prometheus Metrics

**Metrics (Both Modes):**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `trading_sessions_active` | Gauge | `mode` | Active sessions |
| `trading_cycles_total` | Counter | `session_id`, `mode` | Evaluation cycles |
| `trading_orders_total` | Counter | `session_id`, `side`, `mode` | Orders executed |
| `trading_llm_calls_total` | Counter | `session_id` | LLM API calls |
| `trading_llm_cost_dollars` | Counter | `session_id` | LLM cost |
| `trading_portfolio_equity` | Gauge | `session_id` | Portfolio equity |

**Additional Metrics (Live Trading):**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `live_orders_placed` | Counter | `product`, `side` | Real orders placed |
| `live_orders_filled` | Counter | `product`, `side` | Orders filled |
| `live_orders_failed` | Counter | `product`, `reason` | Order failures |
| `live_realized_pnl` | Gauge | `product` | Realized P&L |
| `live_cost_gate_decisions` | Counter | `decision` | proceed/blocked |
| `live_reconciliation_drift` | Gauge | `currency` | Balance drift |
| `live_ledger_entries` | Counter | `source` | Ledger postings |

**Tasks:**
- [ ] Add prometheus_client to dependencies
- [ ] Define metrics in appropriate modules
- [ ] Increment metrics at execution points
- [ ] Add `/metrics` endpoint to ops_api
- [ ] Configure Prometheus scrape target

---

### 7.2 Grafana Dashboards

**Dashboard: Trading Overview**

| Panel | Paper | Live |
|-------|-------|------|
| Active Sessions | ✓ | ✓ |
| Portfolio Equity | ✓ | ✓ |
| Trade Volume | ✓ | ✓ |
| LLM Costs | ✓ | ✓ |
| Order Success Rate | - | ✓ |
| Real P&L | - | ✓ |
| Reconciliation Status | - | ✓ |
| Cost Gate Decisions | - | ✓ |

**Tasks:**
- [ ] Create Grafana dashboard JSON
- [ ] Configure Grafana datasource for Prometheus
- [ ] Add dashboard to Terraform provisioning
- [ ] Create separate views for paper vs live

---

## Cost Estimates

### AWS Monthly (Paper Trading)

| Resource | Specification | Cost |
|----------|---------------|------|
| ECS Fargate (3 tasks) | 0.5 vCPU, 1GB each | ~$30 |
| RDS PostgreSQL | db.t3.micro, Single-AZ | ~$15 |
| ALB | 1 load balancer | ~$20 |
| NAT Gateway | 1 gateway | ~$35 |
| CloudWatch Logs | 5 GB/month | ~$5 |
| Secrets Manager | 4 secrets | ~$2 |
| **Subtotal** | | **~$107/month** |

### AWS Monthly (Live Trading)

| Resource | Specification | Cost |
|----------|---------------|------|
| ECS Fargate (3 tasks) | 1 vCPU, 2GB each | ~$60 |
| RDS PostgreSQL | db.t3.medium, **Multi-AZ** | ~$70 |
| ALB | 1 load balancer | ~$20 |
| NAT Gateway | 2 gateways (HA) | ~$70 |
| CloudWatch Logs | 10 GB/month | ~$10 |
| Secrets Manager | 6 secrets | ~$3 |
| CloudWatch Alarms | 15 alarms | ~$5 |
| **Subtotal** | | **~$238/month** |

### Optional: Temporal Cloud

| Tier | Actions/Month | Cost |
|------|---------------|------|
| Free | 10K | $0 |
| Developer | 100K | ~$50 |
| Production | 1M+ | ~$100+ |

### LLM Costs (GPT-4o-mini)

| Usage | Tokens/Week | Cost/Week |
|-------|-------------|-----------|
| 1 session, 4h plans | 336K | ~$0.50 |
| 5 sessions, 1h plans | 8.4M | ~$12 |

---

## Deployment Checklist

### Paper Trading Deployment

- [ ] Terraform apply for dev/staging
- [ ] Deploy ECS tasks
- [ ] Verify Temporal connectivity
- [ ] Test paper trading session via API
- [ ] Verify UI accessibility
- [ ] Run for 24 hours, check for issues

### Live Trading Deployment (Additional Steps)

- [ ] **Legal/Compliance Review**
- [ ] Terraform apply for production (Multi-AZ RDS)
- [ ] Store Coinbase credentials in Secrets Manager
- [ ] Run database migrations
- [ ] Seed wallets from Coinbase (`ledger seed-from-coinbase`)
- [ ] Set tradeable fractions (`wallet set-tradeable-fraction`)
- [ ] Configure PagerDuty integration
- [ ] Set up scheduled reconciliation
- [ ] **Test with minimal capital first** ($100)
- [ ] Verify cost gate is working
- [ ] Verify reconciliation is working
- [ ] Monitor for 7 days before increasing capital
- [ ] Document incident response procedures
- [ ] Set daily/weekly loss limits

---

## Safety Controls Summary

| Control | Paper | Live |
|---------|-------|------|
| `LIVE_TRADING_ACK=true` | Not required | **Required** |
| `ENABLE_REAL_LEDGER=1` | Optional | **Required** |
| Tradeable Fraction | Not applicable | **Required** (e.g., 20%) |
| Cost Gating | Not applicable | **Required** |
| Reconciliation | Not applicable | **Required** (every 15 min) |
| Multi-AZ Database | Optional | **Required** |
| Deletion Protection | Optional | **Required** |
| Backup Retention | 7 days | 30 days |
| PagerDuty Alerts | Optional | **Required** |

---

## References

- [AWS ECS Fargate Docs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
- [Temporal Cloud](https://temporal.io/cloud)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs)

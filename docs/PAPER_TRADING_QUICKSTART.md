# Paper Trading - Local Quickstart Guide

Run LLM-powered trading strategies against live market data without risking real capital.

## Prerequisites

- Docker and Docker Compose
- OpenAI API key (for LLM strategy generation)
- Node.js 18+ (for UI development)

## Quick Start

### 1. Start the Stack

```bash
# Start all services (Temporal, PostgreSQL, Worker, Ops API)
docker compose up -d

# Verify services are running
docker compose ps
```

Expected output:
```
NAME                    STATUS
crypto-db-1             Up
crypto-temporal-1       Up
crypto-worker-1         Up
crypto-ops-api-1        Up
```

### 2. Start the UI (Development Mode)

```bash
cd ui
npm install
npm run dev
```

The UI will be available at http://localhost:3000

### 3. Navigate to Paper Trading

1. Open http://localhost:3000 in your browser
2. Click the **Paper Trading** tab (green icon)

### 4. Start a Paper Trading Session

**Option A: Via UI**

1. Enter trading pairs (e.g., `BTC-USD, ETH-USD`)
2. Set initial cash (e.g., `10000`)
3. Optionally set initial allocations (e.g., `cash: 5000, BTC-USD: 3000, ETH-USD: 2000`)
4. Select a strategy template (e.g., "Momentum Trend Following")
5. Set plan interval (e.g., "Every 4 hours")
6. Click **Start Paper Trading**

**Option B: Via API**

```bash
curl -X POST http://localhost:8081/paper-trading/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["BTC-USD", "ETH-USD"],
    "initial_cash": 10000,
    "initial_allocations": {
      "cash": 5000,
      "BTC-USD": 3000,
      "ETH-USD": 2000
    },
    "strategy_id": "momentum_trend_following",
    "plan_interval_hours": 4
  }'
```

Response:
```json
{
  "session_id": "paper-trading-a1b2c3d4",
  "status": "running",
  "message": "Paper trading session started..."
}
```

### 5. Monitor the Session

**Check Status:**
```bash
curl http://localhost:8081/paper-trading/sessions/paper-trading-a1b2c3d4
```

**View Portfolio:**
```bash
curl http://localhost:8081/paper-trading/sessions/paper-trading-a1b2c3d4/portfolio
```

**View Strategy Plan:**
```bash
curl http://localhost:8081/paper-trading/sessions/paper-trading-a1b2c3d4/plan
```

**View Trades:**
```bash
curl http://localhost:8081/paper-trading/sessions/paper-trading-a1b2c3d4/trades
```

### 6. Stop the Session

**Via UI:** Click the **Stop Session** button

**Via API:**
```bash
curl -X POST http://localhost:8081/paper-trading/sessions/paper-trading-a1b2c3d4/stop
```

---

## Configuration Options

### Session Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbols` | string[] | required | Trading pairs (e.g., `["BTC-USD", "ETH-USD"]`) |
| `initial_cash` | number | 10000 | Starting cash in USD |
| `initial_allocations` | object | null | Pre-allocate capital (notional USD) |
| `strategy_id` | string | "default" | Strategy template ID |
| `strategy_prompt` | string | null | Custom LLM prompt (overrides strategy_id) |
| `plan_interval_hours` | number | 4 | How often to regenerate strategy |
| `enable_symbol_discovery` | boolean | false | Auto-discover high-volume pairs daily |
| `min_volume_24h` | number | 1000000 | Min 24h volume for discovery |
| `llm_model` | string | "gpt-5-mini" | OpenAI model to use |

### Available Strategy Templates

| ID | Name | Description |
|----|------|-------------|
| `default` | Default Strategist | Balanced approach |
| `momentum_trend_following` | Momentum/Trend Following | Ride trends with wide stops |
| `mean_reversion` | Mean Reversion | Buy oversold, sell overbought |
| `volatility_breakout` | Volatility Breakout | Trade range expansions |
| `conservative_defensive` | Conservative/Defensive | Capital preservation focus |
| `aggressive_active` | Aggressive/Active | Many trades, higher risk |
| `balanced_hybrid` | Balanced/Hybrid | Regime-adaptive |

---

## How It Works

### Architecture

```
PaperTradingWorkflow (Temporal)
├── Generates strategy plans via LLM (every N hours)
├── Evaluates triggers against live market data (every 30s)
├── Executes paper orders via ExecutionLedgerWorkflow
└── Continues-as-new for long-running durability
```

### Execution Flow

1. **Session Start**
   - Initialize portfolio with cash/positions
   - Start market data streaming
   - Generate initial strategy plan

2. **Every 30 Seconds**
   - Fetch latest prices from ExecutionLedgerWorkflow
   - Evaluate strategy triggers against market data
   - Execute any triggered orders (paper fills)

3. **Every N Hours** (configurable)
   - Collect portfolio state and market context
   - Call LLM to generate new strategy plan
   - Compile triggers for the next period

4. **Continue-as-New**
   - Every 3600 cycles or 9000 events
   - Preserves state across workflow restarts
   - Enables week-long runs

---

## Troubleshooting

### Session Not Starting

**Check Temporal UI:**
```
http://localhost:8088
```
Look for the `paper-trading-*` workflow.

**Check Worker Logs:**
```bash
docker compose logs worker -f
```

### No Trades Executing

1. **Check if plan exists:**
   ```bash
   curl http://localhost:8081/paper-trading/sessions/{id}/plan
   ```

2. **Check portfolio has cash:**
   ```bash
   curl http://localhost:8081/paper-trading/sessions/{id}/portfolio
   ```

3. **Check market data is flowing:**
   ```bash
   curl http://localhost:8081/market/ticks?limit=5
   ```

### LLM Errors

**Check OpenAI API key:**
```bash
echo $OPENAI_API_KEY
```

**Check worker logs for LLM errors:**
```bash
docker compose logs worker 2>&1 | grep -i "openai\|llm\|error"
```

### Reset Everything

```bash
# Stop and remove all containers
docker compose down -v

# Start fresh
docker compose up -d
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/paper-trading/sessions` | Start new session |
| GET | `/paper-trading/sessions` | List all sessions |
| GET | `/paper-trading/sessions/{id}` | Get session status |
| POST | `/paper-trading/sessions/{id}/stop` | Stop session |
| GET | `/paper-trading/sessions/{id}/portfolio` | Get portfolio |
| GET | `/paper-trading/sessions/{id}/plan` | Get current plan |
| GET | `/paper-trading/sessions/{id}/plans` | Get plan history (LLM insights) |
| GET | `/paper-trading/sessions/{id}/equity` | Get equity curve |
| POST | `/paper-trading/sessions/{id}/replan` | Force replan |
| PUT | `/paper-trading/sessions/{id}/strategy` | Update strategy prompt |
| GET | `/paper-trading/sessions/{id}/trades` | Get trade history |
| POST | `/paper-trading/sessions/{id}/symbols` | Update symbols |

### Example: Full Session Lifecycle

```bash
# 1. Start session
SESSION_ID=$(curl -s -X POST http://localhost:8081/paper-trading/sessions \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTC-USD"], "initial_cash": 10000}' \
  | jq -r '.session_id')

echo "Started session: $SESSION_ID"

# 2. Wait for plan generation (may take 30-60 seconds)
sleep 60

# 3. Check status
curl -s http://localhost:8081/paper-trading/sessions/$SESSION_ID | jq

# 4. Check portfolio
curl -s http://localhost:8081/paper-trading/sessions/$SESSION_ID/portfolio | jq

# 5. Let it run for a while...
sleep 300

# 6. Check trades
curl -s http://localhost:8081/paper-trading/sessions/$SESSION_ID/trades | jq

# 7. Stop session
curl -s -X POST http://localhost:8081/paper-trading/sessions/$SESSION_ID/stop | jq
```

---

## Environment Variables

Add to `.env` for customization:

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-mini

# Paper Trading Defaults
PAPER_TRADING_PLAN_INTERVAL_HOURS=4
PAPER_TRADING_CONTINUE_EVERY=3600
PAPER_TRADING_HISTORY_LIMIT=9000

# Initial Portfolio
INITIAL_PORTFOLIO_BALANCE=10000

# Temporal
TEMPORAL_ADDRESS=temporal:7233
TASK_QUEUE=mcp-tools
```

---

## Next Steps

- **Run for Extended Period:** Let a session run for 24+ hours to test durability
- **Try Different Strategies:** Experiment with different strategy templates
- **Custom Prompts:** Create your own strategy prompts in `prompts/strategies/`
- **Deploy to AWS:** See `docs/PAPER_TRADING_AWS_SCOPE.md` for cloud deployment

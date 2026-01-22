# Crypto Durable Trading Agents

A 24×7 multi-agent crypto trading stack built on Temporal and Model Context Protocol (MCP) with integrated LLM as Judge performance optimization. Every `@mcp.tool()` is backed by Temporal primitives (workflows, signals, or queries), providing deterministic execution, automatic retries and full replay for audit and compliance.

## ✨ Key Features

- **🤖 Multi-Agent Architecture**: Broker, execution, and judge agents working in coordination
- **📊 LLM as Judge System**: Autonomous performance evaluation and prompt optimization
- **🔄 Dynamic Prompt Management**: Adaptive system prompts based on trading performance
- **📈 Comprehensive Analytics**: Real-time performance metrics, risk analysis, and transaction history
- **🛡️ Risk Management**: Intelligent position sizing, drawdown protection, and portfolio monitoring
- **💰 Profit Scraping**: Configurable profit-taking to secure gains while allowing reinvestment
- **📚 Historical Data Loading**: Automatic 1-hour historical data initialization for informed startup
- **📋 Distributed Logging**: Individual agent workflow logging for better separation of concerns
- **⚡ Durable Execution**: Built on Temporal workflows for fault tolerance and auditability
- **🌐 Unified Web Dashboard**: React-based UI for backtest orchestration, live trading monitoring, wallet reconciliation, and agent inspection
- **🔌 Real-Time WebSocket Streaming**: Live market ticks, trade fills, and position updates via WebSocket connections
- **💼 Wallet Reconciliation**: Automated drift detection between ledger and exchange balances with threshold-based alerting

## Table of Contents

- [Background](#background)
- [Architecture](#architecture)
- [Durable Tools Catalog](#durable-tools-catalog)
- [Web UI & Real-Time Monitoring](#web-ui--real-time-monitoring)
- [WebSocket Configuration](#websocket-configuration)
- [Getting Started](#getting-started)
- [Demo](#demo)
- [Repository Layout](#repository-layout)
- [Contributing](#contributing)
- [License](#license)

## Background

Crypto markets never close. Building an automated trading system therefore demands:

- **Continuous orchestration** – agents must coordinate 24×7 without downtime.
- **Deterministic audit trails** – regulators require you to replay the exact decision path for every trade.
- **Cross-venue execution** – liquidity lives on both CEXs and DEXs; routing logic has to be durable.

Temporal supplies resilient workflows while MCP gives agents a shared, tool-based contract. This repo combines them into a modular engine you can extend one agent at a time.

## Architecture

The system consists of three main agents working together:

### 🏆 Single Interface Design

The **Broker Agent** serves as the sole user interface, providing access to all system functionality including trading, performance analysis, and evaluation triggering.

```
               ┌──────┐
               │ User │
               └──┬───┘
                  │
                  ▼
           ┌──────────────────┐
           │  Broker Agent    │◄────────────── Single User Interface
           │  - Trading       │                  • Stream Market Data
           │  - Analytics     │                  • Portfolio Status
  ┌────────┤  - Evaluation    │─────────────┐    • Transaction History
  │        └──────┬───────────┘             │    • User Feedback
  │               │                         │
  │               │ start_market_stream     │
  │               ▼                         │
  │        ┌──────────────────┐             │
  │        │ Market Stream WF │             │
  │        └──────┬───────────┘             │
  │               │                         │
  │               ▼                         │
  │        ┌──────────────────┐             │
  │    ┌──►│ Feature Vectors  │             │
  │    │   └──────────────────┘             │
  │    │                                    │
  │    │                                    │
  │    │ get_historical_ticks               │
  │    │                                    │
  │    │   ┌──────────────────┐             │
  │    │   │ Scheduled Nudges │             │
  │    │   └──────┬───────────┘             │ send_user_feedback
  │    │          │      ┌────────────────────────────────────┐
  │    │          ▼      ▼                                    ▼
  │    │   ┌──────────────────┐                      ┌──────────────────┐
  │    │   │ Execution Agent  │◄─────────────────────┤  Judge Agent     │
  │    └───│ - Dynamic Prompts│ get_prompt_history   │ - LLM as Judge   │
  │        │ - Risk Management│ update_system_prompt │ - Performance    │
  │        │ - Order Execution│                      │   Evaluation     │
  │        └──────┬───────────┘                      │ - Prompt Updates │
  │               │ place_mock_order                 └───────┬──────────┘
  │               ▼                                          │
  │        ┌──────────────────┐                              │
  │        │ Mock Order WF    │      get_performance_metrics │
  │        └──────┬───────────┘                              │
  │               │ fills                                    │
  │               ▼                                          │
  │        ┌──────────────────┐     ┌──────────────────┐     │
  │        │ Execution Ledger │────►│ Performance      │◄────┘
  │        │ - Transactions   │     │ Analytics        │
  │        │ - P&L Tracking   │     │ - Sharpe Ratio   │
  │        │ - Risk Metrics   │     │ - Drawdown       │
  │        └──────────────────┘     │ - Decision Qual. │
  │                 ▲               └──────────────────┘
  └─────────────────┘
   get_portfolio_status
   get_transaction_history

```

### 🤖 Execution Agent

The execution agent is the core trading decision-maker in the system:

- **Automated Trading**: Receives scheduled nudges every 25 seconds to evaluate market conditions and execute trades
- **Dynamic Adaptation**: System prompts are continuously updated by the Judge Agent based on performance
- **User Alignment**: Incorporates user preferences (risk tolerance, trading style) and feedback into decision-making
- **Portfolio Management**: Queries portfolio status and recent historical ticks before making buy/sell/hold decisions

### 🔄 LLM as Judge Agent

The judge agent continuously monitors performance and automatically optimizes the execution agent:

1. **Performance Monitoring**: Evaluates trading performance dynamically (10-minute startup delay, then adaptive timing)
2. **Decision Analysis**: Uses GPT-4o to analyze decision quality and timing
3. **Prompt Optimization**: Automatically updates execution agent prompts based on performance
4. **Risk Management**: Switches to conservative mode during poor performance periods

Each block corresponds to one or more MCP tools (Temporal workflows) described below.

## Durable Tools Catalog

### Core Trading Tools

| Tool                           | Primitive | Purpose                                                          | Typical Triggers                            |
| ------------------------------ | --------- | ---------------------------------------------------------------- | ------------------------------------------- |
| `subscribe_cex_stream`         | Workflow  | Fan-in ticker data from centralized exchanges                    | Startup, reconnect                          |
| `start_market_stream`          | Workflow  | Begin streaming market data for selected pairs                   | Auto-started by broker after pair selection |
| `HistoricalDataLoaderWorkflow` | Workflow  | Load 1-hour historical data for informed startup                 | System initialization                       |
| `place_mock_order`             | Workflow  | Simulate order execution and return a fill                       | Portfolio rebalance                         |
| `ExecutionLedgerWorkflow`      | Workflow  | Track fills, positions, transaction history, and profit scraping | Fill events                                 |

### Performance & Analytics Tools

| Tool                      | Primitive | Purpose                                     | Typical Triggers          |
| ------------------------- | --------- | ------------------------------------------- | ------------------------- |
| `get_transaction_history` | Query     | Retrieve filtered transaction history       | User queries, evaluations |
| `get_performance_metrics` | Query     | Calculate returns, Sharpe ratio, win rates  | Performance analysis      |
| `get_risk_metrics`        | Query     | Analyze position concentration and leverage | Risk monitoring           |
| `get_portfolio_status`    | Query     | Current cash, positions, and P&L            | User queries              |

### Judge Agent Tools

| Tool                             | Primitive | Purpose                                     | Typical Triggers               |
| -------------------------------- | --------- | ------------------------------------------- | ------------------------------ |
| `trigger_performance_evaluation` | Signal    | Force immediate performance evaluation      | User request, poor performance |
| `get_judge_evaluations`          | Query     | Retrieve recent performance evaluations     | User queries                   |
| `get_prompt_history`             | Query     | View prompt evolution and version history   | System monitoring              |
| `JudgeAgentWorkflow`             | Workflow  | Manage evaluation state and prompt versions | Judge agent lifecycle          |

### User Interaction Tools

| Tool                   | Primitive | Purpose                                    | Typical Triggers   |
| ---------------------- | --------- | ------------------------------------------ | ------------------ |
| `set_user_preferences` | Signal    | Update trading preferences (risk, style)   | User configuration |
| `get_user_preferences` | Query     | Retrieve current user trading preferences  | Agent decisions    |
| `send_user_feedback`   | Signal    | Send feedback to execution or judge agents | User interaction   |
| `get_pending_feedback` | Query     | Retrieve unprocessed user feedback         | Agent processing   |

### Agent Workflows

| Workflow Name            | Purpose                                         | Typical Triggers     |
| ------------------------ | ----------------------------------------------- | -------------------- |
| `ExecutionAgentWorkflow` | Individual execution agent logging and state    | Agent decisions      |
| `JudgeAgentWorkflow`     | Individual judge agent logging and evaluations  | Performance analysis |
| `BrokerAgentWorkflow`    | Broker agent state and user interaction logging | User interactions    |

## Web UI & Real-Time Monitoring

The system includes a modern React-based web dashboard (`ui/`) that provides comprehensive monitoring and control capabilities:

### Dashboard Features

**Backtest Control Tab**
- Configure and launch backtests with custom parameters (symbols, timeframe, initial cash, risk settings)
- Monitor backtest progress in real-time with live status updates
- View equity curves, performance metrics, and trade history
- Analyze daily reports with detailed breakdowns of trades, blocks, and risk budget usage

**Live Trading Monitor Tab**
- Real-time position tracking with P&L calculations
- Recent fills and trade execution logs
- Trade block monitoring with categorized reasons (insufficient budget, max concentration, etc.)
- Risk budget allocation and usage visualization
- Interactive market ticker with WebSocket-powered live price updates

**Wallet Reconciliation Tab**
- View all configured wallets with current ledger balances
- Trigger on-demand reconciliation against exchange (Coinbase) balances
- Drift detection with configurable thresholds
- Color-coded status indicators for balance discrepancies
- Tradeable fraction configuration per wallet

**Agent Inspector Tab**
- Trace decision chains via correlation ID linking
- Event filtering by type, source, run_id, or correlation_id
- LLM telemetry monitoring (model usage, token counts, cost estimates)
- Workflow status cards (Broker, Execution, Judge agents)
- Real-time event stream via WebSocket with polling fallback

### Accessing the Dashboard

1. Start the full stack:
   ```bash
   docker compose up
   ```

2. Start the UI development server:
   ```bash
   cd ui && npm run dev
   ```

3. Open your browser to `http://localhost:3000` (or the port shown in terminal)

The dashboard automatically connects to the Ops API at `localhost:8081` and establishes WebSocket connections for real-time updates.

## WebSocket Configuration

The system uses WebSocket connections for real-time data streaming (market ticks, trade fills, position updates). The UI automatically constructs WebSocket URLs based on the deployment environment.

### Environment Variables

Configure WebSocket endpoints via these environment variables in `ui/.env`:

```bash
# Option 1: Explicit WebSocket URL (highest priority)
VITE_WS_URL=ws://localhost:8081

# Option 2: API URL (automatically converted to ws/wss)
VITE_API_URL=http://localhost:8081

# If neither is set, defaults to current window.location with port 8081 in dev mode
```

### WebSocket Endpoints

The Ops API exposes two WebSocket endpoints:

- **`/ws/live`** - Live trading updates (fills, positions, blocks, risk budget, agent events)
- **`/ws/market`** - Market data updates (ticks, price changes, symbol updates)

### Connection Behavior

- **Automatic Reconnection**: WebSocket hook retries connection with exponential backoff (default 3s delay)
- **Heartbeat/Keep-Alive**: Ping/pong messages every 30 seconds to maintain connection
- **Graceful Fallback**: UI components fall back to HTTP polling if WebSocket unavailable
- **Environment-Aware**: Automatically uses `wss://` for HTTPS deployments and `ws://` for HTTP

### Testing WebSocket Connection

Check WebSocket connection stats:
```bash
curl http://localhost:8081/ws/stats
# Response: {"live_connections": 1, "market_connections": 1}
```

Test WebSocket connection manually (requires `wscat`):
```bash
npm install -g wscat
wscat -c ws://localhost:8081/ws/market

# Send ping
> ping

# Receive pong
< {"type": "pong"}
```

### Production Deployment

For production deployments behind load balancers or proxies:

1. Configure explicit WebSocket URL:
   ```bash
   VITE_WS_URL=wss://your-domain.com
   ```

2. Ensure your proxy/load balancer supports WebSocket upgrade:
   ```nginx
   # Nginx example
   location /ws/ {
       proxy_pass http://ops-api:8081;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
       proxy_set_header Host $host;
   }
   ```

3. For custom API hosts (e.g., internal DNS):
   ```bash
   VITE_API_URL=https://api.internal.company.com:8081
   # Automatically becomes wss://api.internal.company.com:8081
   ```

## Getting Started (Compose-first)

### Prerequisites

| Requirement  | Version       | Notes                                         |
| ------------ | ------------- | --------------------------------------------- |
| Python       | 3.11 or newer | Data & strategy agents                        |
| Temporal CLI | 1.24+         | `brew install temporal` or use Temporal Cloud |
| Docker + Compose | latest    | Required for `docker compose up` bootstrap    |

Required environment variables:

- `OPENAI_API_KEY` – enables the broker and execution agents to use OpenAI models.
- `COINBASE_API_KEY` / `COINBASE_API_SECRET` – Coinbase App (CDP) API key pair; secrets can be pasted directly from the downloaded JSON (newlines are supported).
- `COINBASE_WALLET_SECRET` – optional, only required for wallet-authenticated POST/DELETE calls. Leave empty for read-only trading.
- `COINBASEEXCHANGE_API_KEY` and `COINBASEEXCHANGE_SECRET` – legacy Advanced Trade credentials (fallback if you still need HMAC auth).
- `TEMPORAL_ADDRESS`, `TEMPORAL_NAMESPACE` and `TASK_QUEUE` – Temporal connection settings (defaults are shown in `.env`).
- `MCP_PORT` – port for the MCP server (defaults to `8080`).
- `HISTORICAL_MINUTES` – minutes of historical data to load on startup (defaults to `60` for 1 hour).

### Preparing Coinbase Wallets for Live Trading

1. **Fund Coinbase wallets** – Deposit ETH and BTC into the Coinbase account tied to your API key. Only these balances will be used; live mode also requires `TRADING_MODE=live` **and** `LIVE_TRADING_ACK=true`.
2. **Seed the ledger** – Pull wallets/balances into the internal database so Temporal workflows see real holdings:
   ```bash
   UV_CACHE_DIR=.uv-cache uv run python -m app.cli.main ledger seed-from-coinbase
   ```
3. **Inspect wallet IDs and cached balances** – Use the wallet inspector to note the `wallet_id` for ETH and BTC:
   ```bash
   UV_CACHE_DIR=.uv-cache uv run python -m app.cli.main wallet list
   ```
4. **Mark 20% as tradable** – For each funded wallet, set the tradeable fraction to `0.20` (repeat per wallet):
   ```bash
   UV_CACHE_DIR=.uv-cache uv run python -m app.cli.main wallet set-tradeable-fraction <wallet_id> 0.20
   ```
   These fractions gate how much the execution agent may reserve or spend per wallet.
5. **Reconcile before trading** – Confirm ledger entries match Coinbase using the reconciliation tool:
   ```bash
   UV_CACHE_DIR=.uv-cache uv run python -m app.cli.main reconcile run
   ```
6. **Dry-run trading flows** – Keep `RUN_MODE=dev` (or disable actual order placement in prompts) while observing the broker/execution agents. Only remove the guard once you’re satisfied with monitoring, risk limits, and logging.

### Quick Setup (local dev)

```bash
# Clone and bootstrap
git clone https://github.com/your-org/durable-crypto-agents.git
cd durable-crypto-agents

# Activate Python env
python -m venv .venv && source .venv/bin/activate
pip install uv
uv sync

### Coinbase Ledger & Trading

The production ledger stack lives under `app/` and exposes a CLI for seeding wallets, reserving balances, and placing live trades via Coinbase Advanced Trade.

1. Copy environment defaults and update with your credentials:

   ```bash
   cp .env.example .env
   ```

2. Install dependencies and start the Postgres service with migrations:

   ```bash
   make init
   make db-up
   ```

3. Seed wallets directly from Coinbase (creates internal ledger balances and reservations backing):

   ```bash
   uv run python -m app.cli.main ledger seed-from-coinbase
   ```

4. Configure what fraction of each wallet is available to the trading bots:

   ```bash
   uv run python -m app.cli.main wallet set-tradeable-fraction --wallet 1 --frac 0.5
   ```

5. Place a cost-gated trade **(LIVE Coinbase call — not a mock)**:

   ```bash
   uv run python -m app.cli.main trade place \
     --wallet 1 \
     --product BTC-USD \
     --side buy \
     --qty 0.01 \
     --notional 300 \
     --expected-edge 15 \
     --idempotency-key demo-trade-001
   ```
   ⚠️ This hits Coinbase Advanced Trade with your API keys. There is no paper/dry-run flag in this CLI; only the cost gate and your wallet fraction guard execution. Use sandbox/test credentials or skip this step if you do not intend to place a real order.

6. Reconcile the internal ledger with Coinbase balances (writes a drift report to stdout):

   ```bash
   uv run python -m app.cli.main reconcile run --threshold 0.0001
   ```

### Web Dashboard (Human Supervisor)

Replace the tmux-based workflow with a lightweight web UI that supervises all agents, shows ledger snapshots, and keeps a human in the loop.

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.dashboard
```

Open <http://localhost:8081/> to:

- Start/stop Temporal, the worker, MCP server, and broker/execution/judge agents from a single panel.
- Inspect Coinbase-linked wallets, tradable fractions, and balances per strategy/portfolio.
- Review the short-term backlog (see `docs/ROADMAP.md`) and plan future automation without leaving the UI.

# Launch the full stack (Compose)
```bash
# baseline (agent stack + MCP + Ops API + UI)
docker compose up

# legacy services only when you explicitly need them
docker compose --profile legacy_live up
```

What runs:
- MCP server at `http://localhost:8080` (tools/endpoints).
- Ops API + UI at `http://localhost:8081/` (UI served by the Ops API).
- Temporal dev server at `localhost:7233`.

Notes:
- Set `OPENAI_API_KEY` (and other secrets) in your shell or `.env` before composing.
- Only use the `legacy_live` profile when you intentionally need legacy services; default stack is agent-only.

## Demo

The quickest way to see the stack in action is to run `docker compose up`, which launches the Temporal dev server, Python worker, MCP server, Ops API, and (once wired) UI. Logs stream via container output; use `docker compose logs -f` to inspect services. To stop, press `Ctrl+C` in the compose terminal.

### Walking through the demo

1. Run `docker compose up`
2. When prompted for trading pairs, tell the broker agent **"BTC/USD, ETH/USD, DOGE/USD"** (recommended 2-4 pairs for optimal performance).
3. `start_market_stream` automatically loads 1 hour of historical data, then spawns a `subscribe_cex_stream` workflow that broadcasts each ticker to its `ComputeFeatureVector` child.
4. The execution agent wakes up periodically via a scheduled workflow and analyzes market data to decide whether to trade using `place_mock_order`.
5. Filled orders are recorded in the `ExecutionLedgerWorkflow` with automatic profit scraping.
6. The judge agent monitors performance autonomously (10-minute startup delay) and can be queried through the broker:
   - **"How is the system performing?"** - Triggers evaluation and shows metrics
   - **"What's the transaction history?"** - Shows recent trades and fills
   - **"Evaluate performance"** - Forces immediate performance analysis

### 🤖 Interacting with the System

The broker agent serves as your single interface. Try these commands:

**Trading Commands:**

- `"Start trading BTC/USD and ETH/USD"` (or choose from expanded DeFi/Layer 2 options)
- `"What's my portfolio status?"` (includes profit scraping details)
- `"Show me recent transactions"`

**Performance Analysis:**

- `"How is the system performing?"`
- `"Trigger a performance evaluation"`
- `"Show me the latest evaluation results"`
- `"What are the current risk metrics?"`

**System Insights:**

- `"Show prompt evolution history"`
- `"What performance trends do you see?"`

`subscribe_cex_stream` automatically restarts itself via Temporal's _continue as new_
mechanism after a configurable number of cycles to prevent unbounded workflow
history growth. The default interval is one hour (3600 cycles) and can be
changed by setting the `STREAM_CONTINUE_EVERY` environment variable. The workflow
also checks its current history length and continues early when it exceeds
`STREAM_HISTORY_LIMIT` (defaults to 9000 events).
`ComputeFeatureVector` behaves the same way using the `VECTOR_CONTINUE_EVERY`
and `VECTOR_HISTORY_LIMIT` environment variables.

## Repository Layout

```
├── agents/                    # Multi-agent system components
│   ├── broker_agent_client.py    # Single user interface agent
│   ├── execution_agent_client.py # Trading decision agent
│   ├── judge_agent_client.py     # LLM as Judge performance optimizer
│   ├── workflows.py               # Temporal workflow definitions
│   ├── context_manager.py         # Intelligent conversation management
├── tools/                     # Durable workflows used as MCP tools
│   ├── performance_analysis.py   # Performance metrics and analysis
│   ├── market_data.py            # Market data streaming
│   ├── execution.py              # Order execution
│   └── ...
├── mcp_server/               # FastAPI server exposing the tools
├── ops_api/                  # Ops API (UI backend) for status/controls
├── worker/                   # Temporal worker (agent/legacy split)
├── tests/                    # Unit tests for tools and agents
├── docker-compose.yml        # Canonical bootstrap
└── ticker_ui_service.py     # Simple websocket ticker UI
```

### Key Components

- **`broker_agent_client.py`**: Main user interface providing access to all system functionality
- **`execution_agent_client.py`**: Autonomous trading agent with dynamic prompt loading
- **`judge_agent_client.py`**: Performance evaluator using LLM as Judge pattern
- **`workflows.py`**: Temporal workflows for state management and coordination
- **`context_manager.py`**: Token-based conversation management with summarization
- **`performance_analysis.py`**: Comprehensive trading performance analysis tools
- **`market_data.py`**: Historical data loading and streaming with configurable windows
- **`agent_logger.py`**: Distributed logging system routing to individual agent workflows
- **`ops_api/`**: Ops API (UI backend) exposing status, block reasons, events, and telemetry

## 🧠 LLM as Judge System

The system implements a sophisticated "LLM as Judge" pattern for continuous self-improvement:

### Performance Evaluation Framework

- **Multi-dimensional Analysis**: Evaluates returns, risk management, decision quality, and consistency
- **Automated Scoring**: Combines quantitative metrics with LLM-based decision analysis
- **Trend Detection**: Identifies performance patterns and degradation early

### Dynamic Prompt Optimization

- **Template System**: Modular prompt components for different trading scenarios
- **Automatic Updates**: Switches between conservative, standard, and aggressive modes based on performance
- **Version Tracking**: Full history of prompt changes with performance attribution

### Intelligent Triggers

- **Poor Performance**: Automatic intervention when scores drop below thresholds
- **High Drawdown**: Emergency conservative mode activation
- **Overly Conservative**: Increased risk-taking when performance is too safe

### Example Evaluation Cycle

```python
# Triggered dynamically based on performance or on-demand via broker agent
1. Collect transaction history and portfolio metrics
2. Calculate quantitative performance (Sharpe, drawdown, win rate)
3. Use GPT-4o to analyze decision quality and timing
4. Generate weighted overall score across four dimensions
5. Determine if prompt updates are needed
6. Implement changes and track effectiveness
```

## 💰 Profit Scraping System

The system includes intelligent profit-taking to secure gains while maintaining trading capital:

### User-Configurable Profit Taking

- **Scraping Percentage**: Users set percentage of profits to secure (e.g., 20%)
- **Automatic Execution**: Applied to all profitable trades automatically
- **Separation of Concerns**: Scraped profits kept separate from trading capital
- **Portfolio Tracking**: Displays both available cash and total cash value

### Example Flow

```python
# User sets 20% profit scraping preference
# Trade: Buy BTC at $50,000, Sell at $55,000 = $5,000 profit
# Result: $4,000 reinvested, $1,000 scraped and secured
```

## Contributing

Pull requests are welcome! Please open an issue first to discuss your proposed change. Make sure to:

- Run `make lint test` and fix any CI failures.
- Keep new tools deterministic (no nondeterministic I/O inside workflows).
- Write docs – every public agent or tool needs at least minimal usage notes.

## License

This project is released under the MIT License – see `LICENSE` for details.

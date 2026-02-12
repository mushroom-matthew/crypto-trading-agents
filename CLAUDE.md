# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a 24×7 multi-agent crypto trading system built on Temporal workflows and Model Context Protocol (MCP). The system implements durable execution patterns where every `@mcp.tool()` is backed by Temporal primitives (workflows, signals, queries) for deterministic, fault-tolerant trading operations.

**Three core agents:**
- **Broker Agent** (`agents/broker_agent_client.py`) - Single user interface for all system functionality
- **Execution Agent** (`agents/execution_agent_client.py`) - Autonomous trading decisions with dynamic prompt loading
- **Judge Agent** (`agents/judge_agent_client.py`) - LLM as Judge pattern for performance optimization

## Runbook Workflow Order (Mandatory)

When a task references a runbook in `docs/branching/`, follow this order:
1) Run the Worktree Setup and create the branch **before** editing any files.
2) Implement changes.
3) Run the Test Plan and paste output into the runbook Test Evidence section.
4) Fill Human Verification Evidence and Change Log entries.
5) Commit only after evidence is recorded, using the runbook Git Workflow.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Copy environment template and configure credentials
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, COINBASE_API_KEY, etc.
```

### Running the System
```bash
# Start full stack via Docker Compose (recommended)
docker compose up

# Start database only
make db-up

# Run database migrations
uv run alembic upgrade head

# Generate new migration
make migrate name="description"
```

### Testing
```bash
# Run all tests
make test
# or
uv run pytest

# Run specific test pattern
uv run pytest -k test_ledger -vv

# Run single test file
uv run pytest tests/test_ledger_pnl.py
```

### CLI Tools
```bash
# Seed ledger from Coinbase accounts
uv run python -m app.cli.main ledger seed-from-coinbase

# List wallets with IDs
uv run python -m app.cli.main wallet list

# Set tradeable fraction for a wallet
uv run python -m app.cli.main wallet set-tradeable-fraction <wallet_id> 0.20

# Place a live trade (REAL COINBASE CALL - NO DRY RUN MODE)
uv run python -m app.cli.main trade place \
  --wallet 1 \
  --product BTC-USD \
  --side buy \
  --qty 0.01 \
  --notional 300 \
  --expected-edge 15 \
  --idempotency-key demo-trade-001

# Reconcile ledger with Coinbase
make reconcile
# or
uv run python -m app.cli.main reconcile run --threshold 0.0001
```

### Backtesting
```bash
# Run backtest via CLI
backtest --help
uv run python -m backtesting.cli --help
```

## Architecture

### Multi-Agent Coordination

The system uses a **single interface design** - all user interactions go through the Broker Agent, which coordinates with Execution and Judge agents via Temporal workflows and signals.

**Data Flow:**
1. Broker Agent starts `MarketStreamWorkflow` via `start_market_stream` tool
2. Market stream loads historical data (`HistoricalDataLoaderWorkflow`) then starts `subscribe_cex_stream`
3. Each ticker spawns a `ComputeFeatureVector` child workflow for technical indicators
4. Execution Agent receives scheduled nudges (every 25s) to evaluate market conditions
5. Orders placed via `place_mock_order` trigger `MockOrderWorkflow` which fills to `ExecutionLedgerWorkflow`
6. Judge Agent monitors performance and updates execution agent prompts dynamically
7. User feedback and preferences flow through signals to agent workflows

### Durable Execution Pattern

Every MCP tool is a Temporal workflow, signal, or query. This means:
- **Deterministic replay** - audit trails for regulatory compliance
- **Automatic retries** - network failures don't lose state
- **Long-running coordination** - agents run 24×7 without process restarts
- **Event sourcing** - full history of decisions and state changes

### Workflow Continue-as-New

Workflows use Temporal's continue-as-new pattern to prevent unbounded history growth:
- `subscribe_cex_stream`: continues every `STREAM_CONTINUE_EVERY` cycles (default 3600) or when history exceeds `STREAM_HISTORY_LIMIT` (9000 events)
- `ComputeFeatureVector`: continues every `VECTOR_CONTINUE_EVERY` cycles or when history exceeds `VECTOR_HISTORY_LIMIT`

### Key Workflows

**Agent Workflows** (in `agents/workflows/`):
- `BrokerAgentWorkflow` - Broker agent state and user interaction logging
- `ExecutionAgentWorkflow` - Execution agent logging and dynamic prompt management
- `JudgeAgentWorkflow` - Performance evaluation state and prompt version tracking
- `ExecutionLedgerWorkflow` - Portfolio state, P&L tracking, profit scraping

**Trading Workflows** (exposed as MCP tools in `tools/`):
- `start_market_stream` - Initialize market data streaming with historical backfill
- `place_mock_order` - Execute simulated trades with realistic fills
- `subscribe_cex_stream` - Fan-in ticker data from exchanges
- `HistoricalDataLoaderWorkflow` - Load configurable historical windows (default 60 min)

**Performance Tools** (`tools/performance_analysis.py`):
- `get_transaction_history` - Query filtered transaction history
- `get_performance_metrics` - Sharpe ratio, returns, win rates
- `get_risk_metrics` - Position concentration, leverage analysis
- `get_portfolio_status` - Current cash, positions, P&L

### Dual Ledger Architecture

The system maintains **two separate ledger implementations**:

1. **Mock Ledger** (default) - In `ExecutionLedgerWorkflow`
   - Pure Temporal workflow state (no external DB)
   - Tracks cash, positions, fills, profit scraping
   - Used by default for all agent trading

2. **Production Ledger** - In `app/ledger/`
   - PostgreSQL-backed with SQLAlchemy models (`app/db/models.py`)
   - Double-entry accounting with wallets, balances, reservations
   - Coinbase integration via `app/coinbase/client.py`
   - Cost gating and fee calculation in `app/costing/`
   - Accessed via CLI commands, not currently wired to agent workflows

**Important:** The agents use the mock ledger by default. The production ledger (`app/`) is managed via CLI commands and requires explicit `ENABLE_REAL_LEDGER=1` in preferences to route agent trades through it.

### Worker Architecture

The system runs two Temporal workers (`worker/`):

1. **Agent Worker** (`worker/agent_worker.py`) - Default
   - Runs broker, execution, and judge agent workflows
   - Handles MCP tool workflows (market data, execution, performance)
   - Started via `docker compose up` or `uv run python -m worker.main`

2. **Legacy Live Worker** (`worker/legacy_live_worker.py`) - Optional
   - Handles legacy workflows if needed
   - Only started with `docker compose --profile legacy_live up`

### Prompt Management

The Execution Agent uses a dynamic prompt system (`agents/prompt_manager.py`):
- Prompts stored as Jinja2 templates in `prompts/`
- Judge Agent updates prompts based on performance via `update_system_prompt` signal
- Modes: conservative (poor performance), standard (baseline), aggressive (strong performance)
- Full version history tracked in `JudgeAgentWorkflow` via `get_prompt_history`

### LLM as Judge System

The Judge Agent (`agents/judge_agent_client.py`) implements autonomous optimization:

1. **Performance Monitoring**: Evaluates metrics every 10 minutes (after startup delay)
2. **Multi-dimensional Scoring**: Returns (0-100), risk (0-100), decision quality (0-100), consistency (0-100)
3. **Automatic Intervention**: Switches prompts when scores drop below thresholds
4. **Evaluation Triggers**:
   - Scheduled (dynamic timing based on performance)
   - User-initiated via `trigger_performance_evaluation` signal
   - Automatic on high drawdown or overly conservative behavior

Scoring thresholds configured via `JUDGE_REPLAN_SCORE_THRESHOLD` (default 45) with cooldown via `JUDGE_REPLAN_COOLDOWN_SECONDS`.

### Profit Scraping

The system supports configurable profit-taking (`ExecutionLedgerWorkflow`):
- Users set percentage via `set_user_preferences` signal (default 20%)
- Applied automatically to all profitable trades
- Scraped profits stored separately from trading capital
- Visible in portfolio status as `scraped_profits` vs `available_cash`

### Context Management

All agents use token-based conversation management (`agents/context_manager.py`):
- Automatic summarization when approaching context limits
- Preserves recent messages while summarizing older conversation
- Uses tiktoken for accurate token counting
- Configurable max tokens per agent

### Logging and Observability

**Distributed Logging** (`tools/agent_logger.py`):
- Each agent workflow maintains its own log buffer
- Logs routed to agent-specific workflows (Broker/Execution/Judge)
- Queryable via workflow queries for debugging

**LLM Instrumentation** (`agents/langfuse_utils.py`):
- All LLM calls wrapped in Langfuse spans
- Captures prompts, responses, token usage
- Required for audit trails - don't add OpenAI clients without Langfuse wrapping

**Dual Server Architecture:**

The system runs TWO separate HTTP servers with distinct purposes:

1. **MCP Server** (`mcp_server/app.py`) - Port 8080 - Agent/Programmatic Interface
   - FastMCP wrapper around FastAPI for Claude Desktop integration
   - Exposes trading operations as `@mcp.tool()` decorators
   - Used by: Agents, Claude Desktop, programmatic API clients
   - **When to add endpoints here:**
     - Trading operations (place_mock_order, start_market_stream)
     - Agent interactions (set_user_preferences, send_user_feedback)
     - Workflow queries (get_portfolio_status, get_transaction_history)
     - Real-time signals (POST /signal/{name})
   - **Pattern:** `@app.tool()` for MCP tools, `@app.custom_route()` for HTTP endpoints

2. **Ops API** (`ops_api/app.py`) - Port 8081 - Human Operator Interface
   - Pure FastAPI for web dashboards and monitoring
   - Exposes status, events, telemetry for UI consumption
   - Used by: React UI, monitoring dashboards, human operators
   - **When to add endpoints here:**
     - Backtesting (POST /backtests, GET /backtests/{id}/results)
     - Live trading monitoring (GET /live/fills, GET /live/positions)
     - Market data visualization (GET /market/candles, GET /market/tickers)
     - Analytics and reports (GET /wallets/reconciliation)
   - **Pattern:** Standard FastAPI `@router.get()`, `@router.post()` with APIRouter
   - Event store pattern with materialized views

**Port Assignment:**
- `8080`: MCP Server (agent tools)
- `8081`: Ops API (operator dashboards)
- `8088`: Temporal UI
- `5432`: PostgreSQL
- `7233`: Temporal gRPC

## Environment Variables

Critical variables (see `.env.example` for full list):

**LLM & Observability:**
- `OPENAI_API_KEY` - Required for broker/execution/judge agents
- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` - LLM instrumentation

**Temporal:**
- `TEMPORAL_ADDRESS` - Default `temporal:7233` (compose) or `localhost:7233` (local)
- `TEMPORAL_NAMESPACE` - Default `default`
- `TASK_QUEUE` - Default `mcp-tools`

**Trading:**
- `INITIAL_PORTFOLIO_BALANCE` - Starting cash for mock ledger (default 1000)
- `HISTORICAL_MINUTES` - Historical data window on startup (default 60)
- `EXECUTION_MIN_PRICE_MOVE_PCT` - Min price move to consider trading (default 0.5%)
- `EXECUTION_MAX_STALENESS_SECONDS` - Max age for price data (default 1800s)

**Coinbase Production Ledger:**
- `COINBASE_API_KEY`, `COINBASE_API_SECRET` - Coinbase App (CDP) credentials
- `COINBASE_WALLET_SECRET` - Required for wallet-authenticated operations
- `ENABLE_REAL_LEDGER` - Set to `1` to route agent trades through production ledger

**Database:**
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `DB_DSN` - Production ledger connection string

## Coding Conventions

**Python Style:**
- Python 3.11+ with four-space indentation
- `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Type hints required for public APIs
- Async/await throughout - wrap blocking calls with `asyncio.to_thread`

**Temporal Workflows:**
- Workflows must be deterministic - no direct I/O, use activities for external calls
- Use `workflow.logger` not standard `logging` inside `@workflow.defn`
- Signal handlers for state updates, queries for reads
- Continue-as-new to prevent unbounded history growth

**Testing:**
- Mirror source structure in `tests/`
- Name files `test_<feature>.py`
- Mock external services (Coinbase, OpenAI) in workflow tests
- Verify deterministic replay behavior for Temporal workflows

**Common Test Pitfalls:**
- Always run `uv run pytest -k <test_name> -vv` before committing to catch failures early
- Pydantic models use `extra="forbid"` (SerializableModel base) -- ALL required fields must be present in test fixtures
- Tests instantiate workflow classes directly as plain Python objects (no Temporal runtime needed)
- Mock OpenAI with `SimpleNamespace` stubs, not heavy mock libraries: `SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: SimpleNamespace(output_text='...')))`
- No top-level conftest.py -- each test file is self-contained with inline helpers
- When editing schemas, grep for ALL downstream consumers and test fixtures before changing field names
- Stub plan provider available: `tests/helpers/stub_plan_provider.py` (AlwaysLongPlanProvider, no LLM calls)

**Domain Invariants (Business Logic):**
- Judge evaluations trigger strategy updates -- LLM budget calculations must account for BOTH judge_evals AND strategy_replans
- Emergency exits (`category="emergency_exit"`) MUST bypass all other constraints (budget caps, category restrictions, stand-down state) but still write to the ledger
- Profit scraping happens AFTER fill, not during order placement
- `risk_used=0` is a valid value (falsy-zero bug) -- always use `is not None` checks, never truthiness
- Daily risk budget resets at midnight UTC; max_daily_loss is separate from risk_budget (loss = realized P&L, budget = exposure)
- StrategyRun is the central entity linking plans, judge feedback, and config -- changes cascade through it
- Judge actions have TTL (in evals) and expire automatically -- don't assume they persist indefinitely

**Code Editing Conventions:**
- When using the Edit tool, include 3+ lines of surrounding context in old_string for uniqueness
- For repeated patterns (e.g., multiple similar dict entries, function signatures), include the entire block
- Prefer targeted edits over large rewrites -- smaller old_string = higher success rate
- Before changing any Pydantic schema field: `grep -r "field_name"` across the entire codebase, update all consumers in the same commit

**Logging:**
- Use `agents.logging_utils.setup_logging()` for structured output
- LLM calls must be wrapped in Langfuse spans
- Workflow logs via `tools/agent_logger.py` for distributed logging

## Common Patterns

### Choosing Between MCP Server vs Ops API

**Use MCP Server (port 8080) when:**
- Exposing functionality to agents or Claude Desktop
- Building tools that modify system state (place orders, set preferences)
- Creating real-time signal endpoints
- Implementing workflow triggers

**Use Ops API (port 8081) when:**
- Building UI dashboards for human operators
- Creating read-only analytics endpoints
- Implementing backtest orchestration
- Adding monitoring/observability features

**Example Decision Tree:**
- "Add backtest control" → Ops API (operator feature)
- "Add place_order tool" → MCP Server (agent tool)
- "Add live trade monitoring" → Ops API (operator dashboard)
- "Add set_risk_limits" → MCP Server (agent configuration)

### Adding a New MCP Tool

1. Define workflow in `tools/<category>.py`:
```python
@workflow.defn
class MyToolWorkflow:
    @workflow.run
    async def run(self, param: str) -> dict:
        # Deterministic logic only
        return {"result": param}
```

2. Register in `mcp_server/app.py`:
```python
@mcp.tool()
async def my_tool(param: str) -> dict:
    client = await get_temporal_client()
    result = await client.execute_workflow(
        MyToolWorkflow.run,
        param,
        id=f"my-tool-{uuid4()}",
        task_queue=TASK_QUEUE
    )
    return result
```

3. Add workflow to worker in `worker/agent_worker.py`:
```python
worker = Worker(
    client,
    task_queue=TASK_QUEUE,
    workflows=[..., MyToolWorkflow],
    activities=[...]
)
```

### Sending Signals to Workflows

```python
# Get workflow handle
handle = client.get_workflow_handle("execution-ledger-main")

# Send signal
await handle.signal(ExecutionLedgerWorkflow.set_user_preferences, {
    "profit_scraping_percentage": "30%",
    "enable_real_ledger": True
})
```

### Querying Workflow State

```python
# Query portfolio status
status = await handle.query(ExecutionLedgerWorkflow.get_portfolio_status)
print(status["cash"], status["positions"])
```

### Adding a New Ops API Endpoint

1. Define endpoint in `ops_api/routers/<category>.py`:
```python
from fastapi import APIRouter

router = APIRouter(prefix="/backtests", tags=["backtests"])

@router.get("/{run_id}/results")
async def get_backtest_results(run_id: str):
    # Implementation
    return {"run_id": run_id, "results": ...}
```

2. Include router in `ops_api/app.py`:
```python
from ops_api.routers import backtests

app.include_router(backtests.router)
```

3. Update UI to consume endpoint at `http://localhost:8081/backtests/{id}/results`

## Important Notes

- **Real Trading Risk**: The `app.cli.main trade place` command makes REAL Coinbase API calls with no dry-run mode. Cost gates and wallet fractions are the only guards.
- **Dual Workers**: Default agent worker is sufficient for normal operation. Only use `legacy_live` profile if explicitly needed.
- **Ledger Separation**: Mock ledger (workflows) vs production ledger (app/) are separate systems. Agents use mock by default.
- **Workflow History**: Long-running workflows must implement continue-as-new to avoid hitting Temporal's history size limits.
- **Determinism**: Workflows cannot make HTTP calls, use random(), or read system time directly - use activities for non-deterministic operations.
- **API Compatibility**:
  - MCP tools at port 8080 (agent interface)
  - Ops API at port 8081 (operator interface)
  - Temporal UI at port 8088
  - NEVER mix MCP tools into Ops API or vice versa

## Memory Protocol

This project uses an observational memory system stored in the Claude Code auto-memory directory.

**At the END of each session:**
1. Update `memory/session-log.md` with what was accomplished, what's in progress, and blockers
2. Update `memory/debugging-patterns.md` if a new recurring bug pattern was discovered
3. Update `memory/MEMORY.md` "Active Work" section with current branch and state

**At the START of each session:**
1. Read `memory/session-log.md` for the most recent entry
2. Read any topic file relevant to the current task (architecture.md, test-patterns.md, domain-rules.md)

**Memory files:**
- `memory/MEMORY.md` -- Index and critical rules (loaded into every session's system prompt)
- `memory/architecture.md` -- Key file locations and data model relationships
- `memory/debugging-patterns.md` -- Recurring bugs and their fixes
- `memory/test-patterns.md` -- Test setup conventions and common fixtures
- `memory/domain-rules.md` -- Business logic invariants
- `memory/session-log.md` -- Brief log of what each session accomplished

## Codex CLI Integration

This project uses a hybrid Claude + Codex workflow. Claude handles planning, orchestration, and domain logic. Codex handles token-heavy tasks via headless execution.

**Codex CLI** (v0.98.0): `codex exec` runs headless with structured output. The project is trusted in `~/.codex/config.toml` (model: gpt-5.2-codex).

**When to delegate to Codex:**
- Code review before commits: `/review` (uses `codex exec review`)
- Test running and fixing: `/codex-test` (uses `codex exec --full-auto`)
- Large output analysis (backtest results, logs): `/codex-analyze` (uses `codex exec --output-schema`)

**When to keep in Claude:**
- Planning and architecture decisions
- Domain logic reasoning (trading rules, risk invariants)
- Session continuity and memory management
- Multi-step orchestration across files

**Rules for Codex invocation:**
- Always use `codex exec` (never interactive `codex`) -- Claude's bash environment is non-terminal
- Always suppress stderr: `2>/dev/null` (avoids thinking token noise in Claude's context)
- Use `-o /tmp/codex-<task>.txt` or `-o /tmp/codex-<task>.json` to capture output reliably
- Structured output schemas live in `.claude/codex-schemas/`
- Always review Codex's file changes (`git diff`) before accepting them

## Repository Reference

**Planning & Architecture:**
- `docs/UI_UNIFICATION_PLAN.md` - Comprehensive guide for building unified backtest/live trading dashboard (300+ lines, implementation-ready)
- `docs/SLOP_AUDIT.md` - Critical analysis of current repository issues and prioritized fix list
- `docs/ARCHITECTURE.md` - Detailed system design documentation
- `AGENTS.md` - Agent operating principles (to be moved to `docs/`)

**Operational Docs:**
- `docs/cap_risk_playbook.md` - Reconciliation playbooks
- `docs/STATUS.md` - Current project status
- `docs/backlog.md` - Outstanding work items

**Note**: The repository is at ~60% maturity with significant UI/monitoring infrastructure gaps identified. See `SLOP_AUDIT.md` for critical issues and `UI_UNIFICATION_PLAN.md` for comprehensive implementation roadmap.

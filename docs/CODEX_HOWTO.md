# Codex How-To Guide for crypto-trading-agents

This guide helps Codex work effectively on this codebase when Claude is unavailable.

## Critical Rules

### 1. Never "Harden" to Fix Bugs

**Bad pattern** (masking the real issue):
```python
# BAD - Adding try/except to silence errors
try:
    result = some_operation()
except Exception:
    result = None  # Silently fails
```

**Good pattern** (fix the root cause):
```python
# GOOD - Validate inputs, handle expected cases
if not valid_input:
    raise ValueError("Input must be provided")
result = some_operation()
```

### 2. Order of Operations Matters

When loading data, always check for empty/invalid data BEFORE processing:

```python
# CORRECT order
raw_data = load_data()
if raw_data.empty:
    raise ValueError("No data available")
features = compute_features(raw_data)  # Safe - data exists

# WRONG order (will crash on empty data)
raw_data = load_data()
features = compute_features(raw_data)  # May crash here!
if raw_data.empty:  # Too late - already crashed above
    raise ValueError("No data available")
```

### 3. Test Full Stack Before Declaring Fixed

Always verify fixes with:

```bash
# Run all tests
uv run pytest

# Run specific module tests
uv run pytest tests/test_<module>.py -v

# Check for import errors
uv run python -c "from module import function"
```

### 4. Check Downstream Consumers

Before making a field optional or changing a type:

```bash
# Find all usages
grep -r "field_name" --include="*.py" .

# Check if any code assumes non-None
grep -r "\.field_name\." --include="*.py" .
```

## Architecture Quick Reference

### Two Servers, Two Purposes

| Port | Server | Purpose | Use When |
|------|--------|---------|----------|
| 8080 | MCP Server | Agent tools | Adding trading operations |
| 8081 | Ops API | Dashboards | Adding UI/monitoring features |

### Dual Ledger System

- **Mock Ledger**: Temporal workflow state (default for agents)
- **Production Ledger**: PostgreSQL + Coinbase (requires `ENABLE_REAL_LEDGER=1`)

Agents use mock ledger by default. Don't mix them up.

### Temporal Workflows

Workflows MUST be deterministic:
- NO direct I/O (HTTP calls, file reads)
- NO `random()` or `time.time()`
- Use activities for non-deterministic operations

```python
# BAD - Non-deterministic in workflow
@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self):
        data = requests.get(url)  # WRONG - direct I/O

# GOOD - Use activity for I/O
@activity.defn
async def fetch_data(url: str):
    return requests.get(url).json()

@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self):
        data = await workflow.execute_activity(fetch_data, url)
```

## Testing Checklist

Before submitting changes:

1. **Unit tests pass**: `uv run pytest`
2. **No import errors**: `uv run python -c "import mcp_server.app"`
3. **Type hints valid**: `uv run pyright` (if available)
4. **Docker builds**: `docker compose build`

### Known Test Collection Issues

Some tests require `OPENAI_API_KEY` to collect. If you see collection errors mentioning "LLM client", either:
- Set a dummy key: `OPENAI_API_KEY=test uv run pytest tests/test_specific.py`
- Skip those tests and run unaffected ones

## Common Patterns

### Adding a New Backtest Feature

1. Update `backtesting/simulator.py` - core logic
2. Update `backtesting/activities.py` - Temporal activity wrapper
3. Update `ops_api/routers/backtests.py` - API endpoint
4. Update `ui/src/components/BacktestControl.tsx` - UI
5. Update `ui/src/lib/api.ts` - TypeScript types

### Adding a Configuration Option

1. Add to `.env.example` with default
2. Add to `app/core/config.py` Settings class
3. Use via `get_settings().new_option`
4. Document in `CLAUDE.md`

### Modifying Coinbase Integration

The Coinbase client (`app/coinbase/client.py`) handles real money. Extra care:

1. Never remove validation checks
2. Always preserve idempotency key handling
3. Test with mock responses first
4. Real trading requires explicit `ENABLE_REAL_LEDGER=1`

## Debugging Tips

### Step 1: Check Service Status

Always start by checking which services are running:

```bash
# Check all container status
docker compose ps

# Look for:
# - STATUS: "Up" vs "Restarting" vs "Exit"
# - HEALTH: "(healthy)" vs "(unhealthy)" vs "(health: starting)"
```

### Step 2: Check Container Logs

```bash
# Check specific service logs
docker compose logs --tail=100 worker
docker compose logs --tail=100 ops-api

# Follow logs in real-time
docker compose logs -f worker ops-api

# Search for specific errors
docker compose logs worker 2>&1 | grep -i "error\|exception\|failed"
```

### Step 3: Check Environment Variables

```bash
# Check what env vars a container sees
docker compose exec worker env | grep -E "(TEMPORAL|TASK_QUEUE|DB)"

# Compare with .env file
grep -E "^TASK_QUEUE" .env
```

### Step 4: Test Network Connectivity

```bash
# Test DNS resolution from container
docker compose exec ops-api python -c "import socket; print(socket.gethostbyname('temporal'))"

# Test if service is reachable
docker compose exec ops-api curl -s http://temporal:7233 || echo "Not reachable"
```

### Step 5: View Temporal Workflow State

```bash
# Temporal UI at http://localhost:8088
docker compose up temporal-ui

# Check if namespace is registered
docker compose exec temporal tctl namespace list
```

### Step 6: Check Workflow Logs

```python
# Query workflow state programmatically
handle = client.get_workflow_handle("workflow-id")
status = await handle.query(Workflow.get_status)
```

### Step 7: Database State

```bash
# Connect to PostgreSQL
docker compose exec db psql -U botuser -d botdb

# Run migrations
docker compose exec ops-api uv run alembic upgrade head
```

## Common Issues and Fixes

### Issue: Worker keeps restarting with DNS error

**Symptom:**
```
RuntimeError: Failed client connect: Server connection error:
tonic::transport::Error(Transport, ConnectError("dns error"...))
```

**Diagnosis:**
```bash
# Check if Python can resolve DNS (usually works)
docker compose exec worker python -c "import socket; print(socket.gethostbyname('temporal'))"

# Check Temporal SDK (may fail due to Rust DNS issues)
docker compose logs worker | grep "dns error"
```

**Fix:** Add DNS config to docker-compose.yml:
```yaml
worker:
  dns:
    - 127.0.0.11  # Docker's embedded DNS
  command: sh -lc "sleep 5 && uv run python -m worker.main"
```

### Issue: Backtest stuck at "queued" or not found

**Symptom:** UI shows backtest created but status never updates

**Diagnosis:**
```bash
# Check if workflow was started
# Look in Temporal UI at http://localhost:8088 for workflow ID

# Check task queue mismatch
grep "task_queue" ops_api/routers/backtests.py  # What API uses
grep "^TASK_QUEUE" .env                          # What worker uses
```

**Fix:** Ensure task queues match:
```python
# In ops_api/routers/backtests.py
BACKTEST_TASK_QUEUE = os.environ.get("TASK_QUEUE", "mcp-tools")
# Use BACKTEST_TASK_QUEUE instead of hardcoded value
```

### Issue: API returns 500 with "equity_curve" error

**Symptom:** Backtest completes but `/backtests/{id}/equity` fails

**Diagnosis:**
```bash
# Check what format is persisted
ls -la .cache/backtests/
python -c "import pickle; print(pickle.load(open('.cache/backtests/backtest-xxx.pkl', 'rb')).keys())"
```

**Root Cause:** Data format mismatch between workflow persistence and API expectations

**Fix:** Update API to handle multiple data formats (see `ops_api/routers/backtests.py`)

### Issue: Services marked as unhealthy

**Symptom:** `docker compose ps` shows `(unhealthy)`

**Diagnosis:**
```bash
# Check health check command
docker inspect crypto-trading-agents-ops-api-1 | grep -A 10 "Healthcheck"

# Test health endpoint manually
curl http://localhost:8081/health
```

**Fix:** Usually a startup timing issue. Restart the service:
```bash
docker compose restart ops-api
```

## Red Flags to Watch For

When reviewing changes, watch for:

1. **Silent exception handling**: `except: pass` or `except Exception: return None`
2. **Changed field requirements**: Required -> Optional without checking consumers
3. **Missing input validation**: Especially on financial calculations
4. **Wrong order of operations**: Process before validate
5. **Hardcoded values**: Should be config options
6. **Mixed ledger usage**: Mock vs production ledger confusion

## When to Escalate to Claude

Escalate complex issues to Claude when:

1. Multiple interacting systems need changes
2. Temporal workflow logic modifications
3. Financial calculation changes
4. Security-sensitive code
5. Architectural decisions
6. Test failures you can't diagnose

Document what you tried and the error messages for context.

## File Reference

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Main project documentation |
| `docs/ARCHITECTURE.md` | System design details |
| `docs/SLOP_AUDIT.md` | Known issues and priorities |
| `docs/UI_UNIFICATION_PLAN.md` | Dashboard implementation guide |
| `docs/FULL_STACK_WIRING_ISSUES.md` | Data flow diagrams and known wiring issues |
| `docs/LOCAL_DEV_QUICKSTART.md` | Quick start guide for local development |

## Debugging Method Summary

When debugging full-stack issues, follow this order:

1. **Check service status** (`docker compose ps`)
2. **Read container logs** (`docker compose logs --tail=100 <service>`)
3. **Verify environment variables** (`docker compose exec <service> env`)
4. **Test network connectivity** (DNS, ports)
5. **Check Temporal UI** (http://localhost:8088)
6. **Trace data flow** (UI → API → Workflow → Activity → Data Source)
7. **Check data format compatibility** (what's saved vs what's expected)

Always document what you tried and what error messages you saw before escalating.

---

*Generated for Codex reference. Last updated: 2026-01-09*

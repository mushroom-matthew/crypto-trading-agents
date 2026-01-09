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

### View Temporal Workflow State

```bash
# Temporal UI at http://localhost:8088
docker compose up temporal-ui
```

### Check Workflow Logs

```python
# Query workflow state
handle = client.get_workflow_handle("workflow-id")
status = await handle.query(Workflow.get_status)
```

### Database State

```bash
# Connect to PostgreSQL
docker compose exec db psql -U botuser -d ledger

# Run migrations
uv run alembic upgrade head
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

---

*Generated for Codex reference. Last updated: 2026-01-09*

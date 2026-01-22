# UI & Strategy Primitives Convergence Plan

**Created:** 2026-01-12
**Status:** Phase 1-2 Complete, Phase 3 Pending

## Problem Statement

There is significant divergence between the Backtest, Paper Trading, and (future) Live Trading UI components. Users should have a consistent experience across all three modes, particularly for:

1. Strategy template selection and editing
2. LLM insights and plan visibility
3. Portfolio/market visualization
4. Event timelines and logging

## Current State Analysis

### Feature Matrix

| Feature | Backtest | Paper Trading | Live Trading |
|---------|:--------:|:-------------:|:------------:|
| Strategy Template Selector | Yes | Yes | TBD |
| **Prompt Editor** (edit/version/history) | Yes | **Yes** (added) | TBD |
| **LLM Insights** (plans, costs, reasoning) | Yes | **API ready** | TBD |
| Real-time Portfolio Display | N/A | Yes | TBD |
| Trade History Table | Yes | Yes | TBD |
| **Candlestick/Price Chart** | Yes | Pending | TBD |
| **Event Timeline** | Yes | **Yes** (added) | TBD |
| Interactive Playback | Yes | N/A | N/A |
| Progress Indicator | Yes | Partial | TBD |
| **Equity Curve** | Yes | **API ready** | TBD |
| **Update Strategy Mid-Session** | N/A | **Yes** (added) | TBD |

### Key Missing Features in Paper Trading

1. **PromptEditor Component** - Users cannot edit strategy prompts for live sessions
2. **LLMInsights Component** - No visibility into LLM decision-making
3. **Price/Portfolio Charts** - No visualization of market data or equity curve
4. **EventTimeline** - Events are emitted but not displayed

### Backend Gaps

1. No API endpoint for paper trading plan history (only current plan)
2. No API endpoint for paper trading equity curve
3. No API for updating strategy prompt mid-session (signal exists but no REST endpoint)

---

## Convergence Plan

### Phase 1: Shared Components (UI)

#### 1.1 Add PromptEditor to Paper Trading
**Effort:** Low
**Files:** `ui/src/components/PaperTradingControl.tsx`

The `<PromptEditor />` component is already reusable. Add it to PaperTradingControl:

```tsx
// In PaperTradingControl.tsx
import { PromptEditor } from './PromptEditor';

// In JSX, after the configuration form:
<PromptEditor />
```

The PromptEditor edits `prompts/llm_strategist_prompt.txt` and `prompts/llm_judge_prompt.txt` which are used by all modes.

#### 1.2 Add EventTimeline to Paper Trading
**Effort:** Low
**Files:** `ui/src/components/PaperTradingControl.tsx`

```tsx
import { EventTimeline } from './EventTimeline';

// In JSX, after trade history:
<EventTimeline limit={30} runId={selectedSessionId || undefined} />
```

Paper trading already emits events via `emit_paper_trading_event_activity`. EventTimeline queries the same event store.

#### 1.3 Create PaperTradingInsights Component
**Effort:** Medium
**Files:** `ui/src/components/PaperTradingInsights.tsx` (new)

Create a component similar to `LLMInsights` that shows:
- Total plans generated
- LLM cost tracking (if available)
- Current market regime
- Trigger summary
- Plan history with timestamps

Requires new API endpoint (see Phase 2).

#### 1.4 Add Price Chart to Paper Trading
**Effort:** Medium
**Files:** `ui/src/components/PaperTradingControl.tsx`

Reuse `CandlestickChart` component with live data:

```tsx
import { CandlestickChart } from './CandlestickChart';

// Need to fetch candle data from market stream
// Or create simplified price line chart
```

May need a new API endpoint to fetch recent candles for paper trading symbols.

---

### Phase 2: API Extensions

#### 2.1 Paper Trading Plan History Endpoint
**Effort:** Medium
**Files:** `ops_api/routers/paper_trading.py`, `tools/paper_trading.py`

Add endpoint to retrieve full plan history:

```python
@router.get("/paper-trading/sessions/{session_id}/plans")
async def get_plan_history(session_id: str, limit: int = 20):
    """Get history of all strategy plans generated for this session."""
    handle = client.get_workflow_handle(session_id)
    plans = await handle.query(PaperTradingWorkflow.get_plan_history)
    return {"session_id": session_id, "plans": plans[-limit:]}
```

**Workflow Changes:**
- Add `plan_history: List[Dict]` to workflow state
- Add `get_plan_history` query
- Store each plan with timestamp, trigger count, cost estimate

#### 2.2 Paper Trading Equity Curve Endpoint
**Effort:** Medium
**Files:** `ops_api/routers/paper_trading.py`, `tools/paper_trading.py`

Add endpoint to retrieve equity over time:

```python
@router.get("/paper-trading/sessions/{session_id}/equity")
async def get_equity_curve(session_id: str):
    """Get equity curve for the session."""
    handle = client.get_workflow_handle(session_id)
    equity = await handle.query(PaperTradingWorkflow.get_equity_history)
    return equity
```

**Workflow Changes:**
- Track equity snapshots periodically (e.g., every 5 minutes)
- Store `equity_history: List[{timestamp, equity, cash, positions}]`

#### 2.3 Update Strategy Prompt Endpoint
**Effort:** Low
**Files:** `ops_api/routers/paper_trading.py`

Add REST endpoint to update strategy prompt mid-session:

```python
@router.put("/paper-trading/sessions/{session_id}/strategy")
async def update_strategy(session_id: str, request: UpdateStrategyRequest):
    """Update the strategy prompt for a running session."""
    handle = client.get_workflow_handle(session_id)
    await handle.signal(PaperTradingWorkflow.update_strategy_prompt, request.strategy_prompt)
    return {"session_id": session_id, "message": "Strategy updated, will take effect on next replan"}
```

#### 2.4 Market Candles Endpoint for Paper Trading
**Effort:** Low
**Files:** `ops_api/routers/paper_trading.py` or reuse existing market endpoints

Leverage existing `/market/candles` endpoint or create paper-trading-specific one that fetches from the ExecutionLedgerWorkflow's price history.

---

### Phase 3: Live Trading Alignment

When implementing Live Trading UI, ensure it includes:

1. **Same PromptEditor** - Edit live trading strategy prompts
2. **Same Insights Pattern** - Show LLM costs and reasoning
3. **Same Portfolio Display** - Consistent layout with paper trading
4. **Same Charts** - Candlestick and equity curves
5. **Same EventTimeline** - Unified event display

Additional live trading requirements:
- Risk controls visibility (tradeable fractions, cost gates)
- Real vs Paper indicator
- Coinbase sync status
- Reconciliation alerts

---

## Implementation Priority

### High Priority (Do First) - COMPLETE
1. ~~Add `<PromptEditor />` to PaperTradingControl~~ Done
2. ~~Add `<EventTimeline />` to PaperTradingControl~~ Done
3. ~~Add plan history query to PaperTradingWorkflow~~ Done

### Medium Priority - COMPLETE
4. ~~Create PaperTradingInsights component~~ (API ready, UI component pending)
5. ~~Add equity tracking to workflow~~ Done
6. ~~Add equity curve endpoint~~ Done

### Lower Priority - COMPLETE
7. Add candlestick chart to paper trading (pending UI work)
8. Market candles endpoint for paper trading (can use existing)
9. ~~Update strategy prompt endpoint~~ Done (PUT /sessions/{id}/strategy)

---

## Shared Primitives Inventory

After convergence, these components should be shared:

| Component | Used By | Notes |
|-----------|---------|-------|
| `PromptEditor` | Backtest, Paper, Live | Edit strategist/judge prompts |
| `EventTimeline` | Backtest, Paper, Live | Query by run_id/session_id |
| `CandlestickChart` | Backtest, Paper, Live | Needs data adapter per mode |
| `MarketTicker` | Backtest, Paper, Live | Already shared |
| `PortfolioDisplay` | Paper, Live | Extract from PaperTradingControl |
| `TradeHistoryTable` | Backtest, Paper, Live | Extract common component |
| `InsightsPanel` | Backtest, Paper, Live | Generalize LLMInsights |

---

## API Endpoint Alignment

### Strategy/Prompt Endpoints (All Modes)
- `GET /prompts/` - List available prompts
- `GET /prompts/{name}` - Get prompt content
- `PUT /prompts/{name}` - Update prompt
- `GET /prompts/strategies/` - List strategy templates
- `GET /prompts/strategies/{id}` - Get strategy template

### Session-Specific Endpoints

| Endpoint Pattern | Backtest | Paper Trading | Live Trading |
|-----------------|----------|---------------|--------------|
| `POST /sessions` | `/backtests` | `/paper-trading/sessions` | `/live/sessions` |
| `GET /sessions/{id}` | `/backtests/{id}` | `/paper-trading/sessions/{id}` | `/live/sessions/{id}` |
| `GET /sessions/{id}/portfolio` | `/backtests/{id}/results` | `/paper-trading/sessions/{id}/portfolio` | `/live/sessions/{id}/portfolio` |
| `GET /sessions/{id}/trades` | `/backtests/{id}/trades` | `/paper-trading/sessions/{id}/trades` | `/live/sessions/{id}/trades` |
| `GET /sessions/{id}/plans` | `/backtests/{id}/llm-insights` | `/paper-trading/sessions/{id}/plans` | `/live/sessions/{id}/plans` |
| `GET /sessions/{id}/equity` | `/backtests/{id}/equity` | `/paper-trading/sessions/{id}/equity` | `/live/sessions/{id}/equity` |

---

## Success Criteria

1. Users can edit strategy prompts from any mode (backtest/paper/live)
2. LLM reasoning and costs are visible in all modes
3. Event timeline shows in all modes
4. Portfolio display is consistent across modes
5. Same strategy templates available in all modes
6. Switching between modes feels like the same application

---

## Files to Modify

### UI Components
- `ui/src/components/PaperTradingControl.tsx` - Add PromptEditor, EventTimeline
- `ui/src/components/PaperTradingInsights.tsx` - New file
- `ui/src/lib/api.ts` - Add plan history, equity curve methods

### Backend
- `tools/paper_trading.py` - Add plan_history, equity_history tracking + queries
- `ops_api/routers/paper_trading.py` - Add /plans, /equity, PUT /strategy endpoints

### Shared
- Consider creating `ui/src/components/shared/` directory for truly shared components

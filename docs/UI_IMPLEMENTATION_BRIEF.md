# UI Implementation Brief for Future Agent

**Date**: 2026-01-02
**Status**: Ready for Implementation
**Estimated Effort**: 3-4 weeks full-time

## What You Need to Know

This repository needs a unified web dashboard for backtesting and live trading. Currently, the system has **three separate UIs** with no integration, **incomplete event wiring**, and **zero backtest visualization** despite having a sophisticated backtesting engine.

## Your Mission

Build a single React-based web application with 5 tabs:

1. **Backtest Control** - Start/monitor backtests, view equity curves and results
2. **Live Trading Monitor** - Real-time positions, fills, blocks, risk budget
3. **Market Monitor** - Multi-symbol price charts with real-time ticks
4. **Agent Inspector** - Decision chains, event logs, LLM telemetry
5. **Wallet Reconciliation** - Balance comparison, drift detection, manual corrections

## What to Read First

**Start here** (in order):

1. **`docs/SLOP_AUDIT.md`** (5 min) - Quick overview of the 10 major issues blocking UI development
2. **`docs/UI_UNIFICATION_PLAN.md`** (20 min) - Your complete implementation roadmap with code examples
3. **`CLAUDE.md`** (10 min) - Repository conventions and development commands

## Critical Path (First 3 Days)

### Day 1: Event Emission
**File**: `docs/UI_UNIFICATION_PLAN.md` Section "Phase 1.1"

Wire all agents to emit events consistently:
- Execution agent: `trade_blocked`, `order_submitted`, `fill`
- Broker agent: `intent`, `plan_generated`
- Judge agent: `plan_judged`

**Why this matters**: Without event emission, the UI cannot show decision chains or block reasons.

**Validation**:
```sql
SELECT type, COUNT(*) FROM events GROUP BY type;
```
Should show all event types with counts > 0.

### Day 2: Fix Materializer
**File**: `docs/UI_UNIFICATION_PLAN.md` Section "Phase 1.2"

Replace hardcoded `status="running"` and `mode="paper"` with actual Temporal queries and runtime config.

**Why this matters**: UI currently shows inaccurate workflow status.

**Validation**: Pause workflow → API shows "paused" instead of "running".

### Day 3: Add Database Tables
**File**: `docs/UI_UNIFICATION_PLAN.md` Section "Phase 1.3"

Create migration for:
- `BlockEvent` - Individual trade blocks with reasons
- `RiskAllocation` - Risk budget tracking
- `PositionSnapshot` - Live position state
- `BacktestRun` - Backtest metadata and results

**Why this matters**: Foundation for all tracking and queries.

**Validation**: `psql -d botdb -c "\dt"` shows new tables.

## Quick Start Commands

```bash
# 1. Review current state
cat docs/SLOP_AUDIT.md

# 2. Read full implementation plan
cat docs/UI_UNIFICATION_PLAN.md

# 3. Start Phase 1.1 (event emission)
# Edit: agents/execution_agent_client.py
# Add EventEmitter, emit trade_blocked/order_submitted/fill events

# 4. Test event emission
# Start agents, trigger a trade block, then:
sqlite3 ops_api/events.db "SELECT * FROM events WHERE type='trade_blocked';"

# 5. Continue with Phase 1.2, 1.3, etc.
```

## Expected Deliverables

**Week 1**: Foundation
- [ ] All agents emit events (validate with SQL query)
- [ ] Materializer queries Temporal (validate status accuracy)
- [ ] New DB tables created (validate with `\dt`)
- [ ] Initial API endpoints (`/backtests`, `/live`, `/events`)

**Week 2**: Frontend
- [ ] React app bootstrapped with Vite + TailwindCSS
- [ ] 5 tab layout with routing
- [ ] Backtest control tab (config form + results)
- [ ] Live monitor tab (position table + fills)

**Week 3**: Integration
- [ ] WebSocket streaming for real-time updates
- [ ] Backtest orchestration (start via web, poll status)
- [ ] Live daily reports (match backtest format)
- [ ] Wallet reconciliation UI

**Week 4**: Polish
- [ ] End-to-end testing
- [ ] Documentation updates
- [ ] Clean up repository (delete garbage files)
- [ ] Deprecate old UIs

## Success Criteria

You're done when:

1. ✅ User can start a backtest from web UI and see equity curve
2. ✅ Live fills appear in real-time table within 5 seconds
3. ✅ Block reasons show individual events (not just counts)
4. ✅ Agent decision chains visible (intent → decision → block/fill)
5. ✅ Wallet reconciliation runs from UI and shows drift history
6. ✅ No more hardcoded "running"/"paper" in materializer
7. ✅ Repository root cleaned (garbage files deleted)

## Common Pitfalls to Avoid

### Don't Do This
- ❌ Start building UI before fixing event emission (UI will have no data)
- ❌ Skip database migrations (API queries will fail)
- ❌ Build separate UIs for backtest and live (defeats the purpose)
- ❌ Use HTTP polling for real-time data (implement WebSocket)
- ❌ Ignore correlation IDs (decision chains will be broken)

### Do This Instead
- ✅ Follow the phase order in UI_UNIFICATION_PLAN.md
- ✅ Validate each step before moving to next
- ✅ Emit events at every state transition
- ✅ Use unified schemas for backtest and live trades
- ✅ Thread correlation IDs through all events

## Support Resources

**If stuck on**:
- Event emission → See `UI_UNIFICATION_PLAN.md` Section "Appendix: Code Snippets"
- API design → See `ops_api/app.py` existing endpoints as reference
- Frontend patterns → See example components in `UI_UNIFICATION_PLAN.md` Section "Phase 2.2"
- Database schema → See `app/db/models.py` for existing patterns
- Temporal integration → See `agents/workflows/` for workflow patterns

**Questions to ask user**:
- "Should I keep `ticker_ui_service.py` terminal UI or deprecate it?"
- "Do you want A/B backtest comparison in MVP or defer to V1.1?"
- "Should reconciliation be scheduled automatically or manual only?"

## Quick Reference: Files You'll Modify

**Backend**:
- `agents/execution_agent_client.py` - Add event emission
- `agents/broker_agent_client.py` - Add event emission
- `agents/judge_agent_client.py` - Add event emission
- `ops_api/materializer.py` - Fix hardcoded values
- `app/db/models.py` - Add new tables
- `ops_api/routers/` (new) - API endpoints

**Frontend** (new directory):
- `ui/dashboard/src/App.tsx` - Main layout with tabs
- `ui/dashboard/src/components/BacktestControl.tsx`
- `ui/dashboard/src/components/LiveTradingMonitor.tsx`
- `ui/dashboard/src/components/MarketMonitor.tsx`
- `ui/dashboard/src/components/AgentInspector.tsx`
- `ui/dashboard/src/components/WalletReconciliation.tsx`

## Final Checklist Before You Start

- [ ] I've read `docs/SLOP_AUDIT.md` (understand the problems)
- [ ] I've read `docs/UI_UNIFICATION_PLAN.md` (understand the solution)
- [ ] I've read `CLAUDE.md` (understand the codebase)
- [ ] I understand the critical path (event emission → materializer → DB tables → API → UI)
- [ ] I have a plan for validation at each step
- [ ] I'm ready to ask questions if anything is unclear

---

**You've got this!** The foundation is solid, the plan is detailed, and the path is clear. Focus on Phase 1 first (foundation), validate thoroughly, then move to Phase 2 (frontend). Good luck! 🚀

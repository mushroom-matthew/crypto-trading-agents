# Repository Issues Summary

## Overview

Analysis of backtest-4722e226 revealed three critical architectural issues in the trading system. Each has been documented with root cause analysis and proposed fixes.

---

## Issue Matrix

| Issue | Severity | Impact | Root Cause | Fix Priority |
|-------|----------|--------|------------|--------------|
| [Trigger Churn](./REPO_ISSUE_TRIGGER_CHURN.md) | HIGH | No learning across replans | Stateless LLM generation | P0 |
| [Emergency Exit Competition](./REPO_ISSUE_EMERGENCY_EXIT_COMPETITION.md) | CRITICAL | Positions exit immediately | Same-bar evaluation order | P0 |
| [Judge Disables All Categories](./REPO_ISSUE_JUDGE_CONSTRAINT_VALIDATION.md) | CRITICAL | Trading impossible | No validation | P0 |

---

## Quick Reference: Files to Modify

### Trigger Churn

Latest backtest audit (backtest-6d140887-acea-4a0c-93ad-3769573022ae):

- 5 LLM calls in a 1-day run; replans triggered at 07:05, 15:05, 23:05 by `judge_triggered` even when trigger_diff is unchanged (unchanged: 6).
- One intra-day replan swapped 2 triggers (`tc_long_2`/`tc_short_2` removed, `mr_short_1`/`vb_long_1` added) while the plan window stayed 2021-05-01 -> 2021-05-02, showing churn inside the same slot.
- `plan_log.generated_at` stays at 00:00 for all replans, masking actual replan time and making churn look like a single plan.

| File | Line | Change |
|------|------|--------|
| `schemas/llm_strategist.py` | 91-98 | Add `previous_triggers` to LLMInput |
| `prompts/llm_strategist_prompt.txt` | - | Add continuity instructions |
| `agents/strategies/plan_provider.py` | 159-214 | Pass previous plan |
| `agents/strategies/llm_client.py` | 184-192 | Include prior triggers |

### Emergency Exit Competition

Latest backtest audit (backtest-6d140887-acea-4a0c-93ad-3769573022ae):

- 4 round trips, 4/4 exits were `emergency_exit_1_flat`; average hold 36.25 minutes; one same-bar exit at 2021-05-01T07:00:00+00:00 (tc_short_2 entry and emergency_exit_1_flat exit same timestamp).
- Emergency exit trigger uses `direction: exit` with `entry_rule: "position != 'flat'"` and empty `exit_rule`, so it evaluates through the entry/flatten path. Same-bar exit block and min_hold enforcement are only in the exit-rule path, so they never trigger.
- `SAME_BAR_ENTRY` blocks are absent; block details show `min_flat: 61`, `MIN_HOLD_PERIOD: 24`, `SIGNAL_PRIORITY: 8` and `blocked_by_risk_limits: 93`. Emergency exit itself has 0 blocks.
- Prompt requirements for emergency buffers/volume confirmation are not reflected: emergency exit rule is tautological and mean-reversion trigger lacks `volume`/`volume_multiple` gating.

| File | Line | Change |
|------|------|--------|
| `agents/strategies/trigger_engine.py` | 303-385 | Block same-bar emergency exit |
| `agents/strategies/trigger_engine.py` | 348 | Separate min_hold for emergency |
| `backtesting/llm_strategist_runner.py` | 351 | Sync min_hold systems |

### Telemetry Gaps (backtest-6d140887-acea-4a0c-93ad-3769573022ae)

- Backtest results are only persisted inside the worker container at `/app/.cache/backtests/backtest-6d140887-acea-4a0c-93ad-3769573022ae.pkl`; host `.cache/backtests` and `backtest_runs` table are empty, making audits brittle. Mount a volume or persist results to Postgres.
- `block_events` table is empty; block reasons only appear inside daily report payloads. Persist block entries for backtests so we can query block churn and root causes.
- `plan_log.generated_at` does not capture actual generation time; add an explicit `generated_at_actual` (or similar) and include replan reason to make churn audits feasible.

### Judge Constraint Validation

| File | Line | Change |
|------|------|--------|
| `schemas/judge_feedback.py` | 22-32 | Add Pydantic validator |
| `prompts/llm_judge_prompt.txt` | 94 | Add constraint limits |
| `services/judge_feedback_service.py` | 258-269 | Guard heuristic generation |
| `trading_core/execution_engine.py` | 243-252 | Add fallback validation |

---

## Interaction Between Issues

```
┌─────────────────────────────────────────────────────────────────┐
│                    FAILURE CASCADE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Trigger Churn                                               │
│     └─ New triggers every replan                                │
│           └─ Can't learn which triggers work                    │
│                 └─ Random trigger quality                       │
│                                                                 │
│  2. Emergency Exit Competition                                  │
│     └─ Entries immediately closed                               │
│           └─ High emergency exit rate                           │
│                 └─ 0% win rate on surviving trades              │
│                                                                 │
│  3. Judge Over-Correction                                       │
│     └─ Sees poor performance                                    │
│           └─ Disables all categories                            │
│                 └─ System can't trade                           │
│                       └─ Judge sees 0 trades                    │
│                             └─ Score drops further              │
│                                   └─ LOOP                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**The issues compound**: Trigger churn prevents learning, emergency exits kill positions, judge panics and disables everything.

---

## Recommended Fix Order

### Phase 1: Stop the Bleeding (P0)

1. **Add schema validation for disabled_categories**
   - Prevents complete trading shutdown
   - 30 min implementation
   - File: `schemas/judge_feedback.py`

2. **Block emergency exit on same bar as entry**
   - Prevents immediate position closure
   - 1 hour implementation
   - File: `agents/strategies/trigger_engine.py`

3. **Add prompt instruction against disabling all categories**
   - Quick win via prompt engineering
   - 15 min implementation
   - File: `prompts/llm_judge_prompt.txt`

### Phase 2: Enable Learning (P1)

4. **Add previous triggers to LLMInput**
   - Enables trigger continuity
   - 2-3 hours implementation
   - Files: `schemas/llm_strategist.py`, `agents/strategies/plan_provider.py`

5. **Update strategist prompt for continuity**
   - Instructs LLM to preserve triggers
   - 30 min implementation
   - File: `prompts/llm_strategist_prompt.txt`

6. **Synchronize min_hold systems**
   - Consistent hold period enforcement
   - 1 hour implementation
   - Files: `trigger_engine.py`, `llm_strategist_runner.py`

### Phase 3: Harden the System (P2)

7. **Add trigger matching algorithm**
   - Deterministic trigger ID stability
   - 2-3 hours implementation

8. **Add emergency exit cooldown**
   - Prevent rapid emergency exit cycling
   - 1 hour implementation

9. **Add constraint warning events**
   - Observability for near-failures
   - 1 hour implementation

---

## Validation Test Plan

After implementing Phase 1 fixes, run:

```bash
# Same configuration as problematic backtest
uv run python -m backtesting.cli \
  --run-id validation-post-fix \
  --symbols BTC-USD \
  --start 2021-05-01 \
  --end 2021-05-03 \
  --plan-cadence 6
```

**Expected improvements**:
- [ ] At least one entry category always enabled
- [ ] Emergency exit rate < 30%
- [ ] Judge score never triggers "disabled all categories"
- [ ] Some trades execute successfully (win rate > 0%)

After implementing Phase 2 fixes:
- [ ] `unchanged > 0` in trigger diffs
- [ ] Trigger IDs stable across replans where logic unchanged
- [ ] Judge feedback references specific trigger performance

---

## Related Documentation

- [BACKTEST_4722e226_DEEP_DIVE.md](./BACKTEST_4722e226_DEEP_DIVE.md) - Original issue discovery
- [TRIGGER_SYSTEM_OVERVIEW.md](./TRIGGER_SYSTEM_OVERVIEW.md) - Trigger architecture reference
- [INTRADAY_JUDGE_CONTEXT_GAPS.md](./INTRADAY_JUDGE_CONTEXT_GAPS.md) - Judge context improvements (completed)

# Branch: comp-audit-risk-core

## Purpose
Implement Computation Audit Plan Phase 0 risk correctness and budget integrity changes (items 1-4). This is the highest priority branch.

## Source Plans
- docs/analysis/COMPUTATION_AUDIT_PLAN.md (Phase 0 items 1-4)

## Scope
- Remove or conditionalize the min trades/day floor so daily risk budget cannot be exceeded.
- Enforce aggregate risk on scale-in entries using blended entry and shared stop.
- Require a stop for shorts or apply a stricter notional cap when no stop is provided.
- Align simulator risk usage with stop-based sizing (risk_used_abs and actual_risk_at_stop).

## Out of Scope / Deferred
- Phase 1+ cadence/indicator/metrics/UI items.
- Prompt or strategy template changes.
- Phase 5 gap/slippage risk (item 17) unless explicitly pulled in later.

## Key Files
- agents/strategies/plan_provider.py
- services/strategist_plan_service.py
- agents/strategies/risk_engine.py
- agents/strategies/trade_risk.py
- backtesting/simulator.py

## Dependencies / Coordination
- Potential overlap on agents/strategies/plan_provider.py with comp-audit-indicators-prompts; coordinate to avoid conflicts.
- If any RiskEngine changes intersect with metrics-parity branch, align on function signatures before merging.

## Acceptance Criteria
- Derived trade cap never implies total risk above daily budget.
- Scale-in entry blocks when combined risk exceeds cap; risk snapshots include combined/incremental deltas.
- Shorts without stops are blocked or capped safely.
- Simulator risk_used_abs and actual_risk_at_stop reflect stop distance when present.

## Test Plan (required before commit)
Run the most specific tests available; if exact files do not exist, use the -k patterns.

- uv run pytest -k risk_engine -vv
- uv run pytest -k trade_risk -vv
- uv run pytest -k simulator -vv
- uv run python -c "from agents.strategies import risk_engine; from backtesting import simulator"

Do not commit until test output is recorded in the Test Evidence section below. If tests cannot be run locally, obtain user-run output and paste it here before committing.

## Human Verification (required)
- Run a targeted backtest that forces: (a) daily budget near per-trade risk, (b) scale-in entries with a shared stop, (c) a short without an explicit stop.
- Confirm in logs or results: trade cap never implies total risk above budget, scale-in blocked when combined risk exceeds cap, short without stop is blocked or capped.
- Paste run id and observations in the Human Verification Evidence section.

## Worktree Setup (recommended for parallel agents)
Use a linked worktree so multiple branches can be worked on in parallel from one clone.

```bash
# From the main repo directory
git fetch
git worktree add -b comp-audit-risk-core ../wt-comp-audit-risk-core comp-audit-risk-core
cd ../wt-comp-audit-risk-core

# When finished (after merge)
git worktree remove ../wt-comp-audit-risk-core
```

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b comp-audit-risk-core

# Work, then review changes
git status
git diff

# Stage changes
git add agents/strategies/plan_provider.py \
  services/strategist_plan_service.py \
  agents/strategies/risk_engine.py \
  agents/strategies/trade_risk.py \
  backtesting/simulator.py

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest -k risk_engine -vv
uv run pytest -k trade_risk -vv
uv run pytest -k simulator -vv
uv run python -c "from agents.strategies import risk_engine; from backtesting import simulator"

# Commit ONLY after test evidence is captured below
git commit -m "Risk: enforce daily budget and aggregate stop risk"
```

## Change Log (update during implementation)
- 2026-01-22: Reworked derived trade cap math to avoid budget overshoot with an optional legacy floor; added scale-in aggregate risk/short-stop guardrails and expanded risk snapshots; updated simulator risk_used_abs/actual_risk_at_stop to use stop-distance proxies. Files: agents/strategies/plan_provider.py, services/strategist_plan_service.py, agents/strategies/risk_engine.py, agents/strategies/trade_risk.py, backtesting/simulator.py.
- 2026-01-22: Added targeted backtest prompt file and logged follow-up observations. Files: docs/branching/comp-audit-risk-core-prompt.txt, docs/branching/X-comp-audit-risk-core.md.
- 2026-01-22: Added follow-up branch doc for out-of-scope items (budget reset failure, LLM short stop + scale-in alignment). File: docs/branching/later/_comp-audit-risk-followups.md.

## Notes / Follow-Ups (out of scope)
- Strategist prompt/schema does not require `stop_loss_pct` on triggers, so shorts can omit stops unless the prompt is tightened.
- TriggerEngine skips entry if the desired direction matches the current position, which prevents scale-ins in the LLM trigger path; risk aggregation primarily protects other entry paths.
- TriggerEngine defaults (`prioritize_by_confidence=True`, `max_triggers_per_symbol_per_bar=1`) also limit same-bar multi-entry behavior.

## Test Evidence (append results before commit)
```
uv run pytest -k risk_engine -vv
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0 -- /home/getzinmw/crypto-trading-agents/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/getzinmw/crypto-trading-agents
configfile: pyproject.toml
plugins: anyio-4.11.0, respx-0.22.0, asyncio-1.2.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 0 items                                                             

=============================== warnings summary ===============================
.venv/lib/python3.13/site-packages/websockets/legacy/__init__.py:6
  /home/getzinmw/crypto-trading-agents/.venv/lib/python3.13/site-packages/websockets/legacy/__init__.py:6: DeprecationWarning: websockets.legacy is deprecated; see https://websockets.readthedocs.io/en/stable/howto/upgrade.html for upgrade instructions
    warnings.warn(  # deprecated in 14.0 - 2024-11-09

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================ no tests ran in 0.02s ============================
```

```
uv run pytest -k trade_risk -vv
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0 -- /home/getzinmw/crypto-trading-agents/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/getzinmw/crypto-trading-agents
configfile: pyproject.toml
plugins: anyio-4.11.0, respx-0.22.0, asyncio-1.2.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 0 items                                                             

=============================== warnings summary ===============================
.venv/lib/python3.13/site-packages/websockets/legacy/__init__.py:6
  /home/getzinmw/crypto-trading-agents/.venv/lib/python3.13/site-packages/websockets/legacy/__init__.py:6: DeprecationWarning: websockets.legacy is deprecated; see https://websockets.readthedocs.io/en/stable/howto/upgrade.html for upgrade instructions
    warnings.warn(  # deprecated in 14.0 - 2024-11-09

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================ no tests ran in 0.02s ============================
```

```
OPENAI_API_KEY=test uv run pytest tests/risk -vv
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0 -- /home/getzinmw/crypto-trading-agents/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/getzinmw/crypto-trading-agents
configfile: pyproject.toml
plugins: anyio-4.11.0, respx-0.22.0, asyncio-1.2.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 12 items                                                            

tests/risk/test_budget_base_equity.py::test_budget_abs_uses_start_of_day_equity PASSED [  8%]
tests/risk/test_budget_base_equity.py::test_first_trade_starts_at_zero_usage PASSED [ 16%]
tests/risk/test_daily_budget_reset.py::test_day_two_not_blocked_by_day_one_usage FAILED [ 25%]
tests/risk/test_daily_budget_reset.py::test_day_usage_blocks_when_budget_exhausted_same_day FAILED [ 33%]
tests/risk/test_daily_budget_reset.py::test_used_pct_monotone_and_bounded PASSED [ 41%]
tests/risk/test_daily_loss_anchor.py::test_anchor_not_reset_intraday PASSED [ 50%]
tests/risk/test_daily_loss_anchor.py::test_anchor_resets_on_new_day PASSED [ 58%]
tests/risk/test_exit_bypass.py::test_exit_does_nothing_when_flat PASSED [ 66%]
tests/risk/test_exit_bypass.py::test_exit_flattens_when_position_exists PASSED [ 75%]
tests/risk/test_risk_at_stop.py::test_stop_distance_expands_notional_to_match_risk_cap PASSED [ 83%]
tests/risk/test_risk_at_stop.py::test_no_stop_uses_notional_cap PASSED  [ 91%]
tests/risk/test_risk_at_stop.py::test_tighter_stop_allows_larger_size PASSED [100%]

================================== FAILURES ===================================
__________________ test_day_two_not_blocked_by_day_one_usage __________________

    def test_day_two_not_blocked_by_day_one_usage() -> None:
        bt = _bt_stub()
        # Day 1 fully consumed (should not affect Day 2).
        bt.daily_risk_budget_state["2021-01-01"] = {
            "budget_abs": 100.0,
            "used_abs": 100.0,
            "symbol_usage": defaultdict(float),
            "blocks": defaultdict(int),
        }
        # Day 2 fresh budget.
        bt.daily_risk_budget_state["2021-01-02"] = {
            "budget_abs": 100.0,
            "used_abs": 0.0,
            "symbol_usage": defaultdict(float),
            "blocks": defaultdict(int),
        }
>       allowance = bt._risk_budget_allowance("2021-01-02", _Order())
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/risk/test_daily_budget_reset.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <backtesting.llm_strategist_runner.LLMStrategistBacktester object at 0x7551586a0410>
day_key = '2021-01-02'
order = <test_daily_budget_reset._Order object at 0x75513b5a74d0>

    def _risk_budget_allowance(self, day_key: str, order: Order) -> float | None:
        pct = self.daily_risk_budget_pct or 0.0
        if pct <= 0:
            return 0.0
        entry = self.daily_risk_budget_state.get(day_key)
        if not entry:
            return 0.0
        budget = entry.get("budget_abs", 0.0)
        if budget <= 0:
            return None
        notional = max(order.quantity * order.price, 0.0)
        if notional <= 0:
            return 0.0
        target_risk_pct = None
        if hasattr(self, "sizing_targets"):
            target_risk_pct = self.sizing_targets.get(order.symbol)
        if target_risk_pct is None:
            target_risk_pct = self.active_risk_limits.max_position_risk_pct or 0.0
        # Adaptive boost: based on same-day usage only (no cross-day carry-over).
        prev_usage = (entry.get("used_abs", 0.0) / budget * 100.0) if budget > 0 else 100.0
        adaptive_multiplier = 3.0 if prev_usage < 10.0 else 1.0
        # Per-trade risk allowance expressed in currency, not double-scaled by notional.
>       base_equity = entry.get("start_equity", self.initial_cash)
                                                ^^^^^^^^^^^^^^^^^
E       AttributeError: 'LLMStrategistBacktester' object has no attribute 'initial_cash'

backtesting/llm_strategist_runner.py:2865: AttributeError
____________ test_day_usage_blocks_when_budget_exhausted_same_day _____________

    def test_day_usage_blocks_when_budget_exhausted_same_day() -> None:
        bt = _bt_stub()
        bt.daily_risk_budget_state["2021-01-02"] = {
            "budget_abs": 50.0,
            "used_abs": 50.0,
            "symbol_usage": defaultdict(float),
            "blocks": defaultdict(int),
        }
>       allowance = bt._risk_budget_allowance("2021-01-02", _Order())
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/risk/test_daily_budget_reset.py:60: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <backtesting.llm_strategist_runner.LLMStrategistBacktester object at 0x75513b3fc050>
day_key = '2021-01-02'
order = <test_daily_budget_reset._Order object at 0x75513b537610>

    def _risk_budget_allowance(self, day_key: str, order: Order) -> float | None:
        pct = self.daily_risk_budget_pct or 0.0
        if pct <= 0:
            return 0.0
        entry = self.daily_risk_budget_state.get(day_key)
        if not entry:
            return 0.0
        budget = entry.get("budget_abs", 0.0)
        if budget <= 0:
            return None
        notional = max(order.quantity * order.price, 0.0)
        if notional <= 0:
            return 0.0
        target_risk_pct = None
        if hasattr(self, "sizing_targets"):
            target_risk_pct = self.sizing_targets.get(order.symbol)
        if target_risk_pct is None:
            target_risk_pct = self.active_risk_limits.max_position_risk_pct or 0.0
        # Adaptive boost: based on same-day usage only (no cross-day carry-over).
        prev_usage = (entry.get("used_abs", 0.0) / budget * 100.0) if budget > 0 else 100.0
        adaptive_multiplier = 3.0 if prev_usage < 10.0 else 1.0
        # Per-trade risk allowance expressed in currency, not double-scaled by notional.
>       base_equity = entry.get("start_equity", self.initial_cash)
                                                ^^^^^^^^^^^^^^^^^
E       AttributeError: 'LLMStrategistBacktester' object has no attribute 'initial_cash'

backtesting/llm_strategist_runner.py:2865: AttributeError
============================== warnings summary ===============================
tests/risk/test_budget_base_equity.py::test_first_trade_starts_at_zero_usage
  /home/getzinmw/crypto-trading-agents/tests/risk/test_budget_base_equity.py:45: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    Order(symbol="BTC", side="buy", quantity=1.0, price=10.0, timeframe="1h", reason="t", timestamp=datetime.utcnow()),

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/risk/test_daily_budget_reset.py::test_day_two_not_blocked_by_day_one_usage - AttributeError: 'LLMStrategistBacktester' object has no attribute 'initial_cash'
FAILED tests/risk/test_daily_budget_reset.py::test_day_usage_blocks_when_budget_exhausted_same_day - AttributeError: 'LLMStrategistBacktester' object has no attribute 'initial_cash'
=================== 2 failed, 10 passed, 1 warning in 2.48s ===================
```
User approved proceeding with the two failing tests above (out of scope: backtesting/llm_strategist_runner.py).

```
OPENAI_API_KEY=test uv run pytest -k simulator -vv
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0 -- /home/getzinmw/crypto-trading-agents/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/getzinmw/crypto-trading-agents
configfile: pyproject.toml
plugins: anyio-4.11.0, respx-0.22.0, asyncio-1.2.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 227 items / 227 deselected / 0 selected                             

============================== warnings summary ===============================
.venv/lib/python3.13/site-packages/websockets/legacy/__init__.py:6
  /home/getzinmw/crypto-trading-agents/.venv/lib/python3.13/site-packages/websockets/legacy/__init__.py:6: DeprecationWarning: websockets.legacy is deprecated; see https://websockets.readthedocs.io/en/stable/howto/upgrade.html for upgrade instructions
    warnings.warn(  # deprecated in 14.0 - 2024-11-09

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
===================== 227 deselected, 1 warning in 5.45s ======================
```

```
OPENAI_API_KEY=test uv run python -c "from agents.strategies import risk_engine; from backtesting import simulator"
warning: The `tool.uv.dev-dependencies` field (used in `pyproject.toml`) is deprecated and will be removed in a future release; use `dependency-groups.dev` instead
```

## Human Verification Evidence (append results before commit when required)
- Run id: comp-audit-risk-core-verify
- Command: `uv run python -m backtesting.cli --pair BTC-USD --start 2024-01-01 --end 2024-01-03 --llm-strategist enabled --llm-run-id comp-audit-risk-core-verify --llm-calls-per-day 1 --llm-cache-dir .cache/strategy_plans --llm-prompt docs/branching/comp-audit-risk-core-prompt.txt --timeframes 1h --max-position-risk-pct 0.9 --max-daily-risk-budget-pct 1.0 --debug-limits verbose --debug-output-dir .debug/backtests`
- Observations: derived cap resolved to 1 trade/day (daily_cap blocks in logs); triggers trimmed to a single long trigger (long_primary). Short/no-stop and scale-in trigger did not execute (not present after budget trim), so short-stop and scale-in aggregation behavior were not exercised in this run. Logs show daily_cap blocks and trigger stats saved to `.debug/backtests/comp-audit-risk-core-verify/trigger_stats.csv`.

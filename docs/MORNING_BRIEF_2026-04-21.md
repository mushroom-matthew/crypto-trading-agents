# Morning Brief — 2026-04-21

Session observations from first Fly.io preflight. Organized by priority.

---

## Fixes Shipped This Session

- **`query_ledger_portfolio_activity` timeout** bumped from 30s → 2min at all 4 callsites (`tools/paper_trading.py`). This was the direct cause of session `9812931d` dying with `TIMEOUT_TYPE_SCHEDULE_TO_CLOSE`. The activity creates a fresh Temporal client connection on every call; 30s was too tight over internet-routed Fly.io → VPS.
- **Terminate button** added to the session action bar. Calls `POST /ops-api/paper-trading/sessions/{id}/terminate`, which hard-kills the session workflow AND its orphaned ledger workflow. Appears in two places:
  - Alongside Stop when session is `running` (dark red trash icon, requires confirm dialog)
  - Standalone next to Start when session is `failed`, `timed_out`, or `cancelled` (ledger cleanup path)

---

## Immediate Operational Items

### 1. Upgrade Fly.io machine size (do first)
`shared-cpu-1x` / 1GB is too small for two concurrent paper trading sessions. Each session runs LLM calls (10–30s each), websocket streams, and Temporal polling. Running two sessions depleted the burstable CPU budget, causing silent LLM timeouts that produced the null-strategy loop and the eventual 18-trigger burst.

```
fly machine update --vm-size performance-cpu-1x
# or at minimum:
fly machine update --vm-size shared-cpu-2x --vm-memory 2048
```

### 2. CSRF error is Temporal UI (not our API)
The "missing csrf token" error occurs when cancelling/terminating via Temporal's own web UI (port 8088). Temporal UI v2+ has CSRF protection on mutating actions. Use our Terminate button instead — it goes through our ops API and bypasses Temporal UI entirely.

### 3. Session `774b71a4` null-strategy loop
Likely cause: LLM activity timeouts on a cold/busy worker. The plan generation activity has various timeouts that may also be too tight for Fly.io + VPS latency. Candidate fix: audit `generate_strategy_plan_activity` timeout (currently ~2-5min in most paths) and check if there are retries configured. The 18-trigger burst after silence confirms calls were queuing and then firing together once the worker caught up.

---

## Product Direction: UX Priorities

### 4. Feature-flag non-paper-trading UI
All features that aren't paper trading (backtesting, live trading, multi-wallet, live fills monitor) should be hidden behind a `VITE_FEATURES` env var. Default: paper trading only. This reduces cognitive load and noise for the current phase.

Implementation: `ui/src/lib/features.ts` — read `import.meta.env.VITE_FEATURES` (comma-separated list), export `isEnabled(flag: string): boolean`. Gate route visibility and nav links. No behavior change, just visibility.

### 5. Core session control taxonomy
Two tiers of controls to design around:

**Set-and-forget (session config — set before start):**
- Symbols, timeframe, plan interval, initial cash, direction bias
- Trailing stop mode + params
- Risk limits (max position %, max daily loss %)
- AI planner on/off, symbol discovery on/off

**Interactive (mid-session — can change while running):**
- Inject hypothesis (plain text → AI parses into trigger conditions)
- Override timeframe for next plan cycle
- Pause/resume trigger evaluation
- Manually suppress or promote a specific trigger
- Force replan (already exists)

---

## Strategy Layer: Key Gaps

### 6. Trade hypothesis validity window (highest priority strategy item)
Currently missing. A mean reversion trigger should carry `expected_resolution_bars: int` — the window within which the setup is confirmed or invalidated. After that many bars with no resolution, the trigger should auto-expire regardless of price action.

This matters because:
- Session `9812931d` had two correlated mean-reversion longs (BTC + ETH, 1h) with no expiry — both could persist indefinitely.
- Concurrent long positions on correlated assets under the same strategy hypothesis compounds risk without the user realizing it.

Implementation sketch:
- Add `hypothesis_expiry_bars: int | None` to `TriggerCondition` (None = no expiry)
- LLM prompt asks it to estimate resolution bars based on timeframe + strategy type
- Trigger engine checks `bars_since_armed >= hypothesis_expiry_bars` → emit `hypothesis_expired` event → move trigger to inactive
- Surface remaining bars as a countdown in the trigger card UI

### 7. Trigger card: show values, not just flags
Trigger cards currently show `target: true` but not the actual price. The computed stop/target prices exist in `TriggerCondition` — they just aren't rendered.

UI change: show stop price, target price, and implied R:R ratio in the trigger card. Color-code (green target, red stop). Eventually cross-reference onto the OHLCV chart as horizontal reference lines.

### 8. Trigger portfolio / library concept
The current model generates a fresh set of triggers each plan cycle. A better model: triggers are named, versioned strategies that accumulate performance attribution over time. The system maintains a library (this is the playbook system, already partially built) and selects from it based on current regime.

Key addition needed: **trigger-level outcome tracking** — which named trigger types win/lose in which regimes. This is the training data flywheel for future fine-tuning.

Near-term: persist `trigger_name` (e.g., `mean_reversion_1h_btc`) on fills so outcomes can be attributed back to the trigger class.

### 9. User-defined triggers / hypothesis injection
User sees a 5m structure they want to trade but the AI generated a 1h mean-reversion plan. They want to override or augment.

Two modes:
- **Hypothesis text → AI triggers**: user types "I think BTC will bounce off 94,200 support with a target of 95,800, stop below 93,900" → AI parses into a `TriggerCondition` set and injects it into the active plan
- **Timeframe switch**: dropdown changes `indicator_timeframe` for the next plan cycle without restarting the session

The hypothesis injection endpoint can be a signal to the workflow: `inject_user_hypothesis(text: str)` → plan provider runs a lightweight parse call → appends to trigger set.

### 10. Strategist input size reduction
The current LLM input for plan generation is very large (raw OHLCV history, full indicator snapshots, full portfolio state, memory bundles, etc.). This causes slow calls and high token costs.

Proposed two-phase approach:
- **Phase 1**: Give strategist a compact state summary (regime, recent indicators summary, portfolio summary, 3 most recent candles only). Ask it to state what additional context it needs.
- **Phase 2**: Fill only what it asked for (specific timeframe OHLCV, specific indicator history, specific episode memory).

This mirrors RAG-with-tool-use. Estimated token reduction: 60-70% on typical calls. Side effect: faster calls, fewer timeouts, more predictable latency.

Shorter-term quick win (no architecture change): strip raw OHLCV from the plan prompt entirely (it's already summarized in indicators), cap indicator history to 3 candles, cap episode memory bundle to 5 records.

---

## Experiments Queued (from previous session)

From session 48 backlog — still pending:
1. AI planner: does it correctly select symbols cycle 1? Does plan_interval auto-adjust? What happens in flat regime?
2. Trailing stops: ATR vs breakeven_only vs none — backtest comparison on max drawdown + win rate
3. Structural targets: what % of fills get structural override vs arithmetic? Better R:R?
4. Episode memory: does STRATEGY_KNOWLEDGE block change meaningfully after 50+ episodes?
5. Cadence governor: count PolicyLoopSkipEvent in 4hr session — >50% skipped = working, <10% = broken
6. Judge gate: does it ever reject a plan in a real session? What triggers it?

---

## Suggested Session Order

1. Upgrade Fly.io machine size (5 min, CLI only)
2. Deploy the timeout + terminate button fix (already committed to `aws-deploy`)
3. Run a single paper trading session for 30min to confirm no timeout deaths
4. Implement feature flagging (#4) — quick cleanup, high polish value
5. Add trigger card values (#7) — low effort, high observability value
6. Add `hypothesis_expiry_bars` (#6) — schema + trigger engine change, medium effort
7. Quick-win strategist input pruning (#10 quick win) — strip OHLCV from prompt

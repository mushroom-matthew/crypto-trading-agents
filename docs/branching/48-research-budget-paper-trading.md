# Runbook 48: Research Budget in Paper Trading

## Purpose

The backtest runner already has a learning-book / experiment-spec framework (Runbooks
9–12). Paper trading has none. Hypotheses validated in backtests can only be re-verified
in live data by shipping the strategy — there is no intermediate "test with real market
data, isolated capital, limited downside" layer.

This runbook adds a **research budget** to paper trading sessions: a separate capital
pool, separate ledger, and explicit hypothesis-to-trade-to-evidence loop. Research
trades explicitly test whether a specific playbook pattern holds in current market
conditions. Successful experiments (N validated trades, primary metric met) automatically
update the `## Validation Evidence` section in the relevant `vector_store/playbooks/*.md`
file. The judge gains two new action types to drive the research loop when live trades
flounder.

### Why this matters

The existing playbooks (`bollinger_squeeze`, `donchian_breakout`, etc.) contain priors
— rules of thumb written without live-data evidence. The research budget creates a
feedback loop that either confirms or refutes each prior, and the judge can weaponize
that evidence to update the strategy. Without this loop, the playbooks are permanently
opinions. With it, they become evidence-graded knowledge.

### Relationship to existing schemas

- `ExperimentSpec` (Runbook 11) already has `hypothesis`, `status`, `exposure`, and
  `metrics` — reused as-is.
- `SignalEvent` (Runbook 43) already has `run_id`, `symbol`, and `outcome` — extended
  with `playbook_id` and `experiment_id` tags.
- `SetupEvent` (Runbook 44) already has `strategy_template_version` and outcome fields
  — extended with `playbook_id`.
- `JudgeAction` gains two new action types: `suggest_experiment` and `update_playbook`.

## Scope

1. **`tools/paper_trading.py`** — `SessionState` gains `research: ResearchBudgetState`;
   separate accounting for research trades; research fills charged to `research.cash`
2. **`schemas/research_budget.py`** — `ResearchBudgetState`, `ResearchTrade`,
   `ExperimentAttribution` Pydantic models
3. **`services/playbook_outcome_aggregator.py`** — reads `SignalEvent` / `SetupEvent`
   outcomes tagged with `playbook_id`; writes stats to `## Validation Evidence` sections
   in `vector_store/playbooks/*.md`
4. **`schemas/judge_feedback.py`** — add `suggest_experiment` and `update_playbook`
   to `JudgeActionType` literal; add `ExperimentSuggestion` and `PlaybookEditSuggestion`
   payload models
5. **`agents/judge_agent_client.py`** — handle new action types in the evaluation loop;
   emit experiment suggestion as a new `ExperimentSpec`; surface playbook edit suggestion
   via Ops API for human review
6. **`ops_api/routers/research.py`** — new router: research budget status, experiment
   list, playbook validation summaries, pending playbook edit suggestions
7. **`ops_api/app.py`** — include research router
8. **`prompts/judge_prompt_research.txt`** — additions to judge evaluation prompt
   explaining when to suggest experiments vs. update playbooks
9. **`tests/test_research_budget.py`** — unit tests for research budget accounting
10. **`tests/test_playbook_outcome_aggregator.py`** — unit tests for stats computation
    and `.md` file update

## Out of Scope

- Automated application of `update_playbook` suggestions (human review required)
- Automated experiment creation without judge suggestion (user or judge initiates)
- Research budget in backtest runner (already has experiment spec framework; wiring
  research budget to playbook update is a separate, lower-priority runbook)
- Cross-session experiment continuity (experiment state persists only within a session;
  cross-session persistence is follow-up)

## Key Files

- `tools/paper_trading.py` (modify: `SessionState`, fill routing, research accounting)
- `schemas/research_budget.py` (new)
- `services/playbook_outcome_aggregator.py` (new)
- `schemas/judge_feedback.py` (modify: new action types + payloads)
- `agents/judge_agent_client.py` (modify: handle suggest_experiment + update_playbook)
- `ops_api/routers/research.py` (new)
- `ops_api/app.py` (modify: include router)
- `prompts/judge_prompt_research.txt` (new: addendum to judge prompt)
- `tests/test_research_budget.py` (new)
- `tests/test_playbook_outcome_aggregator.py` (new)

## Implementation Steps

### Step 1: Define schemas in `schemas/research_budget.py`

```python
class ResearchTrade(BaseModel):
    model_config = {"extra": "forbid"}
    trade_id: str
    experiment_id: str
    playbook_id: str | None
    symbol: str
    direction: str
    entry_price: float
    exit_price: float | None = None
    qty: float
    entry_ts: datetime
    exit_ts: datetime | None = None
    pnl: float | None = None
    outcome: Literal["hit_1r", "hit_stop", "ttl_expired", "open"] = "open"
    r_achieved: float | None = None
    entry_indicators: dict = Field(default_factory=dict)


class ResearchBudgetState(BaseModel):
    model_config = {"extra": "forbid"}
    initial_capital: float
    cash: float
    positions: dict = Field(default_factory=dict)
    active_experiment_id: str | None = None
    active_playbook_id: str | None = None
    trades: list[ResearchTrade] = Field(default_factory=list)
    total_pnl: float = 0.0
    max_loss_usd: float  # from MetricSpec.max_loss_usd
    paused: bool = False
    pause_reason: str | None = None


class ExperimentAttribution(BaseModel):
    """Maps a SignalEvent or SetupEvent to an experiment and playbook."""
    model_config = {"extra": "forbid"}
    signal_event_id: str | None = None
    setup_event_id: str | None = None
    experiment_id: str
    playbook_id: str | None
    hypothesis: str
```

### Step 2: Extend `SessionState` in `tools/paper_trading.py`

```python
@dataclass
class SessionState:
    ...
    # Research budget (separate from main trading capital)
    research: ResearchBudgetState | None = None
    active_experiments: list[ExperimentSpec] = field(default_factory=list)
```

**Research budget initialization** (in `create_session`):

```python
research_fraction = float(os.environ.get("RESEARCH_BUDGET_FRACTION", "0.10"))
research_capital = initial_balance * research_fraction
state.research = ResearchBudgetState(
    initial_capital=research_capital,
    cash=research_capital,
    max_loss_usd=float(os.environ.get("RESEARCH_MAX_LOSS_USD", str(research_capital * 0.5))),
)
```

**Research trade routing** (in `_execute_trigger`):

When a trigger fires with `is_research_trigger=True` metadata (set by the strategist
plan when an active ExperimentSpec covers that trigger's category + symbol):

```python
if trigger.metadata.get("is_research") and state.research and not state.research.paused:
    # Check research budget has sufficient cash
    research_position_fraction = float(os.environ.get("RESEARCH_POSITION_FRACTION", "0.05"))
    notional = state.research.cash * research_position_fraction
    # Execute research fill (same simulator, charged to research.cash not main cash)
    fill = _simulate_fill(order, price)
    _record_research_fill(state.research, fill, trigger)
else:
    # Normal execution against main budget
    ...
```

### Step 3: `services/playbook_outcome_aggregator.py`

```python
class PlaybookOutcomeAggregator:
    """Reads signal/setup outcomes tagged with playbook_id and writes evidence to .md files."""

    PLAYBOOK_DIR = Path("vector_store/playbooks")

    def aggregate(self, playbook_id: str) -> PlaybookValidationResult:
        """Compute validation stats for a playbook from all known outcomes."""
        outcomes = self._load_outcomes(playbook_id)
        if not outcomes:
            return PlaybookValidationResult(playbook_id=playbook_id, status="insufficient_data", n_trades=0)

        n = len(outcomes)
        hits = sum(1 for o in outcomes if o.outcome == "hit_1r")
        win_rate = hits / n if n > 0 else None
        avg_r = statistics.mean(o.r_achieved for o in outcomes if o.r_achieved is not None) if outcomes else None
        bars = [o.bars_to_outcome for o in outcomes if o.bars_to_outcome is not None]
        median_bars = statistics.median(bars) if bars else None

        min_sample = self._get_min_sample_size(playbook_id)
        hypothesis = self._get_hypothesis(playbook_id)
        status = self._evaluate_status(n, min_sample, win_rate, avg_r, hypothesis)

        return PlaybookValidationResult(
            playbook_id=playbook_id,
            status=status,
            n_trades=n,
            win_rate=round(win_rate, 3) if win_rate is not None else None,
            avg_r=round(avg_r, 3) if avg_r is not None else None,
            median_bars_to_outcome=median_bars,
        )

    def write_evidence_to_playbook(self, result: PlaybookValidationResult, judge_notes: str | None = None) -> None:
        """Update ## Validation Evidence section in the playbook .md file."""
        path = self.PLAYBOOK_DIR / f"{result.playbook_id}.md"
        if not path.exists():
            logger.warning("Playbook not found: %s", path)
            return
        content = path.read_text(encoding="utf-8")
        evidence_block = self._render_evidence_block(result, judge_notes)
        if "## Validation Evidence" in content:
            # Replace existing block
            content = re.sub(
                r"## Validation Evidence\n.*?(?=\n##|\Z)",
                evidence_block,
                content,
                flags=re.DOTALL,
            )
        else:
            content += f"\n{evidence_block}"
        path.write_text(content, encoding="utf-8")
        logger.info("Updated playbook evidence: %s (status=%s, n=%s)", result.playbook_id, result.status, result.n_trades)
```

**Trigger:** The aggregator runs:
1. When a `ResearchTrade` closes (outcome != "open")
2. On a scheduled cadence via the paper trading session's background loop (every hour)
3. When the judge emits an `update_playbook` action

### Step 4: New judge action types in `schemas/judge_feedback.py`

Extend `JudgeActionType` literal:

```python
JudgeActionType = Literal[
    "hold",
    "replan",
    "policy_adjust",
    "stand_down",
    "suggest_experiment",   # NEW
    "update_playbook",      # NEW
]
```

Add payload models:

```python
class ExperimentSuggestion(BaseModel):
    model_config = {"extra": "forbid"}
    playbook_id: str
    hypothesis: str
    target_symbols: list[str]
    trigger_categories: list[str]
    min_sample_size: int = 20
    max_loss_usd: float = 50.0
    rationale: str  # Why the judge is suggesting this experiment

class PlaybookEditSuggestion(BaseModel):
    model_config = {"extra": "forbid"}
    playbook_id: str
    section: Literal["Notes", "Patterns", "Validation Evidence"]
    suggested_text: str
    evidence_summary: str  # What evidence supports this edit
    requires_human_review: bool = True  # Always True; human approves before applying
```

Extend `JudgeAction`:

```python
class JudgeAction(BaseModel):
    ...
    experiment_suggestion: ExperimentSuggestion | None = None
    playbook_edit_suggestion: PlaybookEditSuggestion | None = None
```

### Step 5: Judge prompt addendum in `prompts/judge_prompt_research.txt`

```
RESEARCH LOOP GUIDANCE:

When evaluating live trade performance, you have access to two new actions:

1. suggest_experiment — Use when:
   - A specific trigger condition is firing frequently but losing (win_rate < 45% over
     10+ trades), AND you suspect the edge is regime-specific or indicator-specific.
   - There is a playbook whose hypothesis has not yet been validated (status="insufficient_data").
   - Propose a focused ExperimentSpec with a clear hypothesis, specific symbols, and
     categories that isolate the suspected variable.
   - Keep experiments narrow: one variable at a time.

2. update_playbook — Use when:
   - A playbook's Validation Evidence shows status="validated" or "refuted" with
     n_trades >= min_sample_size, AND
   - The Notes or Patterns section is inconsistent with the evidence.
   - Example: volume_confirmation playbook claims volume adds 12% win rate but evidence
     shows win_rate_high_volume = 0.48 vs win_rate_low_volume = 0.47 (no edge) — suggest
     removing the volume filter recommendation or revising the threshold.
   - ALWAYS set requires_human_review=True. You are suggesting, not deciding.

WHEN NOT TO USE THESE ACTIONS:
- Do not suggest an experiment if the research budget is currently paused (max_loss_usd
  exceeded) or if an experiment is already running on the same playbook.
- Do not suggest updating a playbook based on fewer than min_sample_size outcomes.
- Do not suggest experiments for emergency-exit patterns — those are correctness features,
  not hypothesis tests.
```

### Step 6: Ops API in `ops_api/routers/research.py`

```python
@router.get("/research-budget")
async def get_research_budget(session_id: str) -> dict:
    """Current research budget state: cash, pnl, paused, active_experiment."""
    ...

@router.get("/research-budget/experiments")
async def list_experiments(session_id: str) -> list[dict]:
    """All ExperimentSpecs for this session with their current status."""
    ...

@router.get("/playbooks/{playbook_id}/validation")
async def get_playbook_validation(playbook_id: str) -> dict:
    """Current validation evidence for a playbook."""
    ...

@router.get("/playbooks/edit-suggestions")
async def get_pending_edit_suggestions() -> list[PlaybookEditSuggestion]:
    """Pending judge-suggested playbook edits awaiting human review."""
    ...

@router.post("/playbooks/{playbook_id}/apply-suggestion")
async def apply_edit_suggestion(playbook_id: str, suggestion_id: str) -> dict:
    """Human approves a playbook edit suggestion → writes to .md file."""
    ...
```

### Step 7: Playbook attribution in existing signal/setup schemas

Extend `SignalEvent` (in `schemas/signal_event.py`) with optional fields:

```python
experiment_id: str | None = Field(default=None, description="ExperimentSpec ID if this signal was a research trade.")
playbook_id: str | None = Field(default=None, description="Playbook being tested by this research trade.")
```

Same extension to `SetupEvent.signal_event_id` linkage — when a `SetupEvent` is linked
to a `SignalEvent` that has `playbook_id` set, the aggregator can trace back from setup
lifecycle to playbook validation.

## Environment Variables

```
RESEARCH_BUDGET_FRACTION=0.10      # Fraction of session capital allocated to research (default 10%)
RESEARCH_POSITION_FRACTION=0.05    # Per-trade size as fraction of research budget (default 5%)
RESEARCH_MAX_LOSS_USD=             # Auto-pause if research cumulative loss exceeds this (default 50% of research capital)
RESEARCH_AUTO_AGGREGATE=true       # Run PlaybookOutcomeAggregator on each closed research trade (default true)
PLAYBOOK_MIN_EVIDENCE_WRITE=5      # Minimum n_trades before writing to .md file (default 5; below min_sample_size)
```

## Routing Table: When Research Trades Fire

A trigger is routed to the research budget when ALL of the following hold:

| Condition | Detail |
|---|---|
| Active ExperimentSpec | `state.research.active_experiment_id` is set and `status == "running"` |
| Symbol match | `ExposureSpec.target_symbols` includes the trigger's symbol (or is empty = all symbols) |
| Category match | `ExposureSpec.trigger_categories` includes the trigger's category (or is empty) |
| Research budget solvent | `state.research.cash > 0` and `not state.research.paused` |
| Trade not main-budget-only | Trigger does not have `force_main_budget=True` metadata |

Emergency exits always route to the main budget regardless of research state.

## Test Plan

```bash
# Unit: research budget accounting
uv run pytest tests/test_research_budget.py -vv

# Unit: playbook outcome aggregation + .md write
uv run pytest tests/test_playbook_outcome_aggregator.py -vv

# Unit: new judge action types validate
uv run pytest -k "judge_action" -vv

# Regression: existing paper trading tests unaffected
uv run pytest tests/test_paper_trading.py -vv

# Full suite
uv run pytest -x -q
```

## Acceptance Criteria

- [ ] `SessionState.research` populated from `RESEARCH_BUDGET_FRACTION` at session creation
- [ ] Research trades charged to `research.cash`, not `state.cash`
- [ ] Research trades tagged with `experiment_id` and `playbook_id` in signal events
- [ ] `PlaybookOutcomeAggregator.aggregate()` computes correct win_rate, avg_r, median_bars
- [ ] `write_evidence_to_playbook()` updates `## Validation Evidence` section without
      corrupting the rest of the `.md` file
- [ ] `suggest_experiment` judge action creates a new `ExperimentSpec` in `running` state
- [ ] `update_playbook` judge action surfaces suggestion via Ops API (does NOT auto-apply)
- [ ] Human `apply-suggestion` endpoint applies edit and marks suggestion consumed
- [ ] Research budget auto-pauses when cumulative loss exceeds `max_loss_usd`
- [ ] Emergency exits always route to main budget (not research)
- [ ] `GET /research-budget` returns current state
- [ ] `GET /playbooks/{id}/validation` returns current evidence

## Human Verification Evidence

```
TODO: Run a 48-hour paper trading session with RESEARCH_BUDGET_FRACTION=0.10.
Manually create an ExperimentSpec for bollinger_squeeze hypothesis.
Verify:
- Research trades appear in /research-budget with correct P&L isolation
- After 5+ closed research trades, bollinger_squeeze.md shows updated Validation Evidence
- Judge evaluation (after 10+ live trade losses on RSI-based exits) suggests an experiment
  on rsi_extremes playbook
- Playbook edit suggestion surfaces in /playbooks/edit-suggestions
- Human apply-suggestion updates the .md file correctly
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-21 | Runbook created — research budget in paper trading + playbook validation loop | Claude |

## Worktree Setup

```bash
git worktree add -b feat/research-budget-paper-trading ../wt-research-budget main
cd ../wt-research-budget
```

## Git Workflow

```bash
git checkout -b feat/research-budget-paper-trading

git add schemas/research_budget.py \
  services/playbook_outcome_aggregator.py \
  schemas/judge_feedback.py \
  agents/judge_agent_client.py \
  tools/paper_trading.py \
  ops_api/routers/research.py \
  ops_api/app.py \
  prompts/judge_prompt_research.txt \
  schemas/signal_event.py \
  vector_store/playbooks/ \
  tests/test_research_budget.py \
  tests/test_playbook_outcome_aggregator.py

uv run pytest tests/test_research_budget.py tests/test_playbook_outcome_aggregator.py -vv
git commit -m "feat: research budget in paper trading + playbook validation loop (Runbook 48)"
```

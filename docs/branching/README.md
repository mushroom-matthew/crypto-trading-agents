# Branching Runbooks

This folder contains branch-specific runbooks for parallel agents. Each runbook includes scope, key files, acceptance criteria, test gating, and explicit git commands.

## How To Use
1) Pick the branch runbook.
2) Run the Worktree Setup and create the branch **before** editing any files.
3) Implement the changes.
4) Run the Test Plan and paste the output into the Test Evidence section.
5) Fill out Human Verification Evidence and Change Log entries.
6) Follow the Git Workflow section to add and commit (only after evidence is recorded).

## Naming Scheme
- `NN-` prefix = priority order (lower is higher priority).
- `X-` prefix = completed runbooks.
- `_` prefix = backlog runbooks (often stored in `docs/branching/later/`).

## Priority Runbooks (Numbered)
- [01-strategist-simplification.md](01-strategist-simplification.md): Simplify LLM strategist - allow empty triggers, remove risk redundancy, vector store prep. **Phase 1 COMPLETE** (schema, classifier, prompt, risk removal, stance tracking). Vector store and regime alerts deferred.
- [37-risk-budget-commit-actual.md](37-risk-budget-commit-actual.md): **P0 — Real-money blocker.** Fix 255x overcharge in `_commit_risk_budget()` — deduct actual_risk_at_stop, not theoretical cap. Budget exhausts after 2–4 trades with microscopic positions.
- [38-candlestick-pattern-features.md](38-candlestick-pattern-features.md): Add candlestick morphology to the feature vector — 15 new identifiers (is_hammer, is_engulfing, is_inside_bar, candle_strength, etc.) in `metrics/candlestick.py`. Prerequisite for reversal and breakout templates.
- [39-universe-screener.md](39-universe-screener.md): Autonomous instrument screening — anomaly scoring across a configurable crypto universe, LLM instrument recommendation with thesis and strategy type. Shifts LLM role from "write RSI rules" to "choose what to trade and why."
- [40-compression-breakout-template.md](40-compression-breakout-template.md): Compression→expansion breakout strategy template — 4 new indicators (bb_bandwidth_pct_rank, compression_flag, expansion_flag, breakout_confirmed) and canonical breakout prompt. Depends on Runbook 38.
- [41-htf-structure-cascade.md](41-htf-structure-cascade.md): Always load daily candles as an anchor layer — 12 new `htf_*` fields (daily_high, daily_low, prev_daily_high, 5d_high, daily_atr, etc.) for level-based stop and target anchoring. Prerequisite for Runbook 42.
- [42-level-anchored-stops.md](42-level-anchored-stops.md): Absolute stop/target prices stored per TradeLeg at fill time — 7 stop anchor types (htf_daily_low, donchian_lower, candle_low, atr, etc.), 6 target types (measured_move, htf_daily_high, r_multiple). Exposes `below_stop` and `above_target` as trigger identifiers. Depends on Runbooks 41 and 38.
- [43-signal-ledger-and-reconciler.md](43-signal-ledger-and-reconciler.md): Signal Ledger + Outcome Reconciler — `SignalEvent` schema with full provenance, persistent `signal_ledger` table, fill drift telemetry (slippage_bps, fill_latency_ms), MFE/MAE tracking, and 5 statistical capital gates. Foundation for monetizable signal track record. Depends on Runbook 42 for stop/target fields.
- [46-template-matched-plan-generation.md](46-template-matched-plan-generation.md): Wire vector store retrieval to concrete prompt templates — adds `compression_breakout.md` to vector store, `RetrievalResult` returns `template_id`, `generate_plan()` loads matching `prompts/strategies/*.txt` automatically. Zero schema changes. Gate: R39 screener in paper trading.
- [47-hard-template-binding.md](47-hard-template-binding.md): `template_id` field on `StrategyPlan` + trigger compiler enforcement — triggers using identifiers outside the declared template's allowed set are blocked at compile time. Backwards compatible (Optional field). Gate: R46 retrieval routing validated ≥ 80% accuracy.
- [48-research-budget-paper-trading.md](48-research-budget-paper-trading.md): Research budget in paper trading — separate ledger + capital pool for hypothesis testing; `PlaybookOutcomeAggregator` writes validated stats to `vector_store/playbooks/*.md`; judge gains `suggest_experiment` and `update_playbook` actions. All 7 playbooks updated with hypothesis + validation evidence structure. Parallel-safe with Runbooks 46/47.
- ~~04-emergency-exit-runbook-hold-cooldown.md~~: Min-hold and cooldown enforcement. → Completed, see [X-emergency-exit-runbook-hold-cooldown.md](X-emergency-exit-runbook-hold-cooldown.md).
- ~~05-emergency-exit-runbook-bypass-override.md~~: Bypass and override behavior. → Completed, see [X-emergency-exit-runbook-bypass-override.md](X-emergency-exit-runbook-bypass-override.md).
- ~~06-emergency-exit-runbook-edge-cases.md~~: Emergency-exit edge cases. → Completed, see [X-emergency-exit-runbook-edge-cases.md](X-emergency-exit-runbook-edge-cases.md).
- [07-aws-deploy.md](07-aws-deploy.md): AWS infrastructure, secrets, and CI/CD wiring.
- [08-multi-wallet.md](08-multi-wallet.md): Multi-wallet architecture (Phantom/Solana/Ethereum read-only), reconciliation, UI.
- ~~09-runbook-architecture-wiring.md~~: Learning-risk wiring and integration points. → Completed, see [X-09-runbook-architecture-wiring.md](X-09-runbook-architecture-wiring.md).
- ~~10-runbook-learning-book.md~~: Learning Book config, tagging, accounting, acceptance criteria. → Completed, see [X-10-runbook-learning-book.md](X-10-runbook-learning-book.md).
- ~~11-runbook-experiment-specs.md~~: ExperimentSpec schemas, exposure taxonomy, metric definitions. → Completed, see [X-11-runbook-experiment-specs.md](X-11-runbook-experiment-specs.md).
- ~~12-runbook-no-learn-zones-and-killswitches.md~~: Enforceable no-learn policies and kill switches. → Completed, see [X-12-runbook-no-learn-zones-and-killswitches.md](X-12-runbook-no-learn-zones-and-killswitches.md).
- ~~13-judge-death-spiral-floor.md~~: Minimum trigger floor to prevent judge death spirals (zero-activity re-enablement). → Completed, see [X-judge-death-spiral-floor.md](X-judge-death-spiral-floor.md).
- ~~14-risk-used-default-to-actual.md~~: Default risk_used_abs to actual_risk_at_stop when budgets are off. → Completed, see [X-14-risk-used-default-to-actual.md](X-14-risk-used-default-to-actual.md).
- ~~15-min-hold-exit-timing-validation.md~~: Validate min_hold vs exit timeframe; track min_hold_binding_pct. → Completed, see [X-15-min-hold-exit-timing-validation.md](X-15-min-hold-exit-timing-validation.md).
- ~~16-judge-stale-snapshot-skip.md~~: Skip or adapt judge evals when snapshot is unchanged since last eval. → Completed, see [X-judge-stale-snapshot-skip.md](X-judge-stale-snapshot-skip.md).
- ~~17-graduated-derisk-taxonomy.md~~: Exit taxonomy & partial exit ladder. → Completed, see [X-17-graduated-derisk-taxonomy.md](X-17-graduated-derisk-taxonomy.md).
- ~~18-phase1-deterministic-policy-integration.md~~: Phase 1 policy pivot contract — deterministic, trigger-gated target-weight policy (mandatory). → Completed, see [X-18-phase1-deterministic-policy-integration.md](18-phase1-deterministic-policy-integration.md). (schemas, policy_engine.py, integration layer, backtest runner wiring, 52 tests).
- [19-phase2-model-phat-integration.md](19-phase2-model-phat-integration.md): Phase 2 contract — `p_hat` as signal source only (optional/reversible).
- ~~20-judge-attribution-rubric.md~~: Judge attribution contract — single-bucket blame model and replan/policy-adjust action gating. → **COMPLETE** (attribution schema, compute_attribution, action gating validators, 62 tests).
- [21-emergency-exit-sensitivity.md](21-emergency-exit-sensitivity.md): Fix overly sensitive emergency exit (`tf_1d_atr > tf_4h_atr` tautology). Emergency exits dominated 80% of all exits in backtest ebf53879.
- [22-exit-binding-prompt-gap.md](22-exit-binding-prompt-gap.md): Document exit category binding rules in LLM prompts. LLM generates cross-category exits that are silently blocked (11 blocks in ebf53879).
- [23-hold-rule-calibration.md](23-hold-rule-calibration.md): Tighten hold rule guidance — `rsi_14 > 45` is near-always true, making normal exits dead code (12 blocks in ebf53879).
- [24-judge-eval-flood.md](24-judge-eval-flood.md): Fix stale snapshot skip not advancing `next_judge_time`, causing 109 hourly stale skips instead of ~28 total evals. Change default cadence from 4h to 12h.
- [25-trade-volume-deficit.md](25-trade-volume-deficit.md): Address systemic under-trading (0.43 entries/day). Dead trigger detection, fire rate guidance, drought telemetry.
- [26-risk-telemetry-accuracy.md](26-risk-telemetry-accuracy.md): Fix phantom `risk=50` in judge snapshot that wastes feedback slots on non-issues.
- [27-stance-diversity.md](27-stance-diversity.md): LLM never uses defensive/wait stance despite judge recommending it. Add defensive examples, structured stance hints.
- [28-judge-action-contract.md](28-judge-action-contract.md): Define structured judge actions, TTLs, and action events; wire recommended_action routing.
- [29-judge-structured-multipliers.md](29-judge-structured-multipliers.md): Replace free-text sizing parsing with structured multipliers and clamps.
- [30-judge-immediate-application.md](30-judge-immediate-application.md): Apply intraday judge constraints to active engines without waiting for replans.
- [31-judge-stance-enforcement.md](31-judge-stance-enforcement.md): Deterministic enforcement of recommended stance (defensive/wait).

Learning-risk runbooks (09-12) are all complete — implemented together on branch `main`. Tag propagation, learning book settings, experiment specs, and no-learn zones/kill switches are all landed.

Judge robustness runbooks (13, 16) are both complete — implemented together on branch `judge-death-spiral-floor`. Runbook 13 prevents death spirals (trigger floor, zero-activity re-enablement). Runbook 16 adds stale snapshot skip, forced re-enablement after consecutive stale evals, and `stale_judge_evals` daily metric.

## Recommended Execution Order

The numbered runbooks reflect creation order, not execution priority. Based on a trust-stack analysis (correctness → enforcement → observability → operator UX → anti-churn), the recommended execution order is:

### Phase 0A — Safety case (emergency exits + judge robustness)
1. **03-06**: Emergency exit series (same-bar dedup → hold/cooldown → bypass/override → edge cases)
2. **13**: Judge death spiral floor (prevents irreversible trading halt)
3. **16**: Judge stale snapshot skip (prevents redundant evaluations reinforcing broken state)

### Phase 0B — Risk correctness ✅ COMPLETE
4. ~~**14**~~: Risk used default to actual (fills show meaningful risk, not $0.00) — Complete.
5. ~~**15**~~: Min-hold exit timing validation (detect when min_hold is the binding constraint) — Complete.

### Phase 0C — Anti-churn control plane (prereq for policy pivot)
6. ~~**policy-pivot-phase0**~~: No-change replan guard + telemetry. → Completed, see [X-policy-pivot-phase0.md](X-policy-pivot-phase0.md).

### Phase 1 — Learning-risk wiring (exploration isolation) ✅ COMPLETE
7. ~~**09-12**~~: Learning-risk series (wiring → learning book → experiment specs → no-learn zones) — All complete.

### Phase 1B — Graduated de-risk (after safety case, before strategist rework) ✅ COMPLETE
- ~~**17**~~: Exit taxonomy & partial exit ladder — All 5 phases complete (schema, partial exit execution, risk_reduce guardrails, risk_off latch, strategist integration). Ready for backtest validation.

### Phase 2 — Strategy architecture
8. **01**: Strategist simplification — **Phase 1 COMPLETE** (schema, classifier, prompt, risk removal, stance tracking). Vector store (RAG) and regime alert monitoring deferred to follow-up.

### Phase 3 — Policy pivot contracts (trigger-gated)
9. **18**: Deterministic policy integration (mandatory). Triggers remain permission/direction authority; policy owns magnitude/risk expression.
10. **19**: Model `p_hat` integration as signal source only (optional/reversible). Bound at plan creation/replan only.

### Phase 3B — Judge attribution governance
11. **20**: Judge attribution rubric and action gating. Enforces single primary attribution with evidence and prevents cross-layer blame smearing.

### Phase 4 — Backtest quality (from ebf53879 analysis)
Runbooks 21-27 address issues discovered in backtest ebf53879. Recommended sub-order:

12. **24**: Judge eval flood (quick bug fix — stale skip timer advancement)
13. **26**: Risk telemetry accuracy (fix phantom risk values polluting judge feedback)
14. **21**: Emergency exit sensitivity (biggest impact — 80% of exits are emergency)
15. **22**: Exit binding prompt gap (11 silent blocks per backtest)
16. **23**: Hold rule calibration (12 blocks, makes normal exits dead code)
17. **25**: Trade volume deficit (meta-fix — depends on 21-23 landing first)
18. **27**: Stance diversity (prompt enrichment, lowest urgency)

### Phase 4B — Judge actionability (from backtest 7c860ae1 analysis)
1. **28**: Judge action contract (structured actions, TTLs, action events)
2. **29**: Structured multipliers with clamps (emergency fix for sizing)
3. **30**: Immediate intraday application (close feedback/action gap)
4. **31**: Stance enforcement (defensive/wait gating)

### Phase 5 — Infrastructure expansion
19. **07**: AWS deploy / CI/CD
20. **08**: Multi-wallet (Phantom/Solana/EVM read-only + reconciliation)

### Phase 6 — Strategy intelligence (new direction, 2026-02)
Runbooks 37–43 represent a product direction shift: from "LLM writes indicator rules" toward "LLM reasons about market structure and chooses instruments." These must be sequenced carefully — candle features and HTF structure are prerequisites for the breakout template and level-anchored stops.

**P0 — Must ship before real money:**
21. **37**: Risk budget commit actual (255x overcharge blocks all real-capital use cases) ✅ implemented

**Stratum A — Feature foundations + Signal infrastructure (parallel-safe):**
22. **38**: Candlestick pattern features (independent; adds 15 identifiers to feature vector) ✅ implemented
23. **41**: HTF structure cascade (independent; adds 12 daily anchor fields) ✅ implemented
24. **43**: Signal ledger & outcome reconciler (Signal→Risk Policy→Execution Adapter architecture; statistical capital gates; requires 42 for stop/target fields) ✅ implemented

**Stratum B — Strategy templates (after Stratum A merges):**
25. **40**: Compression breakout template (requires 38 for is_impulse_candle, is_inside_bar) ✅ implemented
26. **42**: Level-anchored stops (requires 38 for candle_low anchor, 41 for htf_daily_low anchor) ✅ implemented

**Stratum C — Market intelligence (after Stratum B validates in paper trading):**
27. **39**: Universe screener (standalone; can be built in parallel but validated after 40/42)
    - **Amendment (2026-02-21)**: Screener pre-selects `template_id` deterministically from
      composite score breakdown (compression_score > 0.60 → `compression_breakout`, etc.).
      Sniffer tuning via `SCREENER_COMPRESSION_WEIGHT` is now the user's strategic lever —
      it controls which template gets applied at the instrument level without requiring
      trigger authorship. See amendment section in [39-universe-screener.md](39-universe-screener.md).

> **Why this order:** Runbook 37 is a correctness bug with zero-cost fix — ship immediately. Candlestick features (38) and HTF structure (41) are additive with no risk of regression and unlock everything downstream. The breakout template (40) and level-anchored stops (42) depend on those features and should be validated together via a paper trading backtest before the universe screener (39) is enabled — you want a working strategy before adding autonomous instrument selection.

> **Paper trading as the validation gate:** Unlike prior runbooks that were validated via backtest, Runbook 39 (universe screener) can only be meaningfully validated via paper trading. The screener surfaces real-time anomalies; no historical simulation can test whether it would have identified the right instrument at the right time. Plan for a 30-day paper trading period after 39 ships before enabling live capital routing.

### Phase 7 — Template-Bound Instrument Strategy (new direction, 2026-02)

Runbooks 46–47 complete the shift from "LLM authors trigger rules" to "LLM selects and
parameterizes a known template." The vector store retrieval infrastructure is already
active (`STRATEGY_VECTOR_STORE_ENABLED=true`); these runbooks close the gap between
retrieval-as-hints and retrieval-as-binding.

> **Architecture context:** The `vector_store/strategies/` docs are retrieval targets.
> The `prompts/strategies/*.txt` files are the actual system prompts. The two systems are
> currently disconnected — retrieval finds a regime match but does not load the
> corresponding prompt. Runbook 46 wires them together. Runbook 47 enforces the contract.

**Stratum D — Template routing (after Stratum C paper trading validation):**
28. **46**: [Template-matched plan generation](46-template-matched-plan-generation.md) —
    Adds `compression_breakout.md` to vector store; wires retrieval to load the
    corresponding `prompts/strategies/*.txt` template automatically. Zero schema changes.
    _Gate: Runbook 39 screener running in paper trading._

**Stratum E — Hard binding (after Stratum D validated):**
29. **47**: [Hard template binding](47-hard-template-binding.md) — Adds `template_id`
    to `StrategyPlan`; trigger compiler blocks triggers using identifiers outside the
    declared template's allowed set. Backwards compatible (Optional field, enforcement
    skipped when `template_id=None`).
    _Gate: Runbook 46 routing confirmed correct ≥ 80% of plan-generation days._

**Stratum F — Research feedback loop (can run in parallel with Stratum D/E):**
30. **48**: [Research budget in paper trading](48-research-budget-paper-trading.md) —
    Separate capital pool + ledger for hypothesis testing in paper trading. Research
    trades tagged with `experiment_id` and `playbook_id`. `PlaybookOutcomeAggregator`
    writes validated stats to `vector_store/playbooks/*.md` `## Validation Evidence`
    sections. Judge gains `suggest_experiment` and `update_playbook` action types.
    All 7 existing playbooks updated with `hypothesis`, `min_sample_size`, and
    `## Research Trade Attribution` sections. New meta-playbook:
    `vector_store/playbooks/experiment_framework.md`.
    _Can start immediately; does not block on Stratum D/E._

> **Why this order matters:** Binding without accurate retrieval produces wrong plans
> silently (LLM declares `compression_breakout` but indicators are trending — triggers
> are blocked as identifier violations, leaving an empty plan). Retrieval accuracy must
> be validated first. The 30-day paper trading gate for Runbook 39 serves double duty:
> it also validates the template routing table from Runbook 46.
>
> Runbook 48 is parallel-safe: the research budget is additive (separate ledger, no
> coupling to template selection). It should ship during the Stratum D paper trading
> validation period so that playbook evidence starts accumulating while the template
> routing table is being validated.

## Backlog Runbooks (_)
- [_per-instrument-workflow.md](_per-instrument-workflow.md): Per-instrument `InstrumentStrategyWorkflow` (one Temporal workflow per active symbol). Deferred until Runbooks 39+46+47 are validated via 30-day paper trading and open architectural questions (workflow ID namespace, multi-timeframe, judge routing) are resolved with operational evidence.
- [_emergency-exit-runbook-judge-loop-design.md](_emergency-exit-runbook-judge-loop-design.md): Judge/strategist loop design gaps (non-test items).
- [_synthetic-data-testing.md](_synthetic-data-testing.md): Synthetic data generation for deterministic trigger testing.
- [later/_comp-audit-risk-followups.md](later/_comp-audit-risk-followups.md): Follow-ups from comp-audit-risk-core.
- ~~later/_policy-pivot-phase0.md~~: No-change replan guard and telemetry. → Completed, see [X-policy-pivot-phase0.md](X-policy-pivot-phase0.md).
- [later/_judge-unification.md](later/_judge-unification.md): Legacy implementation sketch for unified judge service. Attribution contract now defined in [20-judge-attribution-rubric.md](20-judge-attribution-rubric.md).
- [later/_strategist-tool-loop.md](later/_strategist-tool-loop.md): Read-only tool-call loop for strategist.
- [later/_scalper-mode.md](later/_scalper-mode.md): Full scalper mode feature set and comparison tooling.
- [later/_ui-unification.md](later/_ui-unification.md): Optional UI unification enhancements.
- [later/_ui-config-cleanup.md](later/_ui-config-cleanup.md): UI config cleanup after comp-audit prompt changes.
- ~~[later/_policy-integration.md](later/_policy-integration.md)~~: Superseded by [18-phase1-deterministic-policy-integration.md](18-phase1-deterministic-policy-integration.md).
- ~~[later/_model-integration.md](later/_model-integration.md)~~: Superseded by [19-phase2-model-phat-integration.md](19-phase2-model-phat-integration.md).

## Completed Runbooks (X)
- [X-emergency-exit-runbook-same-bar-dedup.md](X-emergency-exit-runbook-same-bar-dedup.md): Same-bar competition and deduplication priority.
- [X-comp-audit-risk-core.md](X-comp-audit-risk-core.md): Phase 0 risk correctness and budget integrity.
- [X-comp-audit-trigger-cadence.md](X-comp-audit-trigger-cadence.md): Scalper cadence and signal serialization.
- [X-comp-audit-indicators-prompts.md](X-comp-audit-indicators-prompts.md): Fast indicators, Donchian high/low, momentum prompts, compute optimizations.
- [X-comp-audit-metrics-parity.md](X-comp-audit-metrics-parity.md): Live/backtest metrics parity and annualization consistency.
- [X-comp-audit-ui-trade-stats.md](X-comp-audit-ui-trade-stats.md): Per-trade risk/perf stats in UI and APIs.
- [X-judge-feedback-enforcement.md](X-judge-feedback-enforcement.md): Enforce judge feedback in execution paths.
- [X-emergency-exit-runbook-hold-cooldown.md](X-emergency-exit-runbook-hold-cooldown.md): Emergency exit min-hold and cooldown enforcement.
- [X-emergency-exit-runbook-bypass-override.md](X-emergency-exit-runbook-bypass-override.md): Emergency exit bypass/override semantics + judge category kill-switch fix.
- [X-emergency-exit-runbook-edge-cases.md](X-emergency-exit-runbook-edge-cases.md): Emergency exit edge cases (missing exit_rule handling).
- [X-judge-death-spiral-floor.md](X-judge-death-spiral-floor.md): Minimum trigger floor, zero-activity re-enablement, stale snapshot detection.
- [X-judge-stale-snapshot-skip.md](X-judge-stale-snapshot-skip.md): Stale snapshot skip, forced re-enablement after consecutive stale evals, daily metric.
- [X-policy-pivot-phase0.md](X-policy-pivot-phase0.md): No-change replan guard, suppression metrics, decision record metadata.
- [X-09-runbook-architecture-wiring.md](X-09-runbook-architecture-wiring.md): Learning-risk wiring — tag propagation from triggers to orders.
- [X-10-runbook-learning-book.md](X-10-runbook-learning-book.md): Learning Book settings, risk budgets, isolated accounting.
- [X-11-runbook-experiment-specs.md](X-11-runbook-experiment-specs.md): ExperimentSpec schema, lifecycle validation, exposure filtering.
- [X-12-runbook-no-learn-zones-and-killswitches.md](X-12-runbook-no-learn-zones-and-killswitches.md): Learning gate evaluator, kill switches, no-learn zones.
- [X-14-risk-used-default-to-actual.md](X-14-risk-used-default-to-actual.md): Risk used default to actual risk at stop when budgets off.
- [X-15-min-hold-exit-timing-validation.md](X-15-min-hold-exit-timing-validation.md): Min-hold vs exit timeframe validation, min_hold_binding_pct metric.
- [X-17-graduated-derisk-taxonomy.md](X-17-graduated-derisk-taxonomy.md): Graduated de-risk taxonomy — risk_reduce (partial trim), risk_off (defensive flatten), exit_fraction field, precedence tiering.
- [18-phase1-deterministic-policy-integration.md](18-phase1-deterministic-policy-integration.md): Phase 1 policy engine — PolicyConfig, PolicyEngine, PolicyTriggerIntegration, backtest runner wiring.
- [20-judge-attribution-rubric.md](20-judge-attribution-rubric.md): Judge Attribution Rubric — single-primary attribution, action gating (replan for plan/trigger, policy_adjust for policy), evidence requirements.

## Notes
- If tests cannot be run locally, obtain user-run output and paste it into the Test Evidence section before committing.
- Coordinate with other agents to avoid overlapping files in parallel branches.

## Human Verification Evidence
- Follow the Human Verification section in each runbook and paste your observations into the Human Verification Evidence section before committing.

## Change Log
- Each runbook includes a Change Log section. Agents must update it with a brief summary of changes and files touched before committing.
## Worktree Usage
- All parallel branches are intended to run on the same machine from a single clone.
- Use git worktree to create per-branch working directories.
- Each runbook includes a Worktree Setup section with exact commands.

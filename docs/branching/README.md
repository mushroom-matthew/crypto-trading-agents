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

## Status Conventions (Important)
- The **Priority Runbooks** list is a planning/implementation queue. A runbook appearing there does **not** mean the code is already implemented.
- A runbook is considered **implemented** only when one of the following is true:
- it is moved to the **Completed Runbooks (X)** section (`X-*.md`), or
- it is listed in **Completed Runbooks (X)** (even if the file has not yet been renamed with `X-`), or
- it is explicitly marked `✅ implemented` / `✅ COMPLETE` in this README.
- `[docs-only runbook]` means the design/contract doc exists, but code implementation has not landed yet.
- If no explicit status marker is present, treat the runbook as **pending / not yet implemented**.

## Current Progress Snapshot (Git-Derived, `dab1eb1`)
- This is a quick implementation-status read from `git log` + README markers so readers can orient before reading phase details.
- **Overall:** implementation progress is best described as **around Phase 6** (with earlier phases largely complete, Phase 6 mostly landed, and Phases 7–8 primarily design/docs at the moment).
- **Phase 0A / 0B / 0C:** `[largely complete]` (emergency-exit safety series, judge robustness, risk correctness, no-change replan guard all have completion markers and/or implementation commits).
- **Phase 1 / 1B:** `[complete]` (learning-risk wiring runbooks `09–12` and graduated de-risk `17` are implemented/closed).
- **Phase 2:** `[partial]` (`01` core strategist simplification landed via `da166bd`; vector store infra exists, and regime alert schemas exist, but the deferred follow-ups in Runbook `01` are still not fully wired).
- **Phase 3:** `[partial]` (`18` implemented; `19` remains pending and there is no clear `p_hat` integration path in current code).
- **Phase 3B:** `[implemented]` (`20` Judge Attribution Rubric is implemented and already listed in Completed Runbooks).
- **Phase 4:** `[implemented]` (commit evidence exists for runbooks `21–27`, e.g. `73c0d61`, `dedebc9`).
- **Phase 4B:** `[implemented]` (deeper code audit confirms `28–31` are landed: `JudgeAction` contract + TTL/persistence/routing, structured multipliers + clamps, immediate intraday application, and stance enforcement).
- **Phase 5:** `[pending]` (`07`, `08` infra expansion not marked complete).
- **Phase 6:** `[complete]` (`37`, `38`, `39`, `40`, `41`, `42`, `43`, `44`, `45` all implemented; R39 universe screener merged `b7c5d5f`, now in paper trading validation gate).
- **Phase 7:** `[complete]` (`48` research budget runtime implemented `aa90202`; `46` template routing implemented `dc5bb98`; `47` hard template binding implemented `7f24256`; R46/R47 accuracy gates tracked via `GET /analytics/template-routing` during paper trading).
- **Phase 8:** `[planning/docs]` (runbooks `49–56` are authored design contracts; code implementation not yet landed).

## Current Active Priorities (Git-Derived)
- **Primary execution frontier:** **Start paper trading session** to validate Phase 6 (R39 screener + R40/42 templates). This is the gate for R46 and R47.
- **Running in parallel:** **R48 research budget** is live — paper trading session should set `RESEARCH_BUDGET_FRACTION=0.10` and create an ExperimentSpec for at least one playbook (e.g. `bollinger_squeeze`) to begin accumulating validation evidence.
- **Next implementation frontier:** **R46** (template routing, gated on R39 running in paper trading) → **R47** (hard binding, gated on R46 ≥ 80% accuracy).
- **Then:** **Phase 8 (`49–56`)** reasoning-agent operating system contracts (currently docs-only).
- **Parallel / optional tracks:** **Phase 5 (`07`, `08`)** infra expansion and **Runbook `19`** (`p_hat` integration, optional/reversible).

## Priority Runbooks (Numbered)
- [01-strategist-simplification.md](01-strategist-simplification.md): Simplify LLM strategist - allow empty triggers, remove risk redundancy, vector store prep. **Phase 1 COMPLETE** (schema, classifier, prompt, risk removal, stance tracking). Vector store and regime alerts deferred.
- [37-risk-budget-commit-actual.md](37-risk-budget-commit-actual.md): **P0 — Real-money blocker.** Fix 255x overcharge in `_commit_risk_budget()` — deduct actual_risk_at_stop, not theoretical cap. Budget exhausts after 2–4 trades with microscopic positions. `✅ implemented`
- [38-candlestick-pattern-features.md](38-candlestick-pattern-features.md): Add candlestick morphology to the feature vector — 15 new identifiers (is_hammer, is_engulfing, is_inside_bar, candle_strength, etc.) in `metrics/candlestick.py`. Prerequisite for reversal and breakout templates. `✅ implemented`
- [39-universe-screener.md](39-universe-screener.md): Autonomous instrument screening — anomaly scoring across a configurable crypto universe, LLM instrument recommendation with thesis and strategy type. Shifts LLM role from "write RSI rules" to "choose what to trade and why." `✅ implemented` — now in **paper trading validation gate** (30-day window before live capital).
- [40-compression-breakout-template.md](40-compression-breakout-template.md): Compression→expansion breakout strategy template — 4 new indicators (bb_bandwidth_pct_rank, compression_flag, expansion_flag, breakout_confirmed) and canonical breakout prompt. Depends on Runbook 38. `✅ implemented`
- [41-htf-structure-cascade.md](41-htf-structure-cascade.md): Always load daily candles as an anchor layer — 12 new `htf_*` fields (daily_high, daily_low, prev_daily_high, 5d_high, daily_atr, etc.) for level-based stop and target anchoring. Prerequisite for Runbook 42. `✅ implemented`
- [42-level-anchored-stops.md](42-level-anchored-stops.md): Absolute stop/target prices stored per TradeLeg at fill time — 7 stop anchor types (htf_daily_low, donchian_lower, candle_low, atr, etc.), 6 target types (measured_move, htf_daily_high, r_multiple). Exposes `below_stop` and `above_target` as trigger identifiers. Depends on Runbooks 41 and 38. `✅ implemented`
- [43-signal-ledger-and-reconciler.md](43-signal-ledger-and-reconciler.md): Signal Ledger + Outcome Reconciler — `SignalEvent` schema with full provenance, persistent `signal_ledger` table, fill drift telemetry (slippage_bps, fill_latency_ms), MFE/MAE tracking, and 5 statistical capital gates. Foundation for monetizable signal track record. Depends on Runbook 42 for stop/target fields. `✅ implemented`
- [44-setup-event-generator.md](44-setup-event-generator.md): Setup event generator with frozen feature snapshots + hashes and template/version provenance for post-trade analysis. `✅ implemented`
- [45-adaptive-trade-management.md](45-adaptive-trade-management.md): Adaptive trade management (R-multiple state machine, trailing/partial management semantics) integrated into backtest execution paths. `✅ implemented`
- [46-template-matched-plan-generation.md](46-template-matched-plan-generation.md): Wire vector store retrieval to concrete prompt templates — adds `compression_breakout.md` to vector store, `RetrievalResult` returns `template_id`, `generate_plan()` loads matching `prompts/strategies/*.txt` automatically. Zero schema changes. Gate: R39 screener in paper trading. `✅ implemented`
- [47-hard-template-binding.md](47-hard-template-binding.md): `template_id` field on `StrategyPlan` + trigger compiler enforcement — triggers using identifiers outside the declared template's allowed set are blocked at compile time. Backwards compatible (Optional field). Gate: R46 retrieval routing validated ≥ 80% accuracy (track via `GET /analytics/template-routing`). `✅ implemented`
- [48-research-budget-paper-trading.md](48-research-budget-paper-trading.md): Research budget in paper trading — separate ledger + capital pool for hypothesis testing; `PlaybookOutcomeAggregator` writes validated stats to `vector_store/playbooks/*.md`; judge gains `suggest_experiment` and `update_playbook` actions. All 7 playbooks updated with hypothesis + validation evidence structure. Parallel-safe with Runbooks 46/47. `✅ implemented`
- [57-screener-timeframe-threading.md](57-screener-timeframe-threading.md): Thread screener `expected_hold_timeframe` from "Use" button through session config → `fetch_indicator_snapshots_activity` → LLM prompt. Fixes mismatch where 1m screener candidates produce 1h triggers. Adds `indicator_timeframe` field to session config; extends `recommendedPlanIntervalHours` for 1m/5m. `✅ implemented`
- [49-market-snapshot-definition.md](49-market-snapshot-definition.md): Multimodal `MarketSnapshot` contract (numerical + derived + text + visual encodings) as the single source of truth for strategist/judge invocations. Enforces timestamped provenance, normalization, staleness checks, and snapshot hashing. `✅ implemented`
- [50-dual-reflection-templates.md](50-dual-reflection-templates.md): Dual-level reflection framework — fast policy-loop reflection (event-driven coherence/invariants/memory-check) plus scheduled high-level reflection (batch outcomes, regime drift, playbook updates), with deterministic tick-level validation kept separate from LLM reflection. `[docs-only runbook]`
- [51-memory-store-diversified-retrieval.md](51-memory-store-diversified-retrieval.md): Diversified episode memory store and contrastive retrieval (`wins`, `losses`, `failure_modes`) for strategist and judge grounding. Builds on Signal Ledger outcome data and adds regime fingerprints + playbook metadata. `[docs-only runbook]`
- [52-playbook-definition-regime-tags.md](52-playbook-definition-regime-tags.md): Typed playbook schema with regime eligibility, entry/invalidation rules, stop/target logic, time-horizon expectations, and historical stats (including holding-time/MAE/MFE distributions). Strategist selects playbook first, then instantiates a plan. `[docs-only runbook]`
- [53-judge-validation-rules-memory-evidence.md](53-judge-validation-rules-memory-evidence.md): Judge loop upgrade from risk-only checks to evidence-based validation using memory failure patterns and cluster evidence; explicit revise/reject criteria for unsupported or overconfident strategist proposals. `[docs-only runbook]`
- [54-reasoning-agent-cadence-rules.md](54-reasoning-agent-cadence-rules.md): Central cadence runbook for a three-tier model (deterministic tick engine, event-driven policy loop, slow structural learning loop), including policy heartbeats/triggers, reflection cadence, and slow-loop scheduling. `[docs-only runbook]`
- [55-regime-fingerprint-transition-detector.md](55-regime-fingerprint-transition-detector.md): Deterministic regime fingerprint + transition detector (bounded, decomposable distance + asymmetric hysteresis + HTF-close gating) that emits `regime_state_changed` policy events and drives policy-loop cadence without per-tick LLM calls. `[docs-only runbook]`
- [56-structural-target-activation-refinement-enforcement.md](56-structural-target-activation-refinement-enforcement.md): Deterministic enforcement layer between playbook schema and trigger engine — compiler validation for structural target candidate selection (expectancy gate telemetry) and activation refinement mode -> trigger identifier/timeframe/confirmation mapping. `[docs-only runbook]`
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
- [19-phase2-model-phat-integration.md](19-phase2-model-phat-integration.md): Phase 2 contract — `p_hat` as signal source only (optional/reversible). `[pending; no clear p_hat path in current code]`
- ~~20-judge-attribution-rubric.md~~: Judge attribution contract — single-bucket blame model and replan/policy-adjust action gating. → **COMPLETE** (attribution schema, compute_attribution, action gating validators, 62 tests).
- [21-emergency-exit-sensitivity.md](21-emergency-exit-sensitivity.md): Fix overly sensitive emergency exit (`tf_1d_atr > tf_4h_atr` tautology). Emergency exits dominated 80% of all exits in backtest ebf53879. `✅ implemented`
- [22-exit-binding-prompt-gap.md](22-exit-binding-prompt-gap.md): Document exit category binding rules in LLM prompts. LLM generates cross-category exits that are silently blocked (11 blocks in ebf53879). `✅ implemented`
- [23-hold-rule-calibration.md](23-hold-rule-calibration.md): Tighten hold rule guidance — `rsi_14 > 45` is near-always true, making normal exits dead code (12 blocks in ebf53879). `✅ implemented`
- [24-judge-eval-flood.md](24-judge-eval-flood.md): Fix stale snapshot skip not advancing `next_judge_time`, causing 109 hourly stale skips instead of ~28 total evals. Change default cadence from 4h to 12h. `✅ implemented`
- [25-trade-volume-deficit.md](25-trade-volume-deficit.md): Address systemic under-trading (0.43 entries/day). Dead trigger detection, fire rate guidance, drought telemetry. `✅ implemented`
- [26-risk-telemetry-accuracy.md](26-risk-telemetry-accuracy.md): Fix phantom `risk=50` in judge snapshot that wastes feedback slots on non-issues. `✅ implemented`
- [27-stance-diversity.md](27-stance-diversity.md): LLM never uses defensive/wait stance despite judge recommending it. Add defensive examples, structured stance hints. `✅ implemented`
- [28-judge-action-contract.md](28-judge-action-contract.md): Define structured judge actions, TTLs, and action events; wire recommended_action routing. `✅ implemented`
- [29-judge-structured-multipliers.md](29-judge-structured-multipliers.md): Replace free-text sizing parsing with structured multipliers and clamps. `✅ implemented`
- [30-judge-immediate-application.md](30-judge-immediate-application.md): Apply intraday judge constraints to active engines without waiting for replans. `✅ implemented`
- [31-judge-stance-enforcement.md](31-judge-stance-enforcement.md): Deterministic enforcement of recommended stance (defensive/wait). `✅ implemented`
- ~~32–36 compiler-enforcement hardening cluster~~: Completed (see Completed Runbooks section entries for [32-exit-binding-enforcement.md](32-exit-binding-enforcement.md) through [36-judge-action-dedup.md](36-judge-action-dedup.md)).

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

### Phase 2 — Strategy architecture [PARTIAL]
8. **01**: Strategist simplification — **Phase 1 COMPLETE** (schema, classifier, prompt, risk removal, stance tracking). Vector store (RAG) and regime alert monitoring deferred to follow-up.

### Phase 3 — Policy pivot contracts (trigger-gated) [PARTIAL]
9. **18**: Deterministic policy integration (mandatory). Triggers remain permission/direction authority; policy owns magnitude/risk expression.
10. **19**: Model `p_hat` integration as signal source only (optional/reversible). Bound at plan creation/replan only.

### Phase 3B — Judge attribution governance ✅ COMPLETE
11. ~~**20**~~: Judge attribution rubric and action gating. Enforces single primary attribution with evidence and prevents cross-layer blame smearing. → Completed (see Completed Runbooks section).

### Phase 4 — Backtest quality (from ebf53879 analysis) ✅ COMPLETE
Runbooks 21-27 address issues discovered in backtest ebf53879. Recommended sub-order:

12. **24**: Judge eval flood (quick bug fix — stale skip timer advancement)
13. **26**: Risk telemetry accuracy (fix phantom risk values polluting judge feedback)
14. **21**: Emergency exit sensitivity (biggest impact — 80% of exits are emergency)
15. **22**: Exit binding prompt gap (11 silent blocks per backtest)
16. **23**: Hold rule calibration (12 blocks, makes normal exits dead code)
17. **25**: Trade volume deficit (meta-fix — depends on 21-23 landing first)
18. **27**: Stance diversity (prompt enrichment, lowest urgency)

### Phase 4B — Judge actionability (from backtest 7c860ae1 analysis) ✅ COMPLETE
1. ~~**28**~~: Judge action contract (structured actions, TTLs, action events) → Completed.
2. ~~**29**~~: Structured multipliers with clamps (emergency fix for sizing) → Completed.
3. ~~**30**~~: Immediate intraday application (close feedback/action gap) → Completed.
4. ~~**31**~~: Stance enforcement (defensive/wait gating) → Completed.

### Phase 4C — Compiler Enforcement Hardening (58cb897f follow-ups) ✅ COMPLETE
Runbooks 32–36 were implemented as a compile-time enforcement / telemetry hardening cluster.
1. ~~**32**~~: Exit binding enforcement → Completed.
2. ~~**33**~~: Hold rule rejection → Completed.
3. ~~**34**~~: Expression error handling → Completed.
4. ~~**35**~~: Block reason normalization → Completed.
5. ~~**36**~~: Judge action dedup → Completed.

### Phase 5 — Infrastructure expansion
19. **07**: AWS deploy / CI/CD
20. **08**: Multi-wallet (Phantom/Solana/EVM read-only + reconciliation)

### Phase 6 — Strategy intelligence (new direction, 2026-02) ✅ COMPLETE — in paper trading validation
Runbooks 37–45 (plus R39) are fully implemented. The full instrument-selection + breakout-template stack is now running. R39 (Universe Screener) is the final piece and is now in its 30-day paper trading validation gate before live capital routing.

**P0 — Must ship before real money:**
21. **37**: Risk budget commit actual (255x overcharge blocks all real-capital use cases) ✅ implemented

**Stratum A — Feature foundations + Signal infrastructure (parallel-safe):**
22. **38**: Candlestick pattern features (independent; adds 15 identifiers to feature vector) ✅ implemented
23. **41**: HTF structure cascade (independent; adds 12 daily anchor fields) ✅ implemented
24. **43**: Signal ledger & outcome reconciler (Signal→Risk Policy→Execution Adapter architecture; statistical capital gates; requires 42 for stop/target fields) ✅ implemented
25. **44**: Setup event generator (frozen feature snapshots/hashes + template provenance for setup/outcome analysis) ✅ implemented

**Stratum B — Strategy templates (after Stratum A merges):**
26. **40**: Compression breakout template (requires 38 for is_impulse_candle, is_inside_bar) ✅ implemented
27. **42**: Level-anchored stops (requires 38 for candle_low anchor, 41 for htf_daily_low anchor) ✅ implemented

**Stratum B2 — Trade management extensions (after Stratum B):**
28. **45**: Adaptive trade management (R-multiple state machine, trailing/partials) ✅ implemented

**Stratum C — Market intelligence (after Stratum B validates in paper trading):**
29. **39**: Universe screener ✅ implemented (`b7c5d5f`) — **now in paper trading validation gate**.
    - Amendment (2026-02-21): Screener pre-selects `template_id` deterministically from
      composite score breakdown. Sniffer tuning via `SCREENER_COMPRESSION_WEIGHT` is the
      user's strategic lever. See [39-universe-screener.md](39-universe-screener.md).

> **Why this order:** Runbook 37 is a correctness bug with zero-cost fix — ship immediately. Candlestick features (38) and HTF structure (41) are additive with no risk of regression and unlock everything downstream. The breakout template (40) and level-anchored stops (42) depend on those features and should be validated together via a paper trading backtest before the universe screener (39) is enabled — you want a working strategy before adding autonomous instrument selection.

> **Paper trading as the validation gate:** Unlike prior runbooks that were validated via backtest, Runbook 39 (universe screener) can only be meaningfully validated via paper trading. The screener surfaces real-time anomalies; no historical simulation can test whether it would have identified the right instrument at the right time. Plan for a 30-day paper trading period after 39 ships before enabling live capital routing.

### Phase 7 — Template-Bound Instrument Strategy (new direction, 2026-02) [PARTIAL — R48 ✅, R46/47 PENDING]

Runbooks 46–47 complete the shift from "LLM authors trigger rules" to "LLM selects and
parameterizes a known template." The vector store retrieval infrastructure is already
active (`STRATEGY_VECTOR_STORE_ENABLED=true`); these runbooks close the gap between
retrieval-as-hints and retrieval-as-binding.

> **Current status (`aa90202`):** R48 (research budget) is fully implemented and merged.
> R46 and R47 runtime contracts remain pending their gates: R46 is gated on R39 running
> in paper trading; R47 is gated on R46 routing accuracy ≥ 80%. Both gates can be
> assessed during the R39 validation window.

> **Architecture context:** The `vector_store/strategies/` docs are retrieval targets.
> The `prompts/strategies/*.txt` files are the actual system prompts. The two systems are
> currently disconnected — retrieval finds a regime match but does not load the
> corresponding prompt. Runbook 46 wires them together. Runbook 47 enforces the contract.

**Stratum D — Template routing (after Stratum C paper trading validation):**
28. **46**: [Template-matched plan generation](46-template-matched-plan-generation.md) —
    Adds `compression_breakout.md` to vector store; wires retrieval to load the
    corresponding `prompts/strategies/*.txt` template automatically. Zero schema changes.
    _Gate: Runbook 39 screener running in paper trading._ `✅ implemented` (`dc5bb98`)

**Stratum E — Hard binding (after Stratum D validated):**
29. **47**: [Hard template binding](47-hard-template-binding.md) — Adds `template_id`
    to `StrategyPlan`; trigger compiler blocks triggers using identifiers outside the
    declared template's allowed set. Backwards compatible (Optional field, enforcement
    skipped when `template_id=None`).
    _Gate: Runbook 46 routing confirmed correct ≥ 80% of plan-generation days._ `[runbook authored; runtime implementation pending]`

**Stratum F — Research feedback loop (can run in parallel with Stratum D/E):**
30. **48**: [Research budget in paper trading](48-research-budget-paper-trading.md) —
    Separate capital pool + ledger for hypothesis testing in paper trading. Research
    trades tagged with `experiment_id` and `playbook_id`. `PlaybookOutcomeAggregator`
    writes validated stats to `vector_store/playbooks/*.md` `## Validation Evidence`
    sections. Judge gains `suggest_experiment` and `update_playbook` action types.
    _Can start immediately; does not block on Stratum D/E._ `✅ implemented` (`aa90202`)

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

### Phase 8 — Reasoning Agent Operating System (FinAgent-inspired, 2026-02) [DOCS-ONLY]

Runbooks 49–54 translate the paper insights into implementation contracts for this
codebase's existing strategist/judge architecture. The key framing is architectural:
we are not adding a new pretrained "foundation model"; we are upgrading the agent
system around multimodal inputs, reflection, memory, and evidence gating.

**Phase 8 status (2026-02-22):** Runbooks **49–56 are authored design contracts (`docs/branching/*.md`) but are not yet implemented in code** unless/until explicitly marked `✅ implemented` in this README or moved to `X-*`. This note applies to **Phase 8 runbooks only** (see the git-derived snapshot above for overall project progress).

**Stratum G — Inputs + memory substrate (ship first):**
31. **49**: `MarketSnapshot` definition (single source of truth for every reasoning tick) `[docs-only]`
32. **55**: Deterministic regime fingerprint + transition detector (policy-loop trigger keystone) `[docs-only]`
33. **51**: Diversified memory store + retrieval (wins/losses/failure-modes) `[docs-only]`

**Stratum H — Decision structure + reflection (after Stratum G):**
34. **52**: Typed playbook definition with regime tags + expectation distributions `[docs-only]`
35. **56**: Structural target + activation refinement enforcement (compiler + deterministic mapping layer) `[docs-only]`
36. **50**: Dual reflection templates (policy-level fast reflection, high-level batch review) `[docs-only]`
37. **53**: Judge validation rules with memory-backed rejection criteria `[docs-only]`

**Stratum I — Operations control plane (after Stratum H starts landing):**
38. **54**: Reasoning-agent cadence rules (three-tier scheduling, policy triggers/heartbeats, slow-loop thresholds) `[docs-only]`

> **Why this order:** `MarketSnapshot` (49) establishes normalized inputs, but policy-loop
> cadence is only trustworthy once the deterministic regime transition detector (55)
> exists. Diversified memory retrieval (51) then uses that normalized regime structure for
> contrastive evidence. Typed playbooks (52) provide the constrained proposal surface and
> expectation distributions (holding time, MAE/MFE). Runbook 56 then hardens the
> deterministic enforcement layer between playbook schema and trigger engine (structural
> target selection + refinement mode mapping) before reflection/judge layers rely on it.
> Dual reflection (50) and judge validation (53) come after those inputs and enforcement
> contracts exist, otherwise they degrade into prompt-only rituals. Cadence rules (54)
> are codified after the transition detector and policy-loop triggers are concretely defined.

## Backlog Runbooks (_)
- [_per-instrument-workflow.md](_per-instrument-workflow.md): Per-instrument `InstrumentStrategyWorkflow` (one Temporal workflow per active symbol). Deferred until Runbooks 39+46+47 are validated via 30-day paper trading and open architectural questions (workflow ID namespace, multi-timeframe, judge routing) are resolved with operational evidence.
  - Routing note (draft ADR, 2026-02-24): use symbol-local judge feedback plus a shared deterministic portfolio control plane that broadcasts typed constraint envelopes (not direct portfolio-judge broadcasts to symbols). See [ADR-portfolio-judge-routing-and-control-plane.md](ADR-portfolio-judge-routing-and-control-plane.md).
- [_portfolio-control-plane.md](_portfolio-control-plane.md): Shared portfolio allocator / constraint coordinator for per-instrument workflows (risk budgets, concentration/correlation caps, broadcast constraints, reservation/release semantics).
- [_portfolio-monitor-and-reflection.md](_portfolio-monitor-and-reflection.md): Portfolio-level monitoring and slow-loop reflection layer (drawdown/concentration/correlation diagnostics, monitor-only alerts, shared portfolio recommendations).
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
- [21-emergency-exit-sensitivity.md](21-emergency-exit-sensitivity.md): Emergency exit sensitivity fix and calibration corrections from backtest ebf53879 batch.
- [22-exit-binding-prompt-gap.md](22-exit-binding-prompt-gap.md): Exit-category binding prompt guidance and related backtest quality corrections (implemented in Runbooks 21–27 batch).
- [23-hold-rule-calibration.md](23-hold-rule-calibration.md): Hold-rule guidance calibration to reduce degenerate normal-exit suppression.
- [24-judge-eval-flood.md](24-judge-eval-flood.md): Stale snapshot skip timer advancement / judge cadence flood fix.
- [25-trade-volume-deficit.md](25-trade-volume-deficit.md): Under-trading diagnostics and prompt/trigger-volume improvements from backtest quality batch.
- [26-risk-telemetry-accuracy.md](26-risk-telemetry-accuracy.md): Phantom risk telemetry fixes in judge snapshots / summaries.
- [27-stance-diversity.md](27-stance-diversity.md): Prompt/constraint improvements to increase defensive/wait stance usage.
- [28-judge-action-contract.md](28-judge-action-contract.md): Structured `JudgeAction` contract, TTL persistence, action events, and recommended-action routing.
- [29-judge-structured-multipliers.md](29-judge-structured-multipliers.md): Structured judge risk multipliers with clamps and text-parsing fallback.
- [30-judge-immediate-application.md](30-judge-immediate-application.md): Immediate intraday application of judge constraints to active trigger/risk engines.
- [31-judge-stance-enforcement.md](31-judge-stance-enforcement.md): Deterministic enforcement of judge `recommended_stance` at plan generation.
- [32-exit-binding-enforcement.md](32-exit-binding-enforcement.md): Compile-time exit binding enforcement / corrections in trigger compiler.
- [33-hold-rule-rejection.md](33-hold-rule-rejection.md): Compile-time stripping/rejection of degenerate hold rules.
- [34-expression-error-handling.md](34-expression-error-handling.md): Trigger-expression identifier validation / error handling hardening.
- [35-block-reason-normalization.md](35-block-reason-normalization.md): Granular block-reason normalization for judge-facing telemetry.
- [36-judge-action-dedup.md](36-judge-action-dedup.md): Judge action de-dup / supersession tracking in evaluation windows.
- [37-risk-budget-commit-actual.md](37-risk-budget-commit-actual.md): P0 fix — commit actual risk-at-stop to budget instead of theoretical cap.
- [38-candlestick-pattern-features.md](38-candlestick-pattern-features.md): Candlestick morphology features added to feature vector.
- [40-compression-breakout-template.md](40-compression-breakout-template.md): Compression breakout strategy template + prompt implementation.
- [41-htf-structure-cascade.md](41-htf-structure-cascade.md): HTF structure cascade fields (daily anchors and related context).
- [42-level-anchored-stops.md](42-level-anchored-stops.md): Level-anchored stop/target resolution at entry with direction-aware fixes.
- [43-signal-ledger-and-reconciler.md](43-signal-ledger-and-reconciler.md): Signal ledger, outcome reconciler, and signal provenance tracking.
- [44-setup-event-generator.md](44-setup-event-generator.md): Setup event generator with frozen feature snapshots/hashes and template provenance.
- [45-adaptive-trade-management.md](45-adaptive-trade-management.md): Adaptive trade management (R-multiple state machine, trailing/partials) implementation.

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

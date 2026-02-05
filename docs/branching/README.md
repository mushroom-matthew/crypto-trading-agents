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

### Phase 4 — Infrastructure expansion
12. **07**: AWS deploy / CI/CD
13. **08**: Multi-wallet (Phantom/Solana/EVM read-only + reconciliation)

> **Why this order differs from filenames:** The true blocker for the policy pivot is "anti-churn + continuity + clean replans," not just risk math. Learning Book isolation is trust infrastructure, not a feature — without it, experiments muddy PnL and interpretability is lost. Strategist simplification (01) is major but moves later because emergency exits and judge robustness are hard safety invariants that must be machine-enforced first. After those are green, Phase 1 policy integration (18) is mandatory; Phase 2 model integration (19) is optional and reversible.

## Backlog Runbooks (_)
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

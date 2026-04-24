# Agentic SOTA Alignment Plan
*Assessment date: 2026-04-23*

This document assesses the current strategist/judge architecture against two state-of-the-art references:

1. **[2508.17281v2]** *"From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users"* — a 2025 survey synthesising 68 benchmark datasets and identifying 10 future research directions for LLM-based agents.

2. **[2410.06304]** *"FG-PRM: Fine-Grained Process Reward Model for Mitigating Hallucinations in LLM-based Agents"* (EMNLP 2025) — a step-level hallucination classification and verification framework that out-performs coarse-grained reward models trained on 33× more data.

---

## 1. Current Architecture Snapshot

### 1.1 Strengths

| Mechanism | Current Implementation |
|---|---|
| **5-layer Judge gate** | Deterministic → Playbook → Memory → Cluster → Calibration (R53) |
| **Evidence-backed verdicts** | `JudgeValidationVerdict` cites `cited_episode_ids`, failure patterns |
| **Policy boundary enforcement** | `PolicyStateMachine` (IDLE / THESIS_ARMED / POSITION_OPEN / HOLD_LOCK / INVALIDATED) prevents mid-execution replanning |
| **Contrastive memory** | `MemoryRetrievalService` hybrid similarity, recency decay, diversity constraint (R51) |
| **Bidirectional feedback** | `JudgeConstraints` → disabled_trigger_ids, risk_multiplier, regime correction propagate back to strategist |
| **Cadence gating** | `PolicyLoopGate` single-flight + cooldown + heartbeat/event requirements (R54) |
| **Low-level reflection** | `PolicyLevelReflectionService`: deterministic invariant checks + optional LLM coherence (R50) |
| **Confidence grading** | A/B/C trigger priority; dedup confidence-override threshold |
| **Playbook eligibility** | `PlaybookRegistry.list_eligible` filters by regime + HTF direction (R62) |

### 1.2 Architectural Model (LOMAR Mapping)

The survey's five-component **LOMAR** framework maps as follows:

| LOMAR | This System |
|---|---|
| **L — LLM** | `llm_client.py` (OpenAI Responses API, low temperature) |
| **O — Objective** | `StrategyPlan` (regime, stance, triggers, sizing_rules, allowed_symbols) |
| **M — Memory** | `EpisodeMemoryStore` + `MemoryRetrievalService` + `JudgeAgentWorkflow` prompt history |
| **A — Action** | `TriggerEngine` → `place_mock_order` → `ExecutionLedgerWorkflow` |
| **R — Rethink** | `JudgePlanValidationService` + `JudgePlanRevisionLoopOrchestrator` (max 2 revisions) |

---

## 2. Gap Analysis Against the Survey

### Gap 1 — No Chain-of-Thought Reasoning Trace

**What the survey says**: ReAct-style "interleaving structured reasoning with action" and Chain-of-Thought "stepwise decomposition" are the baseline for trustworthy autonomous decisions. Tree-of-Thought extends this with multi-branch exploration. Without an explicit scratchpad the LLM's reasoning is opaque; the system cannot verify *how* a trigger was derived.

**Current state**: The strategist is called once and returns a `StrategyPlan` JSON blob. The `rationale` field on `TriggerCondition` is informational only and is not validated. The judge gate has no access to reasoning steps, only the final plan.

**Impact**: Judge validation cannot catch *reasoning pathway* errors—only structural and schema violations. FG-PRM's `Logical Inconsistency` and `Context Inconsistency` types (steps that contradict prior derivations or the provided market context) are invisible.

---

### Gap 2 — Single-Shot Plan Generation (No Multi-Candidate Ranking)

**What the survey says**: Best-of-N selection via a verifier is the dominant improvement technique for generation quality. FG-PRM's verification task explicitly selects among N candidates. Tree-of-Thought explores multiple branches and prunes. Contrastive reasoning optimises through comparison.

**Current state**: `llm_client.py` makes one LLM call, retries twice on failure, falls back to a deterministic minimal plan. There is no mechanism to generate 2–5 candidate plans, score them, and select the highest-confidence one.

**Impact**: The system always takes the first coherent output regardless of whether better alternatives exist. Revision under `JudgePlanRevisionLoopOrchestrator` is corrective (fix a broken plan) rather than comparative (pick the best of several).

---

### Gap 3 — No Persistent Reflexion Loop

**What the survey says**: The Reflexion pattern—"retrospectively evaluate prior outputs, learn from feedback, and refine subsequent decisions"—is a named architectural primitive. Agents that accumulate error-aware summaries of past episodes and feed them as structured critique back to the planner close the self-improvement loop.

**Current state**: `EpisodeMemoryRecord` captures outcome, failure modes, and regime fingerprint but the *reasoning* behind the LLM's plan decisions is not stored. When the judge issues a revision request (`JudgePlanRevisionRequest`) containing `failing_criteria` and `cited_failure_patterns`, those observations are used for one revision cycle but are not persisted as a reflexion summary for future plan generations.

**Impact**: The system cannot accumulate structured self-critique over time. The judge evaluates fresh each session without a running account of *why* the strategist made repeated errors on similar setups.

---

### Gap 4 — No Debate / Ensemble Deliberation

**What the survey says**: Multi-agent debate—"agents argue opposing views to reach consensus"—is a validated mechanism for improving decision quality in autonomous systems, especially under uncertainty. AutoGen-style "structured role coordination" with a challenger agent provides independent verification.

**Current state**: There is a single strategist and a single judge in series. The judge validates but does not independently generate an alternative hypothesis. There is no "challenger" path that argues the opposite regime/stance.

**Impact**: Confirmation bias from the strategist can propagate unchallenged if the judge's memory bundle and heuristics happen to weakly support the same view.

---

### Gap 5 — No Formal Uncertainty Quantification

**What the survey says**: "Uncertainty-aware planning" and "verifiable reasoning of LLMs" are listed as two of the ten priority future research directions. The review notes that current systems lack mechanisms for provable self-improvement with tractable uncertainty estimates.

**Current state**: Trigger confidence is a discrete A/B/C grade. `judge_confidence_score` on `JudgeValidationVerdict` is a fixed float (0.55–0.90) determined by finding class, not empirically calibrated. There is no per-field uncertainty estimate on `StrategyPlan` outputs (e.g., "how confident is the LLM that the regime is `bull` vs `range`?").

**Impact**: The system cannot adaptively reduce position sizing or require a higher evidence bar proportional to LLM uncertainty. Risk management is rule-based rather than uncertainty-proportional.

---

### Gap 6 — Limited Transparency / Human-Agent Symbiosis

**What the survey says**: Future systems must support "deepening the human-agent symbiosis: personalization, proactivity, and trust" through transparent reasoning. This means the operator should be able to inspect not just the plan output but *why* the model reached it.

**Current state**: The `rationale` field exists but is purely informational; no UI surface exposes it. Operator visibility is limited to `JudgeConstraints` and score summaries. The revision loop's `failing_criteria` is not surfaced to human operators.

---

### Gap 7 — Reactive-Only Architecture (No Proactive Goal-Setting)

**What the survey says**: Moving toward "agents that autonomously set goals and decompose tasks" is one of the 10 research priorities. Proactive agents anticipate market regimes and pre-position rather than only responding to indicators crossing thresholds.

**Current state**: The execution agent is scheduled-nudge-driven (every 25s) and generates plans only when the `PolicyLoopGate` allows. Session intent (`SessionPlanner`, R76) partially addresses this by generating symbol/timeframe recommendations at session start, but mid-session goal reformulation in response to regime drift is not implemented.

---

### Gap 8 — Full Trigger-Set Overwrite on Replan (No Incremental Mutation)

**What is missing**: Every plan generation cycle produces a complete, self-contained `StrategyPlan` with a fresh `triggers: list[TriggerCondition]`. When the plan is accepted, `TriggerEngine` replaces its active trigger set wholesale. This is fine for a cold start, but it is semantically wrong when positions are open or when the prior plan's triggers represent accumulated deliberation about a specific setup — hours of hypothesis building discarded because a single replan cycle generated a slightly different condition set.

**Current state**: `TriggerEngine` loads triggers from the current plan's `triggers` list on each evaluation. There is no concept of a canonical trigger identity that persists across plan versions. Two triggers that express the same entry condition (e.g., `ETH-MR-L-001`) but differ by a single parameter are treated as entirely separate objects. Exit triggers are bound to `originating_plan_id` (R65), which protects cross-plan exit interference, but entry triggers are silently replaced.

**Two related sub-problems**:

1. **Overwrite during open positions**: If a position is open and the strategist replans (e.g., due to regime drift or a judge revision), the new plan's trigger list may not include the matching exit trigger for the open position — or may replace it with a differently-parameterised version. R65 partially mitigates this by pinning exits by plan ID, but new replans can still introduce duplicate or conflicting exit conditions.

2. **Trigger vocabulary recomposition**: Because triggers are free-form LLM outputs, semantically equivalent triggers are recomposed from scratch each cycle. `ETH-MR-L-001` in one plan and `ETH-mean-rev-long` in the next are identical in intent but different in identity. This prevents deduplication, deconfliction, and history tracking at the trigger level.

**What this suggests**:

- A **canonical trigger catalog** — a pre-defined vocabulary of trigger archetypes (e.g., `mean_reversion_long`, `breakout_long`, `momentum_long`, `emergency_exit`) with stable IDs and parameterised slots. The LLM selects from and instantiates catalog entries rather than composing free-form triggers.
- A **stateful trigger registry** (`TriggerRegistry`) that is mutated incrementally: the strategist outputs an **add/remove diff** (`triggers_to_add`, `triggers_to_remove`, `triggers_to_modify`) rather than a full replacement. The registry applies the diff, preserving triggers with open position bindings by default.
- The `PolicyStateMachine` state (`POSITION_OPEN`) becomes the guard: in `POSITION_OPEN`, only exit and emergency triggers may be modified; entry trigger mutations are queued until the position closes.

**Impact of not addressing**: The system behaves as a one-shot planner restarting from zero every 25 seconds, rather than as a persistent deliberative agent. Accumulated thesis confidence evaporates on every replan. The LLM's tendency to produce slightly different phrasings of the same trigger creates noise in deduplication, audit trails, and the hallucination scorer's fabrication check.

---

### Gap 9 — No Token-Level Uncertainty Signal (Logprob-Based Hallucination Detection)

**What is missing**: The OpenAI Responses API can return per-token log-probabilities (`logprobs=True`). These provide a direct probabilistic signal about which parts of the generated plan the model found uncertain — before any downstream verifier sees the output. This is distinct from the scratchpad confidence map (R93), which is the LLM's self-reported uncertainty, and from the deterministic hallucination scorer (R90), which checks structural validity. Logprob-based detection catches *fluent but uncertain* outputs: cases where the model produced a grammatically valid JSON value but with high token entropy, indicating it was guessing rather than deriving.

**Specific trading-domain relevance**: The highest-risk free-text fields in a `StrategyPlan` are numeric parameters — `stop_loss_pct`, `target_pct`, `entry_price_pct`, `min_rr_ratio`. A logprob spike on a specific digit (e.g., the model generates `2.3` with low confidence where `1.5` was the trained prior) is a direct hallucination signal for that parameter. Similarly, a low logprob on the regime token (`"bull"` vs `"bear"`) is more reliable than the model's scratchpad self-assessment, since it reflects the model's actual internal uncertainty before any post-hoc rationalisation.

**Current state**: `llm_client.py` calls the API with `logprobs=False` (default). No logprob data is captured or consumed anywhere in the system.

**Proposed detection approach**: For the three highest-stakes fields — `regime`, `stop_loss_pct`, and `target_pct` — extract the mean log-probability of the tokens composing their JSON values. Threshold: if mean logprob < −1.5 (roughly, less than 22% probability per token averaged across the field value), flag as `HighEntropyField`. This is not itself a reject criterion but feeds into `PlanHallucinationReport` (R90) as a `Factual Inconsistency` signal and into `FieldUncertaintyScore` (R93) as a hardware-level cross-check on the model's self-reported confidence.

**Note on paper coverage**: The FG-PRM paper (Section 3) does not use logprobs; it trains a step-level reward model from preference data. Logprob-based uncertainty is an older technique (see "Semantic Uncertainty", Kuhn et al. 2023) that predates step-level reward models and is complementary — it operates at token resolution rather than step resolution and requires no training data. For a production system where labelled failure data is limited, it is a pragmatic zero-training signal to layer in before the heavier FG-PRM approach becomes viable.

---

## 3. FG-PRM Hallucination Confidence — Domain Adaptation

The FG-PRM paper defines six hallucination types at reasoning step granularity. These map directly to the trading domain as follows:

| FG-PRM Type | Trading Domain Equivalent | Current Coverage |
|---|---|---|
| **Context Inconsistency** | Plan contradicts injected market data (e.g., plan says regime=bull when RSI=85 and LLM was given overbought context) | Partial — judge checks regime vs playbook eligibility but not raw indicator context |
| **Logical Inconsistency** | Triggers contradict each other within the same plan (e.g., long trigger + short trigger on same symbol without explicit condition guards) | Partial — sanitizer drops unknowns but no cross-trigger logical consistency check |
| **Instruction Inconsistency** | Plan violates explicit judge constraints (disabled_categories, max_trades_per_day) | Partial — trigger engine enforces disabled_trigger_ids but downstream, not at plan-generation time |
| **Calculation Error** | Incorrect R:R arithmetic, stop/target distance computation, position sizing (e.g., stop_loss_pct outside allowed band) | Partial — R:R gate (R69) enforces min_rr_ratio but doesn't detect raw arithmetic errors in the plan JSON |
| **Factual Inconsistency** | Incorrect symbol reference, price regime mismatch, wrong indicator value cited in rationale | **None** — rationale is not verified against actual indicator values |
| **Fabrication** | Invented support/resistance levels not present in market data, nonexistent playbook IDs | Partial — playbook_id validated against PlaybookRegistry but support/resistance not cross-checked |

The key insight: FG-PRM operates at *step* granularity. In the trading context, each section of a `StrategyPlan` JSON is analogous to a reasoning step:
- `regime` declaration
- each `TriggerCondition` entry
- `risk_constraints` block
- `allowed_symbols` / `allowed_directions` block

A fine-grained verifier scores each of these sections rather than the plan as a whole.

---

## 4. Implementation Roadmap

The following runbooks are ordered by dependency and impact. Each targets a specific gap from Sections 2–3.

---

### R88 — Strategist Chain-of-Thought Scratchpad

**Gap addressed**: Gap 1 (no reasoning trace)

**Scope**: Add a structured scratchpad section to the LLM response before the JSON plan block. The scratchpad is a short (4–8 sentence) natural-language derivation that the judge can later inspect for logical consistency.

**Changes required**:
1. `prompts/llm_strategist_simple.txt` — add instruction: respond with `<reasoning>…</reasoning>` block followed by the JSON plan; describe which indicators drove regime choice, which playbook was selected and why, and what the primary risk is.
2. `agents/strategies/llm_client.py::_extract_plan_json()` — parse reasoning block before JSON; store it in a new `_scratch` key on the raw response dict (not on `StrategyPlan`).
3. `agents/strategies/plan_provider.py` — persist `scratchpad_text` on the `plan_generated` event payload for downstream consumption.
4. `schemas/llm_strategist.py` — add `scratchpad: str | None` field to `StrategyPlan` (optional, informational).

**Acceptance criteria**: `plan_generated` events contain non-empty `scratchpad_text`; existing test suite passes without modification.

---

### R89 — Multi-Candidate Plan Generation + Best-of-N Selection

**Gap addressed**: Gap 2 (single-shot generation)

**Scope**: Generate N=3 candidate plans per policy loop invocation; score each with the existing judge validation service; select the highest-confidence approved plan. Fall back to best-of-N by heuristic score if none approve.

**Changes required**:
1. `agents/strategies/llm_client.py` — add `generate_candidates(n: int = 3)` method; calls `generate_plan()` N times with temperature slightly varied (0.10 / 0.15 / 0.20) to produce diverse outputs.
2. `agents/strategies/plan_provider.py::get_plan()` — add `use_best_of_n: bool` (env-gated `STRATEGIST_BEST_OF_N=3`); call `generate_candidates()` when enabled; run each candidate through `JudgePlanValidationService.validate_plan()`; rank by `judge_confidence_score`; select highest-confidence approved plan (or highest score if all revise/reject).
3. `services/judge_validation_service.py` — expose `batch_validate(plans: list[StrategyPlan]) -> list[JudgeValidationVerdict]` for efficiency.
4. Token budget: N=3 adds ~2× token cost (candidates are smaller than initial context); gate behind `STRATEGIST_BEST_OF_N` env var defaulting to `1`.

**Acceptance criteria**: With `STRATEGIST_BEST_OF_N=3`, `plan_generated` event contains `candidate_count=3`, `selected_candidate_index`, `candidate_confidence_scores`.

---

### R90 — Trading-Domain Hallucination Taxonomy + Section Scorer

**Gap addressed**: Gap 3 + FG-PRM (Section 3)

**Scope**: Implement a deterministic fine-grained hallucination scorer that classifies each `StrategyPlan` section against the 6-type taxonomy adapted for trading. This is the domain-adapted analogue of FG-PRM's step-level classifiers—implemented deterministically rather than via a trained reward model, since the trading domain has ground truth accessible at plan-generation time.

**Hallucination type → detection rule**:

| Type | Check | Severity |
|---|---|---|
| **Context Inconsistency** | `plan.regime` conflicts with `LLMInput.assets[*].regime_assessment` majority vote | REVISE |
| **Logical Inconsistency** | Two triggers on same symbol have opposite directions without mutually exclusive entry conditions | REVISE |
| **Instruction Inconsistency** | `plan.triggers[*].category` ∈ `judge_constraints.disabled_categories` | REJECT |
| **Calculation Error** | `stop_loss_pct` > `risk_params.max_position_risk_pct` or computed R:R < `min_rr_ratio` | REVISE |
| **Factual Inconsistency** | `plan.allowed_symbols` contains symbol not present in `LLMInput.assets` | REVISE |
| **Fabrication** | `plan.playbook_id` not in `PlaybookRegistry` or `plan.template_id` not in vector store | REJECT |

**Changes required**:
1. New file `services/plan_hallucination_scorer.py` — `PlanHallucinationScorer` class; `score(plan, llm_input, judge_constraints) -> PlanHallucinationReport`; report contains per-section findings (section_id, hallucination_type, severity, detail).
2. `schemas/judge_feedback.py` — add `PlanHallucinationReport`, `SectionHallucinationFinding`.
3. `services/judge_validation_service.py` — call `PlanHallucinationScorer.score()` as a pre-pass (Layer 0) before existing 5-layer gate; inject `REJECT` findings directly as structural violations.
4. `agents/strategies/plan_provider.py` — log hallucination report on `plan_generated` event.

**Acceptance criteria**: Unit tests cover all 6 types; REJECT findings propagate to structural_violation in `JudgeValidationVerdict`; existing revision loop handles them correctly.

---

### R91 — Reflexion Memory: Structured Self-Critique Persistence

**Gap addressed**: Gap 3 (no persistent reflexion loop)

**Scope**: When a plan completes its episode (position closed), synthesise a structured `PlanReflexionSummary` that captures *why* the plan was revised (or not), what the LLM's stated reasoning was (from scratchpad R88), and how that aligned with actual outcome. Persist this as a new memory record type and inject it into future strategist prompts.

**Changes required**:
1. `schemas/episode_memory.py` — add `PlanReflexionSummary` (plan_id, scratchpad_excerpt, revision_count, revision_reasons, final_verdict, outcome_class, regime_at_entry, failure_modes, lesson_text: str); add `reflexion_summaries: list[str]` field to `EpisodeMemoryRecord`.
2. `services/episode_memory_service.py::build_episode_record()` — when `scratchpad_text` is available on the plan event, extract key sentences and populate `reflexion_summaries`.
3. `services/memory_retrieval_service.py` — include top-3 `reflexion_summaries` from loser episodes in `DiversifiedMemoryBundle.failure_mode_patterns`; surface them with a `REFLEXION:` prefix.
4. `agents/strategies/llm_client.py::_build_system_prompt()` — inject retrieved `REFLEXION:` items as a new `## Lessons from Similar Episodes` block (after eligible playbooks block, before schema).
5. `tools/paper_trading.py::build_episode_activity` — pass `scratchpad_text` from plan event when building episode record.

**Acceptance criteria**: After a losing trade, next plan generation for a similar regime contains non-empty `## Lessons from Similar Episodes` block; losing episode `reflexion_summaries` are non-empty.

---

### R92 — Challenger Debate Pass (Two-Hypothesis Deliberation)

**Gap addressed**: Gap 4 (no debate/ensemble deliberation)

**Scope**: After the primary strategist generates a plan, generate a single "challenger plan" that argues the opposite or a defensive stance, then run both through the judge's hallucination scorer and confidence calibration. Use the comparison as an explicit signal: if the challenger plan scores higher or equally, require a higher evidence bar (reduce `judge_confidence_score` floor for approval).

**Changes required**:
1. `agents/strategies/llm_client.py` — add `generate_challenger(primary_plan, llm_input) -> StrategyPlan`; prompt instructs the LLM to explicitly argue the opposing regime or a flat/defensive stance with the same input.
2. `services/plan_deliberation_service.py` (new file) — `DeliberationService.deliberate(primary, challenger, llm_input, judge_constraints) -> DeliberationVerdict`; runs both through `PlanHallucinationScorer`; computes `confidence_delta = primary_score - challenger_score`; outputs: `verdict` (primary_wins | challenger_wins | inconclusive), `confidence_margin`, `divergence_points` (fields where plans differ most).
3. `services/judge_validation_service.py` — accept optional `DeliberationVerdict`; if `inconclusive` or `challenger_wins`, escalate approval threshold (require `weakly_supported` → `supported`).
4. `agents/strategies/plan_provider.py` — gate behind `STRATEGIST_DEBATE=true` env var (default off for now); log `deliberation_verdict` on plan_generated event.

**Acceptance criteria**: With `STRATEGIST_DEBATE=true`, plan_generated event contains `deliberation_verdict`; unit test verifies challenger plan triggers escalated approval threshold.

---

### R93 — Per-Field Uncertainty Quantification

**Gap addressed**: Gap 5 (no formal uncertainty quantification)

**Scope**: Attach calibrated uncertainty scores to the three highest-impact plan fields (`regime`, `stance`, `allowed_directions`) using a lightweight ensemble approach: ask the LLM to express its confidence in those fields within the scratchpad (R88), and cross-reference against the memory bundle cluster win-rate to produce a combined `FieldUncertaintyScore`.

**Changes required**:
1. `prompts/llm_strategist_simple.txt` — extend scratchpad instruction: include a `confidence_map` JSON object at end of `<reasoning>` block: `{"regime": 0.0–1.0, "stance": 0.0–1.0, "allowed_directions": 0.0–1.0}`.
2. `agents/strategies/llm_client.py` — parse `confidence_map` from scratchpad; store on raw response.
3. `services/plan_hallucination_scorer.py` — add `uncertainty_pass(plan, confidence_map, memory_bundle)`: cross-references LLM-stated confidence with cluster win-rate; high LLM confidence + unsupported cluster → `FieldUncertaintyFinding(severity=REVISE, field="regime", lm_confidence=0.9, cluster_support=0.28)`.
4. `schemas/llm_strategist.py` — add `field_uncertainty: dict[str, float] | None` to `StrategyPlan`.
5. `services/judge_feedback_service.py::build_guidance_vector()` — scale risk_multiplier down proportionally to `mean(field_uncertainty.values())` when uncertainty is high: `risk_multiplier *= max(0.6, mean_confidence)`.

**Acceptance criteria**: Unit test: plan with `regime_confidence=0.4` + unsupported cluster produces `risk_multiplier ≤ 0.8`.

---

### R94 — Proactive Intra-Session Goal Reformulation

**Gap addressed**: Gap 7 (reactive-only architecture)

**Scope**: Extend `CadenceGovernor` (R77) to detect regime drift mid-session and trigger a goal reformulation event — a lightweight session-intent regeneration that updates `SessionState.session_intent_dict` without requiring a full strategist plan cycle.

**Changes required**:
1. `schemas/reasoning_cadence.py` — add `RegimeDriftSignal` (prior_regime, current_regime, confidence, detected_at_bar); add `REGIME_DRIFT_THRESHOLD: float = 0.35` to `CadenceConfig`.
2. `services/cadence_governor.py` — add `detect_regime_drift(prior_fingerprint, current_fingerprint) -> RegimeDriftSignal | None`; compare cosine similarity; if `< 1 - REGIME_DRIFT_THRESHOLD` emit signal.
3. `tools/paper_trading.py` — on each bar's `generate_strategy_plan_activity`, call `detect_regime_drift`; if drift detected and THESIS_ARMED not active, call `generate_session_intent_activity` with `force_refresh=True`.
4. `schemas/session_intent.py` — add `drift_triggered: bool` field on `SessionIntent`.
5. Log drift events to agent logger for operator visibility.

**Acceptance criteria**: Regime flip from `bull` → `bear` mid-session triggers fresh `SessionIntent` within 2 plan cycles; unit test covers fingerprint-based drift detection.

---

### R95 — FG-PRM-Style Aggregate Confidence Score + Operator Transparency Surface

**Gap addressed**: Gap 6 (transparency) + completes FG-PRM integration

**Scope**: Aggregate the per-section scores from R90 (`PlanHallucinationReport`) and R93 (`FieldUncertaintyScore`) into a single `PlanConfidenceScore` analogous to FG-PRM's log-sum aggregate reward. Expose this score in the Ops API for operator inspection. Instrument the judge verdict path to surface the `scratchpad`, `hallucination_findings`, and `confidence_margin` (from R92) in operator-visible telemetry.

**Changes required**:
1. `services/plan_hallucination_scorer.py` — add `aggregate_score(report: PlanHallucinationReport, uncertainty: dict) -> PlanConfidenceScore`; score = `Σ log(1 - p_hallucination_i)` over all sections (log-sum of non-hallucination probabilities); maps directly to FG-PRM's R_Φ(x,y) formula.
2. `schemas/judge_feedback.py` — add `PlanConfidenceScore` (aggregate_log_reward: float, per_section_scores: list, interpretation: str).
3. `ops_api/routers/paper_trading.py` (or new `ops_api/routers/plan_audit.py`) — add `GET /plan-audit/{plan_id}` returning `scratchpad`, `hallucination_report`, `deliberation_verdict`, `confidence_score`, `revision_history`.
4. Persist `PlanConfidenceScore` on `plan_generated` event payload.
5. `services/judge_validation_service.py` — include `plan_confidence_score` in approval decision: if `aggregate_log_reward < LOG_REWARD_FLOOR` (env `PLAN_LOG_REWARD_FLOOR=-2.0`), escalate to REVISE.

**Acceptance criteria**: `GET /plan-audit/{plan_id}` returns all fields; plan with ≥2 hallucination findings produces `aggregate_log_reward` below floor; event payload contains score.

---

### R96 — Canonical Trigger Catalog + Incremental Trigger Registry

**Gap addressed**: Gap 8 (full trigger-set overwrite; free-form trigger recomposition)

**Scope**: Replace the current free-form trigger list in `StrategyPlan` with a catalog-based mutation model. The LLM selects from a pre-defined trigger archetype vocabulary and outputs a diff (add/remove/modify) against the active registry. The registry applies the diff, guarded by `PolicyStateMachine` state.

**Canonical trigger catalog** (`schemas/trigger_catalog.py`):

| Archetype ID | Direction | Category | Required Parameters |
|---|---|---|---|
| `mean_reversion_long` | long | mean_reversion | symbol, stop_loss_pct, target_pct |
| `mean_reversion_short` | short | mean_reversion | symbol, stop_loss_pct, target_pct |
| `breakout_long` | long | breakout | symbol, breakout_level, stop_loss_pct |
| `breakout_short` | short | breakout | symbol, breakout_level, stop_loss_pct |
| `momentum_long` | long | momentum | symbol, entry_condition, stop_loss_pct |
| `momentum_short` | short | momentum | symbol, entry_condition, stop_loss_pct |
| `emergency_exit` | either | emergency_exit | symbol |
| `profit_take_exit` | either | exit | symbol, target_price_abs |
| `stop_loss_exit` | either | exit | symbol, stop_price_abs |

Each instantiation gets a stable ID: `{archetype_id}:{symbol}:{session_id[:8]}` — deterministic and deduplicable.

**Changes required**:
1. `schemas/trigger_catalog.py` (new) — `TriggerArchetype` enum, `TriggerInstance` (archetype_id, instance_id, params, priority, state: active/pending/removed), `TriggerDiff` (to_add: list, to_remove: list[instance_id], to_modify: list).
2. `schemas/llm_strategist.py` — replace `triggers: list[TriggerCondition]` with `trigger_diff: TriggerDiff | None` (backward-compatible: if absent, treat all triggers as `to_add` to an empty registry — preserves existing plan format).
3. `services/trigger_registry.py` (new) — `TriggerRegistry`: mutable dict `{instance_id: TriggerInstance}`; `apply_diff(diff, policy_state)` — in `POSITION_OPEN`, raises `PositionProtectedTriggerMutation` if diff attempts to remove/modify an exit trigger bound to the open position; `list_active() -> list[TriggerInstance]`.
4. `agents/strategies/trigger_engine.py` — accept `TriggerRegistry` instead of raw trigger list; `_evaluate_trigger()` maps `TriggerInstance` → existing `TriggerCondition` via archetype parameter mapping.
5. `prompts/llm_strategist_simple.txt` — replace trigger schema section with catalog reference: LLM receives `ACTIVE_TRIGGERS` block listing current registry state and is instructed to output `trigger_diff` JSON (add/remove/modify), not a full trigger list. Include policy state guard note: "if POSITION_OPEN, do not remove or modify exit triggers".
6. `agents/strategies/plan_provider.py` — pass current `TriggerRegistry` state into `LLMInput` before calling `llm_client`; apply returned `trigger_diff` to registry after judge approval.

**Acceptance criteria**: With an open ETH position, a mid-session replan does not remove or replace the `stop_loss_exit:ETH-USD:*` trigger; unit test verifies `PositionProtectedTriggerMutation` is raised when diff attempts removal; registry state serialises/deserialises deterministically for Temporal CaN.

---

### R97 — Logprob-Based Token-Level Uncertainty Extraction

**Gap addressed**: Gap 9 (no token-level uncertainty signal)

**Scope**: Enable `logprobs=True` on the strategist LLM call and extract per-field mean log-probability for the three highest-stakes plan fields. Feed results into the existing `PlanHallucinationReport` (R90) and `FieldUncertaintyScore` (R93) pipelines as a hardware-level cross-check.

**Changes required**:
1. `agents/strategies/llm_client.py` — add `logprobs=True` to Responses API call (note: OpenAI Responses API supports `logprobs` on `output[*].content[*].logprobs`); extract token-level logprobs from response.
2. New helper `services/logprob_extractor.py` — `extract_field_logprobs(response_tokens, plan_json_str) -> dict[str, float]`: finds JSON value token spans for `regime`, `stop_loss_pct`, `target_pct` by character offset alignment; computes mean logprob per field; returns `{"regime": -0.42, "stop_loss_pct": -1.87, "target_pct": -0.91}`.
3. `services/plan_hallucination_scorer.py` — add `logprob_pass(field_logprobs: dict[str, float]) -> list[SectionHallucinationFinding]`; threshold `< -1.5` per field → `Factual Inconsistency` finding with `severity=REVISE`; include raw logprob in `detail` for operator transparency.
4. `agents/strategies/llm_client.py::generate_plan()` — return logprob dict alongside plan; store in `_scratch` raw response dict.
5. `agents/strategies/plan_provider.py` — thread logprob dict into `plan_generated` event payload as `field_logprobs`.
6. `schemas/llm_strategist.py` — add `field_logprobs: dict[str, float] | None` to `StrategyPlan` (informational; not used by trigger engine).

**Acceptance criteria**: `plan_generated` event contains non-null `field_logprobs`; unit test mocks a response where `stop_loss_pct` tokens have mean logprob −2.1 and verifies a `Factual Inconsistency` finding is emitted; logprob extraction handles token boundary edge cases (multi-token numbers) without raising.

---

## 4b. Third Reference — LLM-as-a-Judge Survey [2411.15594]

*"A Survey on LLM-as-a-Judge"* (Gu et al., 2024) is the most comprehensive taxonomy of LLM-as-evaluator patterns published to date. It is directly relevant to this system's judge architecture and introduces several calibration concerns that are not addressed by the two primary references above.

### Relevance to This Architecture

The current 5-layer judge gate (`JudgePlanValidationService`) is a **pointwise scorer** — it produces a single `judge_confidence_score` (0.55–0.90) for a single candidate plan. The survey's taxonomy reveals this as one of the simplest judge configurations, with documented failure modes that affect scoring reliability in exactly the scenarios this system faces (regime uncertainty, contradictory memory, high market volatility).

### Key Survey Findings and Their Trading-Domain Implications

| Bias Type | Survey Definition | Trading-Domain Manifestation | Current Exposure |
|---|---|---|---|
| **Positional bias** | Judge favours the first or last item in a list | Judge gate evaluates triggers in list order; earlier triggers in `StrategyPlan.triggers` may receive more favourable scrutiny | Medium — trigger list is LLM-ordered |
| **Verbosity bias** | Longer, more elaborated outputs score higher regardless of quality | A plan with a long `rationale` field scores higher than a terse but equivalent plan | Medium — rationale is informational only, not currently scored |
| **Self-enhancement bias** | LLM judges favour outputs from the same model family | Judge and strategist are both GPT-4-class models; judge may systematically approve plans from the same model that produced them | High — same model family throughout |
| **Anchoring bias** | Judge's initial impression anchors subsequent scoring | After R88 scratchpad is injected into the judge gate, the scratchpad may anchor the verdict regardless of structural findings | Prospective (R88 not yet implemented) |

### Recommendations Derived from the Survey

**1. Pairwise comparison over pointwise scoring for revision decisions**: The survey demonstrates that pairwise comparisons ("is plan A better than plan B?") are more reliable than pointwise scores ("how good is this plan on a 0–1 scale?"). This directly supports R92 (challenger debate): rather than scoring the challenger plan independently, ask the judge to compare primary vs challenger head-to-head. This sidesteps calibration errors in absolute scoring and produces a more reliable `confidence_delta`.

**2. Reference-based judging using episode memory**: The survey's strongest result is that reference-based evaluation (comparing against known-good examples) dramatically outperforms reference-free scoring. The current judge gate already has access to `DiversifiedMemoryBundle` — this memory is used for contradiction detection (layer 3) but not as an explicit reference set. The survey suggests the judge prompt should include the top-1 memory episode as an explicit positive reference: "this plan should score at least as well as the following approved episode." This is a low-cost improvement to R53 that no new runbook is required for — a prompt amendment.

**3. Judge calibration via historical approval rates**: The survey recommends tracking `judge_confidence_score` distributions over time and recalibrating thresholds when the approval rate departs from a target base rate. Currently, thresholds are static (0.55–0.90). Adding a rolling 7-day approval rate to `JudgeAgentWorkflow` state and using it to adjust the `approval_threshold` dynamically is a lightweight calibration mechanism aligned with the survey's recommendations. This can be folded into R95 (ops surface) without a separate runbook.

**4. Ensemble judging via scratchpad + structural scorer**: The survey's finding that ensemble judges outperform single judges maps directly to combining R88 (scratchpad-derived qualitative verdict) + R90 (deterministic structural verdict) + R97 (logprob signal) into a weighted ensemble decision rather than a waterfall gate. This is the underlying logic of R95's log-sum aggregate score and validates the architectural choice.

### What the Survey Does Not Cover

The survey focuses on static text generation evaluation. It does not address:
- **Temporal grounding**: How judges should handle time-dependent facts (e.g., a regime assessment that was correct 10 minutes ago may be stale now). The current system's market snapshot hash mechanism (R49) handles this but is not validated against any published judge framework.
- **Durable execution context**: The survey assumes stateless judge calls. This system's judge operates within a Temporal workflow with full revision history — a capability that exceeds the survey's architecture.
- **Confidence propagation into downstream risk management**: The survey treats judge output as a terminal verdict. This system propagates `judge_confidence_score` into `risk_multiplier` and position sizing — a novel extension not described in any of the three reference works.

---

## 5. Priority and Dependencies

```
R88 (scratchpad)
  └── R90 (hallucination scorer) ─────────────────────────────┐
        └── R93 (uncertainty quantification)                   │
        └── R97 (logprob uncertainty) ──────────────────────────┤
              └── R95 (aggregate score + ops surface) ◄────────┘
  └── R91 (reflexion memory)
  └── R89 (multi-candidate) ─── R92 (debate)

R94 (regime drift / proactive) — independent
R96 (trigger catalog + registry) — independent (depends only on TriggerEngine)
```

**Recommended implementation order**:

| Phase | Runbooks | Rationale |
|---|---|---|
| **Phase A** | R88 + R90 | Foundation: scratchpad enables all downstream; hallucination scorer closes the largest trust gap at zero inference cost |
| **Phase B** | R91 + R93 + R97 | Memory closure + token-level signal: reflexion loop, uncertainty quantification, and logprob extraction improve both quality and risk management with no additional LLM calls |
| **Phase C** | R89 + R92 | Cost-increasing features: best-of-N and debate require 2–4× LLM calls; validate Phase A/B value first |
| **Phase D** | R94 + R95 | Operational: proactive reformulation and ops surface for transparency; can run in parallel with Phase C |
| **Phase E** | R96 | Structural: trigger catalog and incremental registry is the highest architectural lift; implement after Phase A/B validates plan quality improvements |

**Companion workstream note**:
`docs/analysis/TRADE_LIFECYCLE_REPLAY_ALIGNMENT_PLAN.md` should be treated as an
evidence-plane support track for the phases above. It does not alter the
runbook order. Its main purpose is to give R91 reflexion memory, R95 operator
transparency, and R96 trigger identity a stable lifecycle object
(`setup -> signal -> fill legs -> close`) rather than relying on heuristic
reconstruction.

---

## 6. Estimated Cost Impact

| Runbook | Token Cost Multiplier | Notes |
|---|---|---|
| R88 (scratchpad) | ~1.15× | ~4–8 reasoning sentences per plan |
| R89 (best-of-N=3) | ~3× | Gate behind env var; amortised by quality improvement |
| R90 (hallucination scorer) | ~1.0× | Deterministic; no LLM calls |
| R91 (reflexion) | ~1.1× | Adds one `## Lessons` block to prompt context |
| R92 (debate) | ~2.1× | One additional LLM call per policy loop |
| R93 (uncertainty) | ~1.0× | Parsed from existing scratchpad |
| R94 (proactive drift) | ~1.05× | Session intent call is lightweight |
| R95 (ops surface) | ~1.0× | Read path only |
| R96 (trigger catalog) | ~1.05× | Slightly smaller trigger diff vs full list; registry state adds ~200 tokens to context |
| R97 (logprob extraction) | ~1.0× | Logprobs returned with existing call; no additional tokens |

At the current cadence (one plan every ~25s nudge, gated by `PolicyLoopGate`), the net effective cost of running all phases is approximately **2.5–3.5×** today's LLM spend on plan generation. Phase A + B alone add ~1.3× with negligible latency. R96 and R97 add negligibly to cost.

---

## 7. What Is Already State-of-the-Art

The following capabilities are at or ahead of patterns described in the survey:

- **Multi-layer memory-backed validation** (5-layer judge gate with `DiversifiedMemoryBundle`) surpasses most published single-pass validation approaches.
- **Temporal durable execution** (all workflows are Temporal primitives with deterministic replay) is not addressed in the survey and represents a production-grade capability not present in academic baselines.
- **Exit binding enforcement** (R65: `originating_plan_id` pinning) prevents cross-plan interference — a subtle multi-step consistency mechanism absent from survey architectures.
- **Policy state machine with activation window telemetry** (R54) provides explicit lifecycle management that survey frameworks treat implicitly.
- **Playbook hypothesis testing with evidence accumulation** (R48, R79) is aligned with the survey's vision for "self-evolutionary learning" and exceeds published implementations.

These are genuine competitive advantages and should be preserved and emphasised in any external publication or benchmark evaluation.

---

## 8. Open Questions for Human Review

1. **Best-of-N temperature diversity** (R89): temperatures 0.10/0.15/0.20 are a heuristic; empirically calibrating diversity vs coherence trade-off will require backtest comparison runs.

2. **Challenger plan generation** (R92): the "argue opposite stance" instruction can produce degenerate plans (flat stance with no triggers) that trivially win on hallucination score. The debate prompt needs careful engineering to produce meaningful contrasting hypotheses, not just safe/null plans.

3. **FG-PRM trained model** vs deterministic scorer (R90): the deterministic approach proposed here covers the six types with rules; a learned classifier trained on historical plan failures could cover subtler semantic violations (e.g., rationale claiming a breakout setup when indicators show consolidation). This would require labelled training data from past sessions — a medium-term investment.

4. **Scratchpad faithfulness**: LLMs do not always produce scratchpads that accurately reflect their actual computation. The scratchpad in R88 should be treated as approximate evidence rather than ground truth. Cross-checking against the deterministic hallucination scorer (R90) and logprob extractor (R97) provides the real-world verification layer.

5. **Trigger catalog completeness** (R96): the 9-archetype catalog listed is a starting set. If the LLM generates a thesis that genuinely requires a trigger structure not present in the catalog, it will either hallucinate a catalog ID (detectable via R90 Fabrication check) or be unable to express the strategy. The catalog needs a `custom` escape hatch with an elevated scrutiny flag — otherwise it becomes an expressiveness constraint rather than a quality gate.

6. **Logprob token boundary alignment** (R97): multi-token numeric values (e.g., "2.3" tokenised as ["2", ".", "3"]) require averaging across sub-tokens to get a per-field score. The character offset alignment approach in `logprob_extractor.py` is fragile against tokeniser changes. An alternative is to force the model to produce JSON with a fixed schema where numeric fields are always single-token (e.g., by constraining to 1-decimal precision) — but this constrains plan expressiveness. The safer path is testing the offset alignment against the production tokeniser on a corpus of 1000 historical plan outputs before shipping R97.

7. **LLM-as-a-Judge self-enhancement bias** (from [2411.15594]): both strategist and judge use the same model family. The survey documents that this produces systematically inflated approval rates. The practical mitigation without switching model families is to add a **diversity seed** to the judge call (different system prompt structure than the strategist) and to track the empirical approval rate against the expected base rate derived from historical trade quality.

8. **Lifecycle linkage quality**: R91, R95, and R96 all implicitly assume a
high-fidelity executed trade lifecycle. In the current repo, some of this
linkage is available only through separate ledgers or reconstructed pairing.
The companion document
`docs/analysis/TRADE_LIFECYCLE_REPLAY_ALIGNMENT_PLAN.md` captures the
recommended lifecycle identity, replay, and pattern-retrieval work needed to
close that gap without changing the core runbook ordering above.

---

## 9. Cross-References

| Document | Role |
|---|---|
| `docs/analysis/AI_LED_PORTFOLIO_PHASE_PLAN.md` | Primary roadmap; phase ordering and dependency graph for Phases 5–10 |
| `docs/analysis/PROMPT_AUDIT_2026-04-23.md` | Prompt surface findings; prerequisite for Phase 5/R88 implementation |
| `docs/analysis/TRADE_LIFECYCLE_REPLAY_ALIGNMENT_PLAN.md` | Companion workstream; lifecycle identity and replay substrate for R91, R95, R96 |

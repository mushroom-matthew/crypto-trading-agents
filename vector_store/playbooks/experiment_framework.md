---
title: Experiment Framework
type: playbook
regimes: [bull, bear, range, volatile, uncertain]
tags: [research, experiment, hypothesis, validation, learning]
identifiers: []
hypothesis: "Meta-playbook: describes how research budget trades flow through hypothesis
  definition → attribution → outcome collection → playbook evidence update. Not a
  tradeable pattern; exists to provide the LLM and judge with the experiment contract."
min_sample_size: 0
playbook_id: experiment_framework
---
# Experiment Framework

This playbook describes the **research budget loop** — how the system tests hypotheses
about indicator patterns and accumulates evidence in the other playbooks.

## Research Budget vs. Main Budget

The paper trading session has two independent capital pools:

| | Main Budget | Research Budget |
|---|---|---|
| Purpose | Validated strategies; deploy known-good templates | Hypothesis testing; deliberately exploratory |
| Risk limits | Full `max_position_risk_pct` | Capped at `RESEARCH_BUDGET_FRACTION` of total capital |
| Ledger | `SessionState.cash` / `positions` | `SessionState.research.cash` / `research.positions` |
| Trade tagging | `is_research=False` | `is_research=True`, `experiment_id`, `playbook_id` |
| Outcome flow | Signal Ledger → Judge feedback | Signal Ledger → PlaybookOutcomeAggregator → playbook `## Validation Evidence` |
| Judge reaction | Replan / stance update | `suggest_experiment` or `update_playbook` action |

## Hypothesis Lifecycle

```
1. DRAFT     — ExperimentSpec created (by user or judge suggest_experiment action)
2. RUNNING   — Research trades tagged with experiment_id execute against research budget
3. PAUSED    — Auto-paused if max_loss_usd exceeded or user pauses
4. COMPLETED — min_sample_size reached; PlaybookOutcomeAggregator writes evidence to playbook
5. CANCELLED — Hypothesis abandoned (insufficient activity, regime changed)
```

## Judge Actions Related to Research

The judge may emit two new action types when live main-budget trades flounder:

- **`suggest_experiment`**: Judge proposes a new ExperimentSpec with a specific
  `hypothesis` and `playbook_id` target. Example: "Live BTC trades using rsi_14 < 30
  are losing. Propose experiment: is rsi_14 < 30 edge negative in the current vol_state=high
  regime? Test 20 trades in research budget."
- **`update_playbook`**: Judge proposes text changes to a playbook's `Notes` or
  `Validation Evidence` section based on accumulated evidence. Changes are surfaced to
  the operator for review via the Ops API before being applied to the `.md` file.

## PlaybookOutcomeAggregator Rules

The aggregator runs when any `SetupEvent` or `SignalEvent` with a `playbook_id` tag
reaches `outcome != null`. It:

1. Groups outcomes by `playbook_id`
2. Computes stats: `n_trades`, `win_rate`, `avg_r`, `median_bars_to_outcome`,
   plus any playbook-specific metrics listed in that playbook's `## Validation Evidence`
3. Writes stats to the playbook's `## Validation Evidence` section
4. Sets `status`:
   - `insufficient_data` — `n_trades < min_sample_size`
   - `validated` — `n_trades >= min_sample_size` AND primary hypothesis metric met
   - `refuted` — `n_trades >= min_sample_size` AND primary hypothesis metric NOT met
   - `inconclusive` — `n_trades >= min_sample_size` but effect size too small to decide

## Attribution Rules

A research trade is attributed to a playbook when:
1. The entry trigger's `entry_rule` contains an identifier listed in that playbook's
   `identifiers` field, AND
2. The current regime matches one of the playbook's `regimes`, OR
3. The `ExperimentSpec` explicitly sets `playbook_id`

When multiple playbooks match, attribution goes to the most specific one (smallest
identifier set intersection with the entry rule).

## Research Budget Trade Sizing

Research trades use a fixed fraction of the research budget, not the main risk engine:
- Default: 5% of research budget per trade (`RESEARCH_POSITION_FRACTION=0.05`)
- Maximum single-trade loss capped at `MetricSpec.max_loss_usd`
- No compounding between experiments (each experiment starts from its own capital slice)

## Validation Evidence
<!-- Auto-updated by PlaybookOutcomeAggregator. Do not edit manually. -->
status: not_applicable
n_trades: 0
win_rate: null
avg_r: null
median_bars_to_outcome: null
last_updated: null
judge_notes: null

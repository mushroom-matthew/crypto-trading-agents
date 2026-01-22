# Branching Runbooks

This folder contains branch-specific runbooks for parallel agents. Each runbook includes scope, key files, acceptance criteria, test gating, and explicit git commands.

## How To Use
1) Pick the branch runbook.
2) Follow the Git Workflow section exactly.
3) Run the Test Plan and paste the output into the Test Evidence section.
4) Commit only after test evidence is recorded.

## Active Branch Runbooks
- comp-audit-risk-core.md: Phase 0 risk correctness and budget integrity.
- comp-audit-trigger-cadence.md: Scalper cadence and signal serialization.
- comp-audit-indicators-prompts.md: Fast indicators, Donchian high/low, momentum prompts, compute optimizations.
- comp-audit-metrics-parity.md: Live/backtest metrics parity and annualization consistency.
- comp-audit-ui-trade-stats.md: Per-trade risk/perf stats in UI and APIs.
- aws-deploy.md: AWS infrastructure, secrets, and CI/CD wiring.
- multi-wallet.md: Multi-wallet architecture (Phantom/Solana/Ethereum read-only), reconciliation, UI.

## Later/Queued Runbooks
- later/policy-pivot-phase0.md: No-change replan guard and telemetry (prereq for policy pivot).
- later/judge-unification.md: Unified judge service with heuristics context.
- later/strategist-tool-loop.md: Read-only tool-call loop for strategist.
- later/scalper-mode.md: Full scalper mode feature set and comparison tooling.
- later/ui-unification.md: Optional UI unification enhancements.

## Notes
- If tests cannot be run locally, obtain user-run output and paste it into the Test Evidence section before committing.
- Coordinate with other agents to avoid overlapping files in parallel branches.

## Human Verification Evidence
- Follow the Human Verification section in each runbook and paste your observations into the Human Verification Evidence section before committing.

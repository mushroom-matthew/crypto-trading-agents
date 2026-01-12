# User Stories (Retail-Focused)

This document captures the ideal user profile and expected engagement patterns for the application as currently built, then translates those expectations into user stories.

## Ideal user (current state)
The ideal end user is a retail crypto trader who wants automated, rules-based strategies with strong risk controls and a safe paper-first workflow. They value deterministic behavior, backtests, and clear guardrails more than high-frequency or complex multi-venue tactics. They expect a hosted deployment (AWS) and interact through the Ops API and UI rather than running local stacks. The CLI user is the platform/operator who maintains the deployment and triggers administrative actions.

## Personas
Persona: Cautious Paper Trader. Wants to explore automation safely, avoids live capital, and prioritizes clear risk caps and simple strategies.
Persona: Active Optimizer. Runs frequent backtests, tweaks prompts and risk parameters, and wants rapid feedback on performance trade-offs.
Persona: Platform Operator (CLI user). Manages AWS deployment, runs broker agent commands for admin workflows, and monitors health/alerts.
Persona: Conservative Swing Trader. Trades in slower timeframes, wants strict safety guards, and focuses on capital preservation.

## Expected engagement pattern
1. Platform operator deploys the paper trading stack to AWS and configures secrets.
2. Trader starts in paper mode, runs backtests, and tunes risk limits via Ops API/UI.
3. Broker agent runs as a service; strategy plans are generated on schedule or on request.
4. Monitor execution, judge feedback, and performance metrics via dashboards and reports.
5. Iterate on strategy prompts and risk parameters.
6. Only consider live trading after explicit acknowledgments and safety checks.

## User stories

### Onboarding and setup
Story: As a retail trader, I want a default paper-trading mode so I can experiment without risking real money.
Acceptance criteria: System starts in paper mode by default; live trading is blocked unless explicit acknowledgment is provided; paper mode is visible in Ops API status output.
Success metrics: Zero live orders in default cloud configuration; safety guard test script passes.

Story: As a retail trader, I want clear environment configuration steps so I can safely connect only the services I intend to use.
Acceptance criteria: Setup documentation lists required secrets and environment variables; missing configuration produces actionable error messages; local development can still start in paper mode with .env.example.
Success metrics: AWS deployment boots with Secrets Manager integration; configuration errors are resolved without code changes.

Story: As a platform operator, I want a single CLI entry point (broker agent) for administrative actions so I can manage the system without manual service orchestration.
Acceptance criteria: Broker agent can set preferences, start market streams, and request strategy planning; commands are documented; CLI can run remotely against the AWS deployment.
Success metrics: Operator can complete admin workflows with the broker agent; retail users do not need CLI access.

### Strategy selection and planning
Story: As a retail trader, I want prebuilt strategy archetypes (conservative, mean reversion, trend, breakout, balanced, aggressive) so I can choose a starting point that matches my risk tolerance.
Acceptance criteria: Strategy prompts exist for each archetype; selecting an archetype results in triggers aligned to its category priorities.
Success metrics: Generated plans include the expected dominant trigger categories; archetype selection is documented and repeatable.

Story: As a retail trader, I want the system to generate structured StrategyPlan JSON so I can inspect and reason about the exact triggers being used.
Acceptance criteria: LLM output validates against the StrategyPlan schema; triggers compile deterministically; invalid plans surface readable errors.
Success metrics: Plan validation failure rate stays low; compiled plans are generated without manual fixes.

Story: As a retail trader, I want a daily LLM plan budget and caching so costs stay predictable.
Acceptance criteria: Identical inputs return cached plans; per-day call limits are enforced with explicit errors when exceeded.
Success metrics: LLM call count per day stays within configured budget; cost tracker reports are available per run.

### Trigger evaluation and execution
Story: As a retail trader, I want deterministic trigger evaluation so I can trust that rules execute the same way every time.
Acceptance criteria: Trigger rules compile through the deterministic compiler; rule evaluation uses a constrained DSL; tests cover deterministic replay.
Success metrics: Determinism tests pass; repeated runs on identical data produce identical trade decisions.

Story: As a retail trader, I want entry and exit rules with stop-loss support so I can control downside risk.
Acceptance criteria: Triggers include entry and exit rules; optional stop_loss_pct influences sizing via stop distance.
Success metrics: Orders include stop-distance telemetry when provided; risk snapshots capture allocated vs actual risk.

Story: As a retail trader, I want cooldowns and minimum hold periods to reduce churn and fee drag.
Acceptance criteria: Trigger engine enforces trade cooldown and minimum hold period; blocked entries/exits are logged with reasons.
Success metrics: Overtrading is reduced in backtests; churn-related blocks are visible in daily reports.

### Risk and safety controls
Story: As a retail trader, I want per-trade, per-symbol, portfolio, and daily loss caps so one mistake does not wipe me out.
Acceptance criteria: Risk engine enforces caps on sizing; execution engine blocks trades when caps are hit; block reasons are recorded.
Success metrics: Trades never exceed configured caps; risk block counts appear in daily summaries.

Story: As a retail trader, I want a daily risk budget so I can cap total risk exposure even on high-activity days.
Acceptance criteria: Daily risk budget is tracked and enforced; remaining allowance is reflected in reports.
Success metrics: Daily risk usage stays below 100 percent; budget-related blocks are logged and counted.

Story: As a retail trader, I want explicit acknowledgments before any live trading so I cannot trade real funds by accident.
Acceptance criteria: Live trading requires explicit environment acknowledgment; live requests are blocked without it; blocking is logged.
Success metrics: Safety guard tests pass; no live order attempts succeed without acknowledgment.

### Backtesting and analysis
Story: As a retail trader, I want backtests that use the same logic as live execution so I can trust historical results.
Acceptance criteria: Backtesting uses the same trigger evaluation and risk logic; strategy plans run through the same compile path.
Success metrics: Backtests and paper runs match on shared fixtures; regression tests cover plan execution consistency.

Story: As a retail trader, I want fee and slippage modeling so I can understand realistic performance.
Acceptance criteria: Backtests apply trading fees; live execution uses a cost gate with fee and slippage estimates.
Success metrics: Reports include fee impact fields; cost gate decisions are visible in logs.

Story: As a retail trader, I want performance metrics (Sharpe, drawdown, win rate) so I can compare strategies objectively.
Acceptance criteria: Metrics tools compute Sharpe, drawdown, win rate, and PnL breakdowns; outputs are queryable.
Success metrics: Performance summaries include these metrics for each run; metrics endpoints respond within expected latency.

### Monitoring and iteration
Story: As a retail trader, I want a judge agent that evaluates performance and adjusts prompts so strategy behavior improves over time.
Acceptance criteria: Judge workflows run on schedule; prompt updates are stored with history; feedback is applied to execution agent.
Success metrics: Prompt history shows updates over time; judge evaluations appear in daily summaries.

Story: As a retail trader, I want logs and event traces so I can audit why a trade happened.
Acceptance criteria: Trades emit events and structured logs; event store entries contain trigger IDs and execution reasons.
Success metrics: Logs include trade reason fields; event queries return full trade context.

Story: As a retail trader, I want daily summaries and risk telemetry so I can spot when risk controls are binding.
Acceptance criteria: Daily summaries include risk usage, block reasons, and trade counts; summaries are generated per run.
Success metrics: Daily reports include risk budget usage and block breakdowns; report generation succeeds for each run.

### Limits of current functionality (expectations management)
Story: As a retail trader, I understand that grid trading, arbitrage, and market-making are not supported today.
Acceptance criteria: Documentation explicitly lists unsupported strategy types; no Ops API/UI controls advertise these strategies.
Success metrics: Support inquiries about these strategies are minimized; roadmap references are clear and isolated.

Story: As a retail trader, I understand that margin/shorting and DEX/Phantom integrations are not implemented yet.
Acceptance criteria: Documentation clarifies spot-only execution; shorting support is not marketed; DEX support is labeled roadmap-only.
Success metrics: No production paths attempt margin or DEX execution; roadmap references remain separate from live workflows.

Story: As a retail trader, I understand that fiat on/off-ramp automation is not part of the current system.
Acceptance criteria: Documentation states that fiat transfers are not automated; only internal exchange transfers are supported.
Success metrics: No automated fiat endpoints exist in the live API; operator guidance directs manual off-ramp steps.

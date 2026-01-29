# Runbook: Judge/Strategist Loop Design Gaps

## Overview
Design backlog for judge/strategist loop gaps discovered during emergency-exit test review. These are design issues, not test gaps.

## Working rules
- Track these as design issues or ADRs before writing tests.
- Do not mix these items into emergency-exit test gap branches.
- After issues are filed, delete this runbook.

## Scope (design gaps)
### 1. Judge "competing signals" diagnosis not actionable
The judge can diagnose "competing signals" but this diagnosis cannot be translated into constraint system changes. No mechanism exists to feed this back into trigger conflict detection.

### 2. No mechanism for judge to alter trigger conflict detection logic
The judge can update prompts and thresholds but cannot modify how the trigger engine resolves conflicts between simultaneous signals. This is a structural limitation.

### 3. Emergency exit metrics (count, pct) computation not tested
The computation of emergency exit frequency metrics (count per day, percentage of total exits) is not covered by any test. These metrics feed into the judge's evaluation and could silently produce incorrect values.

## Acceptance
- Design decisions are documented and assigned (issue or ADR) for each gap.
- A follow-up plan exists for any required tests or telemetry changes.

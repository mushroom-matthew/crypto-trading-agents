Run the Phase 8 wiring audit to check which services are connected to the live execution paths.

## Steps

1. Run the audit script:
   ```bash
   uv run python scripts/check_wiring.py
   ```

2. Read the output. It will show a table with columns for each execution-path file and rows for each Phase 8 service. `✅ wired` means the identifier appears in that file. `❌ MISSING` means it does not.

3. For each gap found, report:
   - Which service is unwired
   - Which runbook covers it (R61–R67)
   - Suggested next runbook to implement (lowest number with gaps)

4. If the user asks to fix a specific gap, invoke the relevant runbook skill:
   ```
   /runbook docs/branching/6X-<name>.md
   ```

## Gap → Runbook Mapping

| Gap | Runbook |
|-----|---------|
| PolicyLoopGate, RegimeTransitionDetector, PolicyStateMachineRecord | R61 |
| PlaybookRegistry.list_eligible, originating_plan_id | R62 |
| SetupEventGenerator, AdaptiveTradeManagement, build_episode_record | R63 |
| build_tick_snapshot, StructuralTargetSelector | R64 |
| PositionExitContract enforcement, originating_plan_id | R65 |
| JudgePlanValidationService | R66 |
| Any of the above in backtest runner | R67 |

## Notes
- The script exits 0 if all targets are wired, 1 if any gaps remain.
- After implementing a runbook, re-run this check to confirm the gap is closed.
- `❌ MISSING` means the identifier string is absent from that file — it does not distinguish between "imported but not called" vs. "not imported at all". Read the file to confirm actual call sites.

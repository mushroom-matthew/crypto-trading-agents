Delegate test fixing to Codex. Target: $ARGUMENTS

This skill delegates test running and fixing to OpenAI Codex CLI in full-auto mode, saving Claude's context window for planning and orchestration.

## Step 1: Prepare the Delegation

Determine what to send to Codex based on `$ARGUMENTS`:
- If a specific test path: delegate that test
- If "all" or empty: delegate the full test suite
- If a description (e.g., "risk budget tests"): identify the relevant test files first, then delegate

## Step 2: Run Codex in Full-Auto Mode

Execute Codex with workspace-write sandbox so it can edit files and run tests:

```bash
codex exec \
  --full-auto \
  -o /tmp/codex-test-result.txt \
  -C /home/getzinmw/crypto-trading-agents \
  "Fix the failing test(s). Target: $ARGUMENTS

Instructions:
1. Run the failing test with: uv run pytest $ARGUMENTS -vv
2. Read the test file AND the source file it tests
3. Determine if the fix belongs in the test (wrong fixture/stale assertion) or the source (bug)
4. Apply the minimal fix
5. Re-run the test to verify it passes
6. Run the full suite: uv run pytest --tb=short
7. Report what was wrong, what you fixed, and final test results

Project conventions:
- Pydantic models use extra='forbid' -- ALL required fields must be in fixtures
- Mock OpenAI with SimpleNamespace stubs, not heavy mock libraries
- risk_used=0 is a valid value -- use 'is not None' not truthiness
- Emergency exits must bypass all other constraints
- Do NOT modify test assertions unless source behavior intentionally changed" 2>/dev/null
```

## Step 3: Collect Results

Read the output file:
```bash
cat /tmp/codex-test-result.txt
```

## Step 4: Present Summary to User

Summarize what Codex did:
1. **Root cause**: What was wrong (stale test, source bug, fixture issue, environment issue)
2. **Files changed**: List each file and what changed
3. **Test results**: Pass/fail counts after the fix
4. **Remaining failures**: Any tests still failing

## Step 5: Review Changes

Run `git diff` to show the user exactly what Codex changed. Ask if they want to:
- **Accept all**: Keep Codex's changes
- **Accept with modifications**: Keep some, adjust others (Claude applies the adjustments)
- **Reject**: Revert Codex's changes with `git checkout -- <files>`

## Step 6: Update Memory

If a new debugging pattern was discovered, update `memory/debugging-patterns.md`.

## Notes
- Codex runs in a sandbox with workspace-write access -- it CAN edit files but cannot access network/secrets
- The `--full-auto` flag means Codex won't ask for approval during execution
- If Codex times out on large test suites, narrow the scope to specific test files
- Output is written to /tmp/codex-test-result.txt for reliable capture
- Always review Codex's changes before committing -- it may make different tradeoffs than expected

Fix the failing test: $ARGUMENTS

Follow this sequence:

## Step 1: Reproduce
- Run the test with full verbosity: `uv run pytest $ARGUMENTS -vv`
- Read the complete error output carefully

## Step 2: Understand
- Read the test file to understand what it's asserting
- Read the source file(s) it tests to understand current behavior
- Check if the test uses fixtures from conftest.py

## Step 3: Diagnose
Determine which category this failure falls into:
- **Stale test**: Test expectations don't match intentional source changes → update the test
- **Source bug**: Source behavior is wrong → fix the source
- **Fixture issue**: Test setup is incomplete (missing required Pydantic fields, wrong mock) → fix the fixture
- **Environment issue**: Needs DB/Temporal/external service → mark with appropriate skip decorator

## Step 4: Fix
- Apply the minimal fix for the identified category
- Do NOT modify test assertions unless source behavior intentionally changed
- For Pydantic validation errors: check ALL required fields in the model definition
- For mock issues: ensure mocks match the current function signatures

## Step 5: Verify
- Re-run the specific test: `uv run pytest $ARGUMENTS -vv`
- Run related tests in the same file
- Run the full suite: `uv run pytest` to check for regressions

## Step 6: Report
- Summarize: what broke, why, and what was fixed
- If this is a recurring pattern, update `memory/debugging-patterns.md`

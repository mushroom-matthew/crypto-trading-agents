Implement the runbook at: $ARGUMENTS

Follow this exact sequence — do NOT skip or reorder steps:

## Phase 1: Setup
1. Read the runbook file completely before making any changes
2. Run `git status` to ensure a clean working tree (stash or commit if dirty)
3. Create the git branch specified in the runbook's Git Workflow section
4. Switch to the new branch

## Phase 2: Implement
5. Implement ALL changes listed in the Implementation section
6. For each file changed, read it first to understand existing patterns
7. When editing schemas, grep for ALL downstream consumers before changing field names
8. Include 3+ lines of surrounding context in Edit tool old_string for uniqueness

## Phase 3: Verify
9. Run the test plan from the runbook with `-vv` flag
10. If any test fails, fix it before proceeding (do NOT skip)
11. Run the full test suite: `uv run pytest` to check for regressions
12. Paste test output into the runbook's Test Evidence section (never write "N/A")

## Phase 4: Document
13. Fill the Human Verification Evidence section
14. Fill the Change Log entries with actual file paths and descriptions
15. Update `memory/session-log.md` with what was accomplished

## Phase 5: Commit
16. Stage specific files (not `git add .`)
17. Commit using the runbook's Git Workflow commit message format
18. Do NOT push unless the user explicitly asks

## Error Recovery
- If tests fail: fix the source, not the test assertions (unless behavior intentionally changed)
- If Edit tool fails: re-read the file and use more surrounding context
- If a schema change breaks things: grep for all usages before fixing

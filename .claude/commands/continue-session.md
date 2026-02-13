Continue work from a previous session. Context: $ARGUMENTS

**Note:** All `memory/` paths below refer to the auto-memory directory (shown in your system prompt as your persistent auto memory directory), NOT the repo root.

## Step 1: Recover State
1. Read `session-log.md` from your auto-memory directory for the most recent entry
2. Run `git log --oneline -10` to see recent commits
3. Run `git status` to check for uncommitted work in progress
4. Run `git diff --stat` to see any staged/unstaged changes
5. Check current branch: `git branch --show-current`

## Step 2: Load Context
6. If a runbook or plan is referenced, read it
7. If there's a specific branch in progress, check its diff from main: `git diff main...HEAD --stat`
8. Read any files that have uncommitted changes

## Step 3: Summarize & Confirm
9. Present a summary to the user:
   - What was completed last session
   - What's currently in progress (uncommitted changes)
   - What remains to be done
   - Any blockers or known issues
10. Wait for user confirmation before proceeding with implementation

## Step 4: Resume Work
11. Pick up where the previous session left off
12. Follow the relevant workflow (runbook, plan, etc.)
13. At the end, update `session-log.md` in your auto-memory directory

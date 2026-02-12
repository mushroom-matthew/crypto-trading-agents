Review code changes using Codex before committing. Scope: $ARGUMENTS

This skill delegates code review to OpenAI Codex CLI, keeping review tokens out of the Claude context window.

## Step 1: Determine Review Scope

Based on the user's argument, determine which review mode to use:
- If `$ARGUMENTS` is empty or "uncommitted": review all staged/unstaged changes
- If `$ARGUMENTS` is a branch name: review changes against that branch
- If `$ARGUMENTS` is a commit SHA: review that specific commit

## Step 2: Run Codex Review

Execute the appropriate command. IMPORTANT: Always use `codex exec` (never interactive `codex`), always suppress stderr with `2>/dev/null` to avoid context bloat from thinking tokens.

**For uncommitted changes (default):**
```bash
codex exec review --uncommitted --json 2>/dev/null
```

**For branch diff (e.g., "main"):**
```bash
codex exec review --base main --json 2>/dev/null
```

**For a specific commit:**
```bash
codex exec review --commit <SHA> --json 2>/dev/null
```

If you want domain-specific review context, append custom instructions:
```bash
codex exec review --uncommitted "Focus on: correctness of risk/P&L calculations, emergency exit bypass logic, Pydantic schema compatibility, and falsy-zero bugs (risk_used=0 is valid). Flag security issues in trading execution paths." 2>/dev/null
```

## Step 3: Parse and Present Results

Read the Codex output and present findings to the user organized by severity:
1. **Critical** -- Must fix before committing (correctness bugs, security issues)
2. **Warning** -- Should fix (performance, maintainability)
3. **Info** -- Optional improvements

For each finding, show:
- File and line range
- What the issue is
- Suggested fix

## Step 4: Act on Findings

Ask the user what to do:
- **Fix all**: Apply fixes for all findings
- **Fix critical only**: Only address critical/warning items
- **Dismiss**: Proceed without changes
- **Re-review**: Run Codex again after making manual changes

If fixing, apply the changes yourself (Claude), then optionally re-run the review to verify.

## Notes
- Codex uses model `gpt-5.2-codex` by default (from ~/.codex/config.toml)
- The project is already trusted in Codex config
- For large diffs (20+ files), consider reviewing in batches by passing specific file paths
- If Codex returns an error, check that OPENAI_API_KEY is set in the environment

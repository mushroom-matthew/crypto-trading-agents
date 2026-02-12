Delegate heavy analysis to Codex. Target: $ARGUMENTS

This skill delegates token-heavy analysis (backtest outputs, large logs, performance reports) to Codex, keeping Claude's context window clean for decision-making.

## Step 1: Determine Analysis Target

Based on `$ARGUMENTS`, identify what needs analysis:
- **File path**: Analyze the contents of that file
- **Backtest run ID**: Find and analyze the backtest results
- **"last backtest"**: Find the most recent backtest output
- **Description**: Identify relevant files to analyze

## Step 2: Prepare the Analysis Prompt

Build a targeted prompt based on what's being analyzed. Include domain context so Codex understands the trading system.

## Step 3: Run Codex with Structured Output

```bash
codex exec \
  --sandbox read-only \
  --output-schema .claude/codex-schemas/analyze-output.json \
  -o /tmp/codex-analysis.json \
  -C /home/getzinmw/crypto-trading-agents \
  "Analyze: $ARGUMENTS

You are analyzing output from a crypto trading backtest system. Key concepts:
- TradeSet: complete position lifecycle (entry to exit), tracks WAC-based P&L
- TriggerCondition: rule-based entry/exit signals with categories (mean_reversion, emergency_exit, etc.)
- JudgeFeedback: scoring system (0-100) that adjusts strategy parameters
- Risk budget: daily allocation that resets at midnight UTC; risk_used=0 is valid
- PolicyDecisionRecord: per-bar audit showing weight progression and overrides

Analyze the data and return structured findings. Focus on:
1. Trading performance (returns, Sharpe, drawdown, win rate)
2. Risk behavior (budget utilization, position sizing, emergency exits)
3. Strategy effectiveness (which triggers fire, category distribution)
4. Anomalies or issues (no-trade periods, stuck positions, judge feedback loops)
5. Actionable recommendations for the next backtest run

Read the relevant files and provide your analysis." 2>/dev/null
```

## Step 4: Parse Structured Output

Read the JSON output:
```bash
cat /tmp/codex-analysis.json
```

Parse the structured response which contains:
- `analysis.summary` -- High-level overview
- `key_findings[]` -- Specific findings with evidence and severity
- `recommendations[]` -- Prioritized action items
- `metrics` -- Extracted numeric metrics

## Step 5: Present to User

Format the analysis as a concise report:

1. **Summary** (from analysis.summary)
2. **Key Findings** (sorted by severity: critical > important > informational)
   - Finding + evidence for each
3. **Metrics** (formatted as a table)
4. **Recommendations** (sorted by priority: high > medium > low)
   - Each with action + rationale

## Step 6: Bridge to Action

Ask the user if they want to:
- **Implement recommendations**: Claude takes the top recommendations and creates a plan
- **Run another backtest**: With adjusted parameters based on findings
- **Deep dive**: Pick a specific finding and investigate further
- **Save to memory**: Record key learnings in `memory/debugging-patterns.md` or `memory/session-log.md`

## Notes
- Uses `--sandbox read-only` since analysis doesn't need to write files
- Uses `--output-schema` for structured JSON responses that Claude can reliably parse
- For very large files (>1MB), consider extracting relevant sections first
- Codex has access to read all project files, including CLAUDE.md for project context
- If the structured output fails to parse, fall back to reading the raw text output

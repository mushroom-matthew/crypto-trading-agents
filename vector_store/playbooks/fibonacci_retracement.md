---
title: Fibonacci Retracement
type: playbook
regimes: [bull, bear]
tags: [fibonacci, pullback, support_resistance, mean_reversion]
identifiers: [fib_236, fib_382, fib_500, fib_618, fib_786, close, rsi_14]
hypothesis: "Entries in the fib_500–fib_618 golden pocket (close between fib_500 and
  fib_618) with rsi_14 turning up from 40–50 achieve win_rate > 58% in confirmed bull
  regimes. The golden pocket has statistically higher edge than the fib_382 or fib_786
  levels across the same regime filter."
min_sample_size: 20
playbook_id: fibonacci_retracement
---
# Fibonacci Retracement

Use Fibonacci levels to identify high-probability pullback entries.

Patterns
- Shallow pullback (trending): `close` holds above `fib_382` in a bull regime. Enter
  long on bounce with `rsi_14` turning up from 40–50 range.
- Deep pullback (reversal risk): `close` reaches `fib_618` or `fib_786`. Valid entry
  only if price stabilizes and `rsi_14` shows bullish divergence.
- Golden pocket: `close` between `fib_618` and `fib_500` is the highest-probability
  reversal zone in strong trends. Prefer this level for entries.

Notes
- Fibonacci levels are most reliable in trending regimes; avoid in range/choppy markets.
- Use `fib_236` as a take-profit target for mean-reversion trades (first scale-out).
- Breaks below `fib_786` typically signal trend failure; exit or reverse.
- Fibonacci levels require a clear prior swing (swing_high → swing_low) to anchor.
  If no clear prior swing exists, skip Fibonacci-based entries.

## Research Trade Attribution
<!-- Conditions that tag a research trade to this playbook in the research budget -->
- Entry rule includes any `fib_` identifier, OR
- Research hypothesis tests golden pocket (fib_500–fib_618) vs. other retracement levels, OR
- Entry trigger category is `mean_reversion` with fib identifier present

## Validation Evidence
<!-- Auto-updated by PlaybookOutcomeAggregator. Do not edit manually. -->
status: insufficient_data
n_trades: 0
win_rate: null
avg_r: null
median_bars_to_outcome: null
win_rate_by_fib_level: null
last_updated: null
judge_notes: null

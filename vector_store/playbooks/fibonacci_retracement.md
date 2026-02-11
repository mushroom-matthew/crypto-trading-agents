---
title: Fibonacci Retracement
type: playbook
regimes: [bull, bear]
tags: [fibonacci, pullback, support_resistance]
identifiers: [fib_236, fib_382, fib_500, fib_618, fib_786, close, rsi_14]
---
# Fibonacci Retracement

Use Fibonacci levels to identify high-probability pullback entries.

Patterns
- Shallow pullback (trending): `close` holds above `fib_382` in a bull regime. Enter long on bounce with `rsi_14` turning up from 40-50 range.
- Deep pullback (reversal risk): `close` reaches `fib_618` or `fib_786`. Valid entry only if price stabilizes and `rsi_14` shows bullish divergence.
- Golden pocket: `close` between `fib_618` and `fib_500` is the highest-probability reversal zone in strong trends.

Notes
- Fibonacci levels are most reliable in trending regimes; avoid in range/choppy markets.
- Use `fib_236` as a take-profit target for mean-reversion trades.
- Breaks below `fib_786` typically signal trend failure; exit or reverse.

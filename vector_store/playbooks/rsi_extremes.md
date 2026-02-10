---
title: RSI Extremes
type: playbook
regimes: [range, bull, bear]
tags: [rsi, mean_reversion]
identifiers: [rsi_14, close, sma_short, sma_medium, bollinger_upper, bollinger_lower]
---
# RSI Extremes

Use RSI extremes to time entries when price is stretched.

Patterns
- Oversold bounce: `rsi_14 < 30` with `close` stabilizing above `sma_short` or `bollinger_lower`.
- Overbought fade: `rsi_14 > 70` with `close` losing `sma_short` or failing above `bollinger_upper`.

Notes
- Confirm with regime: in bull regimes, oversold bounces have higher odds; in bear regimes, overbought fades are cleaner.

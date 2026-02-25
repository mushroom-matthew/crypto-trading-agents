---
title: Range Mean Revert
type: strategy
regimes: [range]
tags: [mean_reversion, range]
identifiers: [close, bollinger_upper, bollinger_lower, bollinger_middle, rsi_14, sma_medium, atr_14, vol_state, trend_state]
template_file: mean_reversion
---
# Range Mean Revert

Context
- Sideways regime with `trend_state` = "sideways" and `vol_state` low/normal.
- Price oscillates around `bollinger_middle` or `sma_medium`.

Entry patterns
- Buy dips: `close < bollinger_lower` and `rsi_14 < 35`.
- Sell rallies: `close > bollinger_upper` and `rsi_14 > 65`.

Exit / risk reduce
- Mean reversion target: `close` returns near `bollinger_middle` or `sma_medium`.
- Volatility shift: `atr_14` rising and `vol_state` moves to "high".

Regime alerts
- Breakout: consecutive closes beyond `bollinger_upper` or `bollinger_lower`.

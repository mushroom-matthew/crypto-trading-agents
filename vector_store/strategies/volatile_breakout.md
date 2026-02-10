---
title: Volatile Breakout
type: strategy
regimes: [volatile]
tags: [breakout, volatility]
identifiers: [close, atr_14, realized_vol_short, realized_vol_medium, bollinger_upper, bollinger_lower, vol_state, trend_state]
---
# Volatile Breakout

Context
- High volatility regime where `vol_state` is "high" or "extreme".
- Expect larger moves and wider stops; reduce position sizing.

Entry patterns
- Breakout continuation: `close > bollinger_upper` and `realized_vol_short` elevated.
- Downside break: `close < bollinger_lower` and `realized_vol_short` elevated.

Exit / risk reduce
- Volatility mean reversion: `realized_vol_short` drops materially or `close` re-enters the band.
- Risk reduction when `vol_state` flips to "extreme".

Regime alerts
- Volatility contraction: `realized_vol_short` falls below `realized_vol_medium`.

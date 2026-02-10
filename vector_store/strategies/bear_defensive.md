---
title: Bear Defensive
type: strategy
regimes: [bear]
tags: [defensive, risk_off, mean_reversion]
identifiers: [close, sma_medium, sma_long, ema_medium, rsi_14, macd_hist, atr_14, vol_state, trend_state]
---
# Bear Defensive

Context
- Downtrend conditions with `trend_state` = "downtrend" and weak momentum.
- Prioritize capital preservation and selective shorts or quick bounces only.

Entry patterns
- Weak rally fade: `close < ema_medium` and `macd_hist < 0` with `rsi_14` between 35-50.
- Oversold bounce (short-term only): `rsi_14 < 30` and `close` stabilizes above `sma_medium` for a bar.

Exit / risk reduce
- Cover on momentum loss: `macd_hist > 0` or `close > sma_medium`.
- Reduce exposure if `vol_state` moves to "extreme".

Regime alerts
- Potential regime shift: `close > sma_long` or `trend_state` flips to "sideways".

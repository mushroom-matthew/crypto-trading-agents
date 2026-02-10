---
title: Bull Trending
type: strategy
regimes: [bull]
tags: [trend, momentum, breakout]
identifiers: [close, sma_medium, sma_long, ema_long, rsi_14, macd_hist, atr_14, donchian_upper_short, vol_state, trend_state]
---
# Bull Trending

Context
- Favors uptrends where `trend_state` is "uptrend" and `vol_state` is not "extreme".
- Price tends to hold above `sma_medium` or `sma_long` and momentum is positive.

Entry patterns
- Trend continuation: `close > sma_medium` and `macd_hist > 0` and `rsi_14` between 50-70.
- Breakout: `close > donchian_upper_short` and `atr_14` rising relative to its recent range.

Exit / risk reduce
- Momentum fade: `macd_hist < 0` or `close < sma_medium`.
- Exhaustion: `rsi_14 > 75` or `close < ema_long` after an extended run.

Regime alerts
- Trend break: `close < sma_long` or `trend_state` flips to "sideways" for multiple bars.

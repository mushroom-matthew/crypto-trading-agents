---
title: Uncertain Wait
type: strategy
regimes: [uncertain]
tags: [wait, defensive]
identifiers: [trend_state, vol_state, rsi_14, macd_hist, close, sma_medium]
---
# Uncertain Wait

Context
- Conflicting signals or weak trend conviction.
- Prioritize capital preservation; wait for clarity.

Behavior
- Stance should often be "wait" with no triggers.
- Allow only the cleanest setups if signals align strongly.

Regime alerts
- Trend confirmation: `close > sma_medium` and `macd_hist > 0` with `rsi_14` above 55.
- Risk-off confirmation: `close < sma_medium` and `macd_hist < 0` with `rsi_14` below 45.

---
title: MACD Divergence
type: playbook
regimes: [bull, bear, range]
tags: [macd, momentum]
identifiers: [macd, macd_signal, macd_hist, close, sma_medium, rsi_14]
---
# MACD Divergence

Use MACD momentum shifts to confirm trend continuation or exhaustion.

Patterns
- Bullish momentum: `macd > macd_signal` and `macd_hist > 0` while `close > sma_medium`.
- Bearish momentum: `macd < macd_signal` and `macd_hist < 0` while `close < sma_medium`.

Notes
- When `rsi_14` is neutral (40-60), MACD crossovers carry more weight than RSI alone.

---
title: MACD Divergence
type: playbook
regimes: [bull, bear, range]
tags: [macd, momentum, trend_continuation, exhaustion]
identifiers: [macd, macd_signal, macd_hist, close, sma_medium, rsi_14]
hypothesis: "macd > macd_signal with close > sma_medium identifies trend continuation
  entries where price advances at least 1R within 24 bars at win_rate > 55%. Bearish
  MACD crossovers (macd < macd_signal) with close < sma_medium have symmetric short-side
  edge in bear regimes."
min_sample_size: 15
playbook_id: macd_divergence
---
# MACD Divergence

Use MACD momentum shifts to confirm trend continuation or exhaustion.

Patterns
- Bullish momentum: `macd > macd_signal` and `macd_hist > 0` while `close > sma_medium`.
  Use for trend continuation entries; hold while `macd_hist > 0`.
- Bearish momentum: `macd < macd_signal` and `macd_hist < 0` while `close < sma_medium`.
  Short-side entry confirmation in bear regimes.
- Divergence (advanced): price makes a new high but `macd_hist` is lower than the prior
  high — momentum is weakening even as price extends. This is an exhaustion signal,
  not a continuation signal.

Notes
- When `rsi_14` is neutral (40–60), MACD crossovers carry more weight than RSI alone.
- `macd_hist` slope is more informative than the zero-cross: a rising `macd_hist` below
  zero means momentum is turning bullish before the crossover.
- Avoid using MACD as the sole entry criterion in choppy/range regimes — it generates
  excessive false crossovers. Require `close > sma_medium` as a regime filter.

## Research Trade Attribution
<!-- Conditions that tag a research trade to this playbook in the research budget -->
- Entry rule includes `macd_hist`, `macd`, or `macd_signal`, OR
- Entry trigger category is `trend_continuation` with MACD identifier present, OR
- `macd_hist > 0` at entry bar and entry is long direction

## Validation Evidence
<!-- Auto-updated by PlaybookOutcomeAggregator. Do not edit manually. -->
status: insufficient_data
n_trades: 0
win_rate: null
avg_r: null
median_bars_to_outcome: null
win_rate_by_regime: null
last_updated: null
judge_notes: null

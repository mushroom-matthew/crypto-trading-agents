---
title: RSI Extremes
type: playbook
regimes: [range, bull, bear]
tags: [rsi, mean_reversion, oversold, overbought]
identifiers: [rsi_14, close, sma_short, sma_medium, bollinger_upper, bollinger_lower]
hypothesis: "rsi_14 < 30 in range or bull regime (not extreme downtrend) generates
  mean-reversion entries with win_rate > 52% when confirmed by price stabilizing above
  sma_short or bollinger_lower. In bear regimes, overbought fades (rsi_14 > 70) carry
  stronger edge than oversold bounces."
min_sample_size: 20
playbook_id: rsi_extremes
---
# RSI Extremes

Use RSI extremes to time entries when price is stretched.

Patterns
- Oversold bounce: `rsi_14 < 30` with `close` stabilizing above `sma_short` or
  `bollinger_lower`. Best in range or bull regimes.
- Overbought fade: `rsi_14 > 70` with `close` losing `sma_short` or failing above
  `bollinger_upper`. Best in bear or range regimes.

Notes
- Confirm with regime: in bull regimes, oversold bounces have higher odds; in bear
  regimes, overbought fades are cleaner.
- RSI extremes in strong trending regimes can persist for many bars — do not treat
  `rsi_14 > 70` as a sell signal in a confirmed uptrend.
- Combine with `macd_divergence` playbook: RSI extreme + MACD momentum cross is a
  stronger signal than RSI alone.

## Research Trade Attribution
<!-- Conditions that tag a research trade to this playbook in the research budget -->
- Entry rule includes `rsi_14` with threshold < 35 or > 65, OR
- Entry trigger category is `mean_reversion` with `rsi_14` in entry rule, OR
- `rsi_14 < 30` or `rsi_14 > 70` at time of entry bar

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

---
title: Bollinger Squeeze
type: playbook
regimes: [range, volatile]
tags: [bollinger, volatility, compression, breakout]
identifiers: [bollinger_upper, bollinger_lower, bollinger_middle, atr_14, realized_vol_short,
              close, bb_bandwidth_pct_rank, compression_flag, expansion_flag, breakout_confirmed]
hypothesis: "BB bandwidth in the bottom quintile (compression_flag=1) followed by a close
  outside the band generates profitable breakout trades at win_rate > 55% with avg_r > 0.8
  within 48 bars, across range and volatile regimes."
min_sample_size: 20
playbook_id: bollinger_squeeze
---
# Bollinger Squeeze

Use volatility contraction to anticipate breakout moves.

Patterns
- Compression: band width narrows (price oscillates near `bollinger_middle`) with low
  `realized_vol_short` and `compression_flag = 1`.
- Expansion trigger: `close` breaks above `bollinger_upper` or below `bollinger_lower`
  with `atr_14` rising and `breakout_confirmed = 1`.

Notes
- Avoid forcing direction; use follow-through confirmation after the band break.
- Compression must precede the expansion — entering on expansion alone (without prior
  compression) reduces edge significantly.
- Combine with `volume_confirmation` playbook: low-volume breakouts out of Bollinger
  bands have materially higher false-breakout rates.

## Research Trade Attribution
<!-- Conditions that tag a research trade to this playbook in the research budget -->
- `compression_flag > 0` at entry bar, OR
- `bb_bandwidth_pct_rank < 0.30` at entry bar, OR
- Entry identifier includes `bollinger_upper` or `bollinger_lower` in exit rule

## Validation Evidence
<!-- Auto-updated by PlaybookOutcomeAggregator. Do not edit manually. -->
status: insufficient_data
n_trades: 0
win_rate: null
avg_r: null
median_bars_to_outcome: null
last_updated: null
judge_notes: null

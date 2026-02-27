---
title: Range Long (Buy the Dip)
type: strategy
direction: long
regimes: [range]
tags: [mean_reversion, range, long, support]
identifiers: [close, bollinger_lower, bollinger_middle, rsi_14, sma_medium, atr_14, vol_state, trend_state, candle_strength]
template_file: range_long
---

Context: Sideways regime, price at lower band (price_position_in_range <= 0.30) — buy into support with mean-reversion target.
Entry: `close < bollinger_lower` and `rsi_14 < 38` and `vol_state` not "extreme" and `trend_state` == "range".
Stop: 0.5× ATR below entry (tight — support level defines invalidation).
Target: `bollinger_middle` or `sma_medium` (mean reversion, not a trend trade). Mandatory — must specify `target_anchor_type` or `target_price_abs`.
Invalidation: close below stop, or `trend_state` flips to "downtrend".

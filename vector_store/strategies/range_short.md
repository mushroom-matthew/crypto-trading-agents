---
title: Range Short (Sell the Rally)
type: strategy
direction: short
regimes: [range]
tags: [mean_reversion, range, short, resistance]
identifiers: [close, bollinger_upper, bollinger_middle, rsi_14, sma_medium, atr_14, vol_state, trend_state, candle_strength]
template_file: range_short
---

Context: Sideways regime, price at upper band (price_position_in_range >= 0.70) — sell into resistance with mean-reversion target.
Entry: `close > bollinger_upper` and `rsi_14 > 62` and `vol_state` not "extreme" and `trend_state` == "range".
Stop: 0.5× ATR above entry.
Target: `bollinger_middle` or `sma_medium`. Mandatory — must specify `target_anchor_type` or `target_price_abs`.
Invalidation: close above stop, or `trend_state` flips to "uptrend".

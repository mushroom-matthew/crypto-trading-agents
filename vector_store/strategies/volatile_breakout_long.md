---
title: Volatile Breakout Long
type: strategy
direction: long
regimes: [volatile]
tags: [breakout, volatility, long, momentum]
identifiers: [close, atr_14, realized_vol_short, realized_vol_medium, bollinger_upper, vol_state, trend_state, candle_strength, volume_multiple]
template_file: volatile_breakout_long
---

Context: High-vol regime, price near/above range top (price_position_in_range >= 0.65) — ride upside momentum.
Entry: `close > bollinger_upper` and `realized_vol_short` elevated and `trend_state` not "downtrend" and `volume_multiple > 1.5`.
Stop: 1.5–2× ATR below entry (wide stops required for vol regime).
Target: 2× ATR measured move from entry, or `htf_daily_high`. Mandatory — must specify `target_anchor_type` or `target_price_abs`.
Hold: position valid while `vol_state` remains high/extreme and `trend_state` not "downtrend".

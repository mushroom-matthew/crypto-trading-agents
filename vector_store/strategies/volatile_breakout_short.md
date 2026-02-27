---
title: Volatile Breakout Short
type: strategy
direction: short
regimes: [volatile]
tags: [breakout, volatility, short, breakdown]
identifiers: [close, atr_14, realized_vol_short, realized_vol_medium, bollinger_lower, vol_state, trend_state, candle_strength, volume_multiple]
template_file: volatile_breakout_short
---

Context: High-vol regime, price near/below range bottom (price_position_in_range <= 0.35) — ride downside momentum.
Entry: `close < bollinger_lower` and `realized_vol_short` elevated and `trend_state` not "uptrend" and `volume_multiple > 1.5`.
Stop: 1.5–2× ATR above entry.
Target: 2× ATR measured move downward, or `htf_daily_low`. Mandatory — must specify `target_anchor_type` or `target_price_abs`.
Hold: position valid while `vol_state` remains high/extreme and `trend_state` not "uptrend".
